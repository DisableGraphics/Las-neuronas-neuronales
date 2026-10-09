#include "arclas.hpp"
#include <gtkmm.h>
#include <fstream>
#include <sstream>

class MyWindow : public Gtk::Window
{
	public:
		MyWindow();
	private:
		void load_labels();
		Gtk::Box bocs;
		Gtk::Image img;
		Gtk::Label result_label;
		Gtk::Button upload_btn;
		std::shared_ptr<Arclas> net;
		Gtk::FileChooserDialog fgd;
		std::unordered_map<int, std::string> labels;
		torch::Device device;
};

void MyWindow::load_labels() {
	std::ifstream artist_mappings{"arclas/artist_mapping"};
	std::string line;
	while(std::getline(artist_mappings, line)) {
		std::stringstream ss{line};
		int label;
		std::string artist;
		ss >> label;
		ss >> artist;
		labels[label] = std::move(artist);
	}
}

MyWindow::MyWindow() : fgd("Open Image", Gtk::FileChooserDialog::Action::OPEN, false), 
device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
	load_labels();
	
	net = std::make_shared<Arclas>(labels.size());
	net->to(device);
	set_title("Basic application");
	set_default_size(400, 400);
	img.set_size_request(256, 256);
	img.set_vexpand();
	img.set_hexpand();

	bocs.set_orientation(Gtk::Orientation::VERTICAL);

	bocs.set_vexpand();
	bocs.set_valign(Gtk::Align::FILL);

	bocs.append(result_label);
	bocs.append(img);
	bocs.append(upload_btn);

	result_label.set_valign(Gtk::Align::START);
	img.set_valign(Gtk::Align::FILL);
	upload_btn.set_valign(Gtk::Align::END);

	upload_btn.set_label("Upload");

	set_child(bocs);

	fgd.add_button("Cancel", 1);
	fgd.add_button("Ok", 0);

	fgd.set_transient_for(*this);

	fgd.signal_response().connect([&](int response) {
		if(response == 0) {
			 	// oki doki
				const auto& file = fgd.get_file();
				const auto filename = file->get_parse_name();
				auto pix = Gdk::Pixbuf::create_from_file(filename);

				pix = pix->scale_simple(256, 256, Gdk::InterpType::BILINEAR);

				assert(pix->get_n_channels() == 3);

				img.set(pix);
				img.set_pixel_size(256);

				auto raw = pix->get_pixels();

				const int rowstride = pix->get_rowstride();

				auto x = torch::from_blob(
					pix->get_pixels(),
					{256, rowstride},
					torch::kUInt8
				).clone();

				x = x
					.view({256, rowstride})
					.slice(1, 0, 256 * 3)
					.view({256, 256, 3})
					.permute({2, 0, 1})
					.unsqueeze(0)
					.to(device, torch::kFloat32);

				auto pred = net->forward(x);
				auto cls = torch::argmax(pred).item<int>();
				result_label.set_label(std::format("Artist: {}", labels[cls]));
		}
		fgd.hide();
	});

	upload_btn.signal_clicked().connect([this]() {
		fgd.show();
	});
}

int main(int argc, char** argv) {
	auto app = Gtk::Application::create("org.gtkmm.examples.base");
  	return app->make_window_and_run<MyWindow>(argc, argv);
}