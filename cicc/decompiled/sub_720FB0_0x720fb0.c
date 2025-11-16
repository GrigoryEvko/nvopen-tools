// Function: sub_720FB0
// Address: 0x720fb0
//
int sub_720FB0()
{
  _DWORD *v0; // rax

  v0 = &dword_4D04198;
  if ( dword_4D04198 == 1 )
    LODWORD(v0) = fprintf(
                    qword_4F07510,
                    "{\"version\":\"2.1.0\",\"$schema\":\"https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/S"
                    "chemata/sarif-schema-2.1.0.json\",\"runs\":[{\"tool\":{\"driver\":{\"name\":\"EDG CPFE\",\"version\""
                    ":\"%s\",\"organization\":\"Edison Design Group\",\"fullName\":\"Edison Design Group C/C++ Front End "
                    "- %s\",\"informationUri\":\"https://edg.com/c\"}},\"columnKind\":\"unicodeCodePoints\",\"results\":[",
                    "6.6",
                    "6.6");
  return (int)v0;
}
