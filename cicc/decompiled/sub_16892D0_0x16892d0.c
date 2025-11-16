// Function: sub_16892D0
// Address: 0x16892d0
//
int __fastcall sub_16892D0(char *format, __gnuc_va_list arg)
{
  if ( qword_4F9F878 )
    return vfprintf(qword_4F9F878, format, arg);
  else
    return vfprintf(stderr, format, arg);
}
