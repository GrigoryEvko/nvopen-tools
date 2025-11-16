// Function: sub_5D37C0
// Address: 0x5d37c0
//
__int64 sub_5D37C0()
{
  __int64 result; // rax
  int *v1; // rax

  if ( putc(10, stream) == -1 )
  {
    v1 = __errno_location();
    sub_6866A0(1700, (unsigned int)*v1);
  }
  result = (unsigned int)dword_4CF7F3C;
  if ( dword_4CF7F3C )
    ++dword_4CF7F44;
  dword_4CF7F40 = 0;
  return result;
}
