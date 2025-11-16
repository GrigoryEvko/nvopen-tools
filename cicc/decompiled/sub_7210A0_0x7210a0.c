// Function: sub_7210A0
// Address: 0x7210a0
//
__sighandler_t sub_7210A0()
{
  __sighandler_t result; // rax

  result = (__sighandler_t)dword_4F07908;
  if ( dword_4F07908 )
  {
    signal(2, qword_4F07900);
    signal(15, qword_4F078F8);
    result = signal(25, qword_4F078F0);
    dword_4F07908 = 0;
    qword_4F07900 = 0;
    qword_4F078F8 = 0;
    qword_4F078F0 = 0;
  }
  return result;
}
