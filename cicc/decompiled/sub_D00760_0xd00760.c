// Function: sub_D00760
// Address: 0xd00760
//
char __fastcall sub_D00760(
        unsigned __int8 *a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6)
{
  char result; // al
  _QWORD v10[7]; // [rsp+8h] [rbp-38h] BYREF

  result = sub_CF7060(a1);
  if ( result )
  {
    result = sub_D62CA0(a1, v10, a4, a5, (a6 << 16) | 0x100u, 0);
    if ( result )
      return a2 > v10[0];
  }
  return result;
}
