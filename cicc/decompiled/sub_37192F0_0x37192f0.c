// Function: sub_37192F0
// Address: 0x37192f0
//
unsigned __int64 *__fastcall sub_37192F0(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rax
  char v6; // [rsp+7h] [rbp-29h] BYREF
  unsigned __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_37192D0(v7, a2, a3, a4);
  if ( (v7[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v7[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  else
  {
    v6 = 0;
    sub_3719260(v7, a2, (__int64)&v6, 1);
    v4 = v7[0] | 1;
    if ( (v7[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
      v4 = 1;
    *a1 = v4;
    return a1;
  }
}
