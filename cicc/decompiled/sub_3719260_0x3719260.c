// Function: sub_3719260
// Address: 0x3719260
//
unsigned __int64 *__fastcall sub_3719260(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_12558B0(v6, a2 + 8, *(_QWORD *)(a2 + 56), a3, a4);
  if ( (v6[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v6[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  else
  {
    *(_QWORD *)(a2 + 56) += a4;
    *a1 = 1;
    return a1;
  }
}
