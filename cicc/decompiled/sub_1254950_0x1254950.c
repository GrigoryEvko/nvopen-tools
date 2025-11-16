// Function: sub_1254950
// Address: 0x1254950
//
unsigned __int64 *__fastcall sub_1254950(unsigned __int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // r13
  _QWORD v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = a4;
  sub_1255430(v6, a2 + 8, *(_QWORD *)(a2 + 56), a4, a3);
  if ( (v6[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v6[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  else
  {
    *(_QWORD *)(a2 + 56) += v4;
    *a1 = 1;
    return a1;
  }
}
