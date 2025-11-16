// Function: sub_1254040
// Address: 0x1254040
//
unsigned __int64 *__fastcall sub_1254040(
        unsigned __int64 *a1,
        _QWORD *a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 *a5)
{
  unsigned __int64 v8; // r12
  __int64 v10[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( a3 > (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 40LL))(a2) )
  {
    sub_1253E40(v10, 3u);
  }
  else
  {
    if ( (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 40LL))(a2) >= a3 + a4 )
    {
LABEL_6:
      v8 = a2[2] + a3;
      a5[1] = a4;
      *a5 = v8;
      *a1 = 1;
      return a1;
    }
    sub_1253E40(v10, 1u);
  }
  if ( (v10[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    goto LABEL_6;
  *a1 = v10[0] & 0xFFFFFFFFFFFFFFFELL | 1;
  return a1;
}
