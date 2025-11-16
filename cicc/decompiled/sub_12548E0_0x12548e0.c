// Function: sub_12548E0
// Address: 0x12548e0
//
unsigned __int64 *__fastcall sub_12548E0(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  _QWORD v5[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_1255540(v5, a2 + 8, *(_QWORD *)(a2 + 56), a3);
  if ( (v5[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v5[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  else
  {
    *(_QWORD *)(a2 + 56) += *(_QWORD *)(a3 + 8);
    *a1 = 1;
    return a1;
  }
}
