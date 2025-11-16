// Function: sub_1563B90
// Address: 0x1563b90
//
__int64 __fastcall sub_1563B90(__int64 *a1, __int64 *a2, int a3)
{
  __int64 v5; // rbx
  __m128i v6; // [rsp+0h] [rbp-80h] BYREF
  _QWORD *v7; // [rsp+18h] [rbp-68h]

  if ( !(unsigned __int8)sub_155EE10((__int64)a1, a3) )
    return *a1;
  sub_1563030(&v6, *a1);
  sub_1560700(&v6, a3);
  v5 = sub_1560BF0(a2, &v6);
  sub_155CC10(v7);
  return v5;
}
