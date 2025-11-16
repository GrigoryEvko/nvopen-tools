// Function: sub_210BBD0
// Address: 0x210bbd0
//
__int64 __fastcall sub_210BBD0(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi

  a1[1] = a2[31];
  v5 = a2[29];
  a1[3] = a2;
  a1[2] = v5;
  a1[4] = a3;
  a1[5] = a4;
  sub_1E69F60(v5, a2[32]);
  return sub_1ED7320(a1 + 6, a2[32]);
}
