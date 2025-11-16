// Function: sub_1481F90
// Address: 0x1481f90
//
__int64 __fastcall sub_1481F90(_QWORD *a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v4; // rax
  int v5; // eax

  v4 = sub_1481F60(a1, a2, a3, a4);
  LOBYTE(v5) = sub_14562D0(v4);
  return v5 ^ 1u;
}
