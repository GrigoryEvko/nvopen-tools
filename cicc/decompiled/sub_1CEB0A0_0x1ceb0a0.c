// Function: sub_1CEB0A0
// Address: 0x1ceb0a0
//
__int64 __fastcall sub_1CEB0A0(
        __int64 *a1,
        __int64 a2,
        __m128 si128,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rax
  __int64 v11; // rdx

  v10 = *(_QWORD *)(a2 + 80);
  if ( v10 == a2 + 72 )
    return 0;
  v11 = 0;
  do
  {
    v10 = *(_QWORD *)(v10 + 8);
    ++v11;
  }
  while ( v10 != a2 + 72 );
  if ( v11 == 1 )
    return 0;
  else
    return sub_1CE7DD0(a1, a2, si128, a4, a5, a6, a7, a8, a9, a10);
}
