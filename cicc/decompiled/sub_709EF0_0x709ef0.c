// Function: sub_709EF0
// Address: 0x709ef0
//
void __fastcall sub_709EF0(
        const __m128i *a1,
        unsigned __int8 a2,
        _OWORD *a3,
        unsigned __int8 a4,
        _DWORD *a5,
        _DWORD *a6)
{
  __int128 v9; // rax
  __int64 v10; // r9

  *a5 = 0;
  *a6 = 0;
  if ( a2 == a4 )
  {
    *a3 = _mm_loadu_si128(a1);
  }
  else
  {
    *(_QWORD *)&v9 = sub_709B30(a2, a1);
    sub_709750(v9, a4, a3, a5, v10);
  }
}
