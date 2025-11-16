// Function: sub_8E3410
// Address: 0x8e3410
//
__int64 __fastcall sub_8E3410(__int64 a1, __int64 a2, __m128i **a3)
{
  __m128i *v4; // rax
  unsigned int v5; // r8d

  v4 = sub_8E3390(a1);
  v5 = 0;
  *a3 = v4;
  if ( v4 != (__m128i *)a1 && (v5 = 1, a1) && v4 && dword_4F07588 )
    return (*(_QWORD *)(a1 + 32) == 0) | (unsigned __int8)(v4[2].m128i_i64[0] != *(_QWORD *)(a1 + 32));
  else
    return v5;
}
