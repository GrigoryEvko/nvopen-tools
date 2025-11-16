// Function: sub_887500
// Address: 0x887500
//
_QWORD *__fastcall sub_887500(unsigned __int8 a1, __m128i *a2, int a3, __int64 a4, _QWORD *a5)
{
  _QWORD *v7; // r12
  char v8; // al
  _BYTE *v9; // rcx
  _BOOL4 v10; // edx
  __int64 v12; // rax

  v7 = sub_87EBB0(a1, a2->m128i_i64[0], &a2->m128i_i64[1]);
  *((_DWORD *)v7 + 10) = *(_DWORD *)(a4 + 40);
  a2[1].m128i_i64[1] = (__int64)v7;
  a2[1].m128i_i8[0] &= ~1u;
  v8 = *(_BYTE *)(a4 + 81) & 0x10;
  if ( a3 )
  {
LABEL_8:
    v10 = 0;
    v9 = 0;
    if ( !v8 )
    {
      v9 = *(_BYTE **)(a4 + 64);
      v10 = v9 != 0;
    }
    goto LABEL_4;
  }
  if ( v8 )
  {
    if ( (unsigned __int8)sub_877F80(a4) == 1 )
    {
      sub_6851C0(0x195u, &a2->m128i_i32[2]);
      *a2 = _mm_loadu_si128(xmmword_4F06660);
      a2[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
      a2[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
      v12 = *(_QWORD *)dword_4F07508;
      a2[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
      a2[1].m128i_i8[1] |= 0x20u;
      a2->m128i_i64[1] = v12;
      *((_BYTE *)v7 + 81) |= 0x20u;
      *a5 = 0;
      return v7;
    }
    v8 = *(_BYTE *)(a4 + 81) & 0x10;
    goto LABEL_8;
  }
  v9 = *(_BYTE **)(a4 + 64);
  v10 = v9 != 0;
LABEL_4:
  *a5 = sub_887160((__int64)v7, a4, v10, v9);
  return v7;
}
