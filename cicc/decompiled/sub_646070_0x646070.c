// Function: sub_646070
// Address: 0x646070
//
void __fastcall sub_646070(__int64 a1, __int64 a2, __m128i *a3)
{
  __int8 v4; // al
  __int64 *v5; // rax
  __int64 v6; // rax
  __m128i v7; // xmm3
  __int64 v8; // rax

  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  if ( (a3[1].m128i_i8[1] & 0x20) == 0 )
  {
    v4 = a3[1].m128i_i8[0];
    if ( (v4 & 0x10) != 0 )
    {
      v5 = *(__int64 **)(a1 + 168);
      if ( (v5[2] & 1) != 0 || (v6 = *v5) != 0 && ((*(_BYTE *)(v6 + 35) & 1) == 0 || *(_QWORD *)v6) )
      {
        sub_6851C0(343, &a3->m128i_u64[1]);
LABEL_10:
        *a3 = _mm_loadu_si128(xmmword_4F06660);
        a3[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
        a3[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
        v7 = _mm_loadu_si128(&xmmword_4F06660[3]);
        v8 = *(_QWORD *)dword_4F07508;
        a3[1].m128i_i8[1] |= 0x20u;
        a3[3] = v7;
        a3->m128i_i64[1] = v8;
      }
    }
    else if ( (v4 & 8) != 0 && (unsigned int)sub_645720(a3[3].m128i_u8[8], a1, a2, (__int64)&a3->m128i_i64[1]) )
    {
      goto LABEL_10;
    }
  }
}
