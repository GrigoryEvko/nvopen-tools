// Function: sub_889970
// Address: 0x889970
//
__int64 __fastcall sub_889970(char *s, __int64 a2, unsigned __int16 a3, __int64 *a4)
{
  __int64 *v5; // rdi
  __int64 v6; // r12
  __int64 v8; // rax
  char i; // r15
  __int64 v10; // r12
  unsigned __int64 v11; // rdi
  bool v12; // al
  bool v13; // al
  size_t v15; // rax
  char *src; // [rsp+0h] [rbp-80h]
  __int64 v17[2]; // [rsp+10h] [rbp-70h] BYREF
  __m128i v18; // [rsp+20h] [rbp-60h]
  __m128i v19; // [rsp+30h] [rbp-50h]
  __m128i v20; // [rsp+40h] [rbp-40h]

  v5 = a4;
  v6 = a2;
  v8 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
  for ( i = (*(_BYTE *)(v8 + 9) >> 1) & 7; *(_BYTE *)(v6 + 140) == 12; v6 = *(_QWORD *)(v6 + 160) )
    ;
  if ( !a4 )
  {
    src = s;
    v17[0] = _mm_loadu_si128(xmmword_4F06660).m128i_u64[0];
    v18 = _mm_loadu_si128(&xmmword_4F06660[1]);
    v19 = _mm_loadu_si128(&xmmword_4F06660[2]);
    v20 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v17[1] = *(_QWORD *)&dword_4F077C8;
    v15 = strlen(s);
    sub_878540(src, v15, v17);
    v5 = v17;
    v8 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
  }
  *(_BYTE *)(v8 + 9) = *(_BYTE *)(v8 + 9) & 0xF1 | 6;
  v10 = sub_8800F0((__int64)v5, v6);
  *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C + 9) = *(_BYTE *)(qword_4F04C68[0]
                                                                           + 776LL * (int)dword_4F04C5C
                                                                           + 9)
                                                                & 0xF1
                                                                | (2 * (i & 7));
  v11 = *(_QWORD *)v10;
  *(_BYTE *)(v10 + 81) = (4 * (dword_4F077C4 == 2)) | *(_BYTE *)(v10 + 81) & 0xFB;
  sub_8896D0(v11);
  *(_WORD *)(*(_QWORD *)(v10 + 88) + 176LL) = a3;
  if ( a3 == 16688 )
    goto LABEL_21;
  if ( a3 <= 0x4130u )
  {
    if ( a3 <= 3u )
    {
      v12 = a3 != 0;
LABEL_9:
      *(_BYTE *)(*(_QWORD *)(v10 + 88) + 193LL) = *(_BYTE *)(*(_QWORD *)(v10 + 88) + 193LL) & 0xFB | (4 * v12);
      goto LABEL_10;
    }
    if ( a3 != 82 )
    {
      *(_BYTE *)(*(_QWORD *)(v10 + 88) + 193LL) &= ~4u;
      v13 = 1;
      if ( a3 == 4686 )
        goto LABEL_14;
LABEL_10:
      v13 = 1;
      if ( (unsigned __int16)(a3 - 4138) <= 1u )
        goto LABEL_14;
      goto LABEL_11;
    }
LABEL_21:
    v12 = 1;
    goto LABEL_9;
  }
  if ( (unsigned __int16)(a3 - 25771) <= 1u )
    goto LABEL_21;
  *(_BYTE *)(*(_QWORD *)(v10 + 88) + 193LL) &= ~4u;
  if ( a3 == 24949 )
  {
    v13 = 1;
    goto LABEL_14;
  }
LABEL_11:
  v13 = a3 == 10203 || a3 == 25767;
  if ( !v13 )
  {
    v13 = 1;
    if ( a3 != 3421 )
      v13 = (unsigned __int16)(a3 - 24994) <= 0x302u;
  }
LABEL_14:
  *(_BYTE *)(*(_QWORD *)(v10 + 88) + 199LL) = (2 * v13) | *(_BYTE *)(*(_QWORD *)(v10 + 88) + 199LL) & 0xFD;
  return v10;
}
