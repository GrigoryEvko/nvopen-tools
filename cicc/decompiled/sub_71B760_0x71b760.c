// Function: sub_71B760
// Address: 0x71b760
//
__m128i *__fastcall sub_71B760(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rbx
  char v8; // al
  __int64 i; // r12
  __int64 v10; // rax
  __int64 v11; // r12
  char v12; // al
  char v13; // dl
  bool v14; // al
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rcx
  __m128i *result; // rax
  __int64 v19; // rax
  __m128i *v20; // rax
  __int64 v21; // rdi
  __m128i *v22; // rax
  unsigned int v23; // esi
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rax
  int v27; // [rsp+8h] [rbp-78h]
  char v28; // [rsp+Fh] [rbp-71h]
  _BYTE v29[112]; // [rsp+10h] [rbp-70h] BYREF

  v27 = a4;
  if ( (_DWORD)a4 )
  {
    v28 = ((*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) >> 1) ^ 1) & 1;
    v7 = *(_QWORD *)(a3 + 8);
    if ( unk_4F0690C )
    {
      if ( (a1[2].m128i_i8[9] & 0x7F) != 0 && !(unsigned int)sub_8D3410(*(_QWORD *)(a3 + 16)) )
        *(_DWORD *)(a3 + 32) = *(_DWORD *)(a3 + 32) & 0xFFFC07FF
                             | (((a1[2].m128i_i8[9] | (unsigned __int8)(*(_DWORD *)(a3 + 32) >> 11)) & 0x7F) << 11);
      v23 = *(_DWORD *)(a3 + 32);
      if ( (v23 & 0x3F800) != 0 )
        v7 = sub_73C570(v7, (v23 >> 11) & 0x7F, -1);
    }
  }
  else
  {
    v28 = 0;
    v7 = a1[1].m128i_i64[0];
  }
  v8 = *(_BYTE *)(v7 + 140);
  for ( i = v7; v8 == 12; v8 = *(_BYTE *)(i + 140) )
    i = *(_QWORD *)(i + 160);
  if ( (unsigned __int8)(v8 - 9) <= 2u && dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(i) )
    sub_8AE000(i);
  if ( (*(_BYTE *)(i + 141) & 0x20) != 0 )
  {
    v25 = (unsigned int)sub_67F240();
    sub_685A50(v25, (const __m128i *)a1[2].m128i_i32, (FILE *)i, 8u);
    v26 = sub_72C930(v25);
    *(_QWORD *)(a3 + 8) = v26;
    v7 = v26;
  }
  if ( (a1[2].m128i_i8[10] & 1) != 0 && !*(_DWORD *)(a3 + 36) )
  {
    v24 = sub_735FB0(dword_4D03B80, 2, (unsigned int)dword_4F04C64, a4);
    *(_BYTE *)(v24 + 89) |= 1u;
    v11 = v24;
    *(_WORD *)(v24 + 175) |= 0x4020u;
  }
  else
  {
    v10 = sub_735FB0(v7, a1[2].m128i_u8[8], 0xFFFFFFFFLL, a4);
    *(_BYTE *)(v10 + 169) |= 0x80u;
    v11 = v10;
    *(_BYTE *)(v10 + 89) |= 1u;
    *(_QWORD *)(v10 + 256) = v7;
    sub_72FBE0(v10);
  }
  v12 = (32 * (*(_BYTE *)(a3 + 33) & 1)) | *(_BYTE *)(v11 + 175) & 0xDF;
  *(_BYTE *)(v11 + 175) = v12;
  v13 = *(_BYTE *)(a3 + 33);
  *(_QWORD *)(v11 + 256) = a2;
  *(_BYTE *)(v11 + 175) = (32 * v13) & 0x40 | v12 & 0xBF;
  v14 = 0;
  if ( dword_4D047EC )
    v14 = (unsigned int)sub_8DD010(v7) != 0;
  *(_BYTE *)(v11 + 173) = (2 * v14) | *(_BYTE *)(v11 + 173) & 0xFD;
  v15 = a1->m128i_i64[1];
  if ( !HIDWORD(qword_4F077B4) )
  {
    if ( v15 )
      goto LABEL_17;
LABEL_35:
    *(_BYTE *)(v11 + 88) &= ~4u;
    if ( dword_4F04C3C )
    {
      if ( !a1[2].m128i_i32[0] )
        goto LABEL_27;
    }
    else
    {
      sub_8699D0(v11, 7, a1[3].m128i_i64[0]);
      if ( !a1[2].m128i_i32[0] )
        goto LABEL_27;
    }
    *(_QWORD *)(v11 + 72) = sub_7274B0(*(_BYTE *)(v11 - 8) & 1);
    goto LABEL_27;
  }
  if ( !v15 )
    goto LABEL_35;
  if ( (*(_BYTE *)(v15 + 82) & 4) != 0 )
  {
    sub_6851C0(0x64u, (_DWORD *)(v15 + 48));
    goto LABEL_35;
  }
LABEL_17:
  sub_878710(a1->m128i_i64[1], v29);
  if ( v27 )
  {
    v16 = sub_647630(7u, (__int64)v29, (unsigned int)dword_4F04C5C, 0);
    v15 = v16;
    if ( (a1[2].m128i_i8[10] & 4) != 0 )
      *(_BYTE *)(v16 + 84) |= 0x40u;
  }
  else
  {
    sub_87E690(v15, 7);
    if ( (*(_BYTE *)(v15 + 82) & 4) == 0 )
      *(_BYTE *)(v15 + 83) &= ~0x40u;
  }
  if ( (*(_BYTE *)(a3 + 33) & 2) != 0 )
    *(_BYTE *)(v15 + 84) |= 0x20u;
  *(_QWORD *)(v15 + 88) = v11;
  sub_877D80(v11, v15);
  v17 = 0;
  if ( !v28 )
    v17 = a1[3].m128i_i64[0];
  sub_8756F0(3, v15, v15 + 48, v17);
  if ( (a1[2].m128i_i8[10] & 0x10) != 0 )
    *(_BYTE *)(v11 + 90) |= 1u;
  sub_8756B0(v15);
LABEL_27:
  if ( (*(_BYTE *)(v11 + 175) & 0x40) != 0 )
    *(_BYTE *)(v11 + 89) |= 1u;
  if ( (a1[2].m128i_i8[10] & 1) == 0 || *(_DWORD *)(a3 + 36) )
  {
    v19 = *(_QWORD *)(v11 + 8);
    *(_QWORD *)(v11 + 128) = a3;
    *(_QWORD *)(a3 + 24) = v19;
    v20 = *(__m128i **)(a3 + 72);
    if ( v20 )
    {
      *v20 = _mm_loadu_si128(a1 + 6);
      v20[1] = _mm_loadu_si128(a1 + 4);
      v20[2] = _mm_loadu_si128(a1 + 5);
    }
    v21 = *(_QWORD *)(v11 + 128);
    if ( *(_QWORD *)(v21 + 64) )
    {
      v22 = (__m128i *)sub_5CF790(v21);
      sub_5CEC90(v22, v11, 7);
    }
  }
  result = *(__m128i **)(v11 + 72);
  if ( result )
  {
    *result = _mm_loadu_si128(a1 + 6);
    result[1] = _mm_loadu_si128(a1 + 4);
    result[2] = _mm_loadu_si128(a1 + 5);
  }
  return result;
}
