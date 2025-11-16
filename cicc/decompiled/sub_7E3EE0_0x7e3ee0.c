// Function: sub_7E3EE0
// Address: 0x7e3ee0
//
void __fastcall sub_7E3EE0(__int64 a1)
{
  __int64 **v1; // r13
  char *v2; // rbx
  __int64 *v4; // r12
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdx
  const char *v8; // rsi
  __int64 *i; // r15
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  char v13; // al
  __int64 v14; // r12
  char v15; // al
  _QWORD *v16; // r12
  __int64 v17; // rax
  char v18; // al
  char v19; // dl
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r15
  char *v23; // rsi
  size_t v24; // rax
  char *v25; // rax
  __m128i *v26; // rax
  __m128i v27; // xmm4
  __int64 j; // rbx
  __int64 *k; // rbx
  char v30; // al
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  char v35; // al
  __int64 v36; // rdx
  int v37; // esi
  __int64 v38; // rdx
  __int64 v39; // rax
  unsigned __int64 v40; // r15
  __int64 v41; // r12
  char *v42; // rax
  _QWORD *v43; // [rsp+0h] [rbp-70h]
  __int16 v44; // [rsp+Ah] [rbp-66h]
  int v45; // [rsp+Ch] [rbp-64h]
  __int64 v46; // [rsp+10h] [rbp-60h]
  char *srca; // [rsp+18h] [rbp-58h]
  char *src; // [rsp+18h] [rbp-58h]
  __int16 v49; // [rsp+26h] [rbp-4Ah] BYREF
  __int64 v50; // [rsp+28h] [rbp-48h] BYREF
  __int64 v51; // [rsp+30h] [rbp-40h] BYREF
  __int64 v52[7]; // [rsp+38h] [rbp-38h] BYREF

  v1 = *(__int64 ***)(a1 + 168);
  v2 = (char *)v1[26];
  if ( v2 )
    return;
  v45 = dword_4F07508[0];
  v44 = dword_4F07508[1];
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(a1 + 64);
  sub_7E08C0(a1, 0);
  v4 = v1[3];
  if ( v4 )
  {
    sub_7E3EE0(v4[5]);
    v5 = v4[5];
    v6 = *(_QWORD *)(v5 + 168);
    v7 = v5;
    if ( (*(_BYTE *)(v6 + 109) & 0x10) != 0 )
      v7 = *(_QWORD *)(v6 + 208);
    v8 = "__v_";
    if ( (v4[12] & 2) == 0 )
      v8 = "__b_";
    sub_7E1520(v5, v8, v7, v4[13], a1);
  }
  for ( i = *v1; i; i = (__int64 *)*i )
  {
    sub_7E3EE0(i[5]);
    v10 = i[5];
    v11 = *(_QWORD *)(v10 + 168);
    v12 = v10;
    if ( (*(_BYTE *)(v11 + 109) & 0x10) != 0 )
      v12 = *(_QWORD *)(v11 + 208);
    v13 = *((_BYTE *)i + 96);
    if ( (v13 & 2) == 0 && (v13 & 0x21) == 1 && v1[3] != i )
      sub_7E1520(v10, "__b_", v12, i[13], a1);
  }
  v50 = 0;
  v14 = *(_QWORD *)(a1 + 168);
  if ( ((*(_BYTE *)(a1 + 176) & 0x50) != 0 || (*(_BYTE *)(a1 + 177) & 1) != 0) && !*(_QWORD *)(v14 + 192) )
    sub_7E34E0(a1, 0, &v50);
  v49 = 0;
  v51 = 0;
  v52[0] = 0;
  sub_7E3970(a1, 0, &v51, v52, &v49);
  *(_QWORD *)(v14 + 224) = v51;
  v15 = *(_BYTE *)(a1 + 176);
  if ( (v15 & 0x50) != 0 && !v1[10] )
  {
    v40 = (unsigned __int64)v1[9];
    v41 = sub_7E1DC0();
    v42 = (char *)sub_7E1510(7);
    strcpy(v42, "__vptr");
    sub_7DF750((__int64)v42, v41, v40, a1, 0);
    v15 = *(_BYTE *)(a1 + 176);
  }
  v46 = *(_QWORD *)(a1 + 168);
  if ( (v15 & 0x10) == 0
    && (!unk_4F068A8 || *(_BYTE *)(a1 + 140) == 11 || *(_QWORD *)(*(_QWORD *)(a1 + 168) + 32LL) == *(_QWORD *)(a1 + 128)) )
  {
    for ( j = *(_QWORD *)(a1 + 160); j; j = *(_QWORD *)(j + 112) )
    {
      if ( (*(_BYTE *)(j + 146) & 4) != 0 )
        sub_7DFBB0(j);
    }
    v16 = (_QWORD *)a1;
    goto LABEL_47;
  }
  v16 = sub_7E16B0(10);
  v17 = v16[21];
  *(_BYTE *)(v17 + 109) |= 8u;
  v43 = (_QWORD *)v17;
  sub_810A10(a1, v16);
  v16[8] = *(_QWORD *)(a1 + 64);
  v18 = *(_BYTE *)(a1 + 89) & 4 | *((_BYTE *)v16 + 89) & 0xFB;
  *((_BYTE *)v16 + 89) = v18;
  v19 = *(_BYTE *)(a1 + 89);
  v16[5] = 0;
  *((_BYTE *)v16 + 89) = v19 & 1 | v18 & 0xFE;
  if ( (*(_BYTE *)(a1 + 89) & 2) != 0 )
    v20 = sub_72F070(a1);
  else
    v20 = *(_QWORD *)(a1 + 40);
  sub_72EE40((__int64)v16, 6u, v20);
  *((_BYTE *)v16 + 88) |= 4u;
  *((_DWORD *)v16 + 46) = *(_DWORD *)(a1 + 184);
  v21 = *(_QWORD *)(a1 + 112);
  if ( !v21 )
  {
    v35 = *(_BYTE *)(a1 + 89);
    if ( (v35 & 4) != 0 || (v36 = *(_QWORD *)(a1 + 40)) != 0 && *(_BYTE *)(v36 + 28) == 3 || (v35 & 1) == 0 )
    {
      sub_7365B0((__int64)v16, -1);
      goto LABEL_27;
    }
    v37 = dword_4F04C64;
    if ( dword_4F04C64 >= 0 )
    {
      v38 = qword_4F04C68[0] + 776LL * dword_4F04C64 + 32;
      while ( 1 )
      {
        v39 = *(_QWORD *)(v38 - 8);
        if ( !v39 )
          v39 = v38;
        if ( a1 == *(_QWORD *)(v39 + 32) )
          break;
        --v37;
        v38 -= 776;
        if ( v37 == -1 )
          goto LABEL_26;
      }
      sub_7365B0((__int64)v16, v37);
      goto LABEL_27;
    }
  }
LABEL_26:
  v16[14] = v21;
  *(_QWORD *)(a1 + 112) = v16;
LABEL_27:
  v22 = *(_QWORD *)(a1 + 160);
  if ( v22 )
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)(v22 + 146) & 4) != 0 )
        sub_7DFBB0(v22);
      v23 = *(char **)(v22 + 8);
      if ( v23 )
      {
        srca = *(char **)(v22 + 8);
        v24 = strlen(srca);
        v25 = (char *)sub_7E1510(v24 + 1);
        v23 = strcpy(v25, srca);
      }
      v26 = (__m128i *)sub_725D60();
      *v26 = _mm_loadu_si128((const __m128i *)v22);
      v26[1] = _mm_loadu_si128((const __m128i *)(v22 + 16));
      v26[2] = _mm_loadu_si128((const __m128i *)(v22 + 32));
      v26[3] = _mm_loadu_si128((const __m128i *)(v22 + 48));
      v26[4] = _mm_loadu_si128((const __m128i *)(v22 + 64));
      v26[5] = _mm_loadu_si128((const __m128i *)(v22 + 80));
      v26[6] = _mm_loadu_si128((const __m128i *)(v22 + 96));
      v26[7] = _mm_loadu_si128((const __m128i *)(v22 + 112));
      v26[8] = _mm_loadu_si128((const __m128i *)(v22 + 128));
      v26[9] = _mm_loadu_si128((const __m128i *)(v22 + 144));
      v26[10] = _mm_loadu_si128((const __m128i *)(v22 + 160));
      v26[11] = _mm_loadu_si128((const __m128i *)(v22 + 176));
      v27 = _mm_loadu_si128((const __m128i *)(v22 + 192));
      v26->m128i_i64[1] = (__int64)v23;
      v26[12] = v27;
      v26[2].m128i_i64[1] = *(_QWORD *)(v16[21] + 152LL);
      v26[5].m128i_i8[8] &= ~0x80u;
      src = (char *)v26;
      v26[6].m128i_i64[1] = sub_5CF190(*(const __m128i **)(v22 + 104));
      *((_QWORD *)src + 14) = 0;
      if ( v2 )
        *((_QWORD *)v2 + 14) = src;
      else
        v16[20] = src;
      v22 = *(_QWORD *)(v22 + 112);
      if ( !v22 )
        break;
      v2 = src;
    }
  }
  *((_BYTE *)v16 + 141) &= ~0x20u;
  v16[16] = *(_QWORD *)(v46 + 32);
  *((_DWORD *)v16 + 34) = *(_DWORD *)(v46 + 40);
  *((_BYTE *)v16 + 176) = *(_BYTE *)(a1 + 176) & 0x40 | v16[22] & 0xBF;
  *((_BYTE *)v16 + 177) = *(_BYTE *)(a1 + 177) & 1 | *((_BYTE *)v16 + 177) & 0xFE;
  v43[4] = *(_QWORD *)(v46 + 32);
  v43[9] = *(_QWORD *)(v46 + 72);
  v34 = *(_QWORD *)(v46 + 80);
  v43[26] = a1;
  v43[10] = v34;
  *(_BYTE *)(v46 + 109) |= 0x10u;
LABEL_47:
  *(_QWORD *)(v46 + 208) = v16;
  if ( (*(_BYTE *)(a1 + 176) & 0x10) != 0 )
  {
    for ( k = v1[2]; k; k = (__int64 *)k[2] )
    {
      v30 = *((_BYTE *)k + 96);
      if ( (v30 & 2) != 0 && v1[3] != k && (v30 & 0x28) == 0 )
      {
        v31 = k[5];
        v32 = *(_QWORD *)(v31 + 168);
        v33 = v31;
        if ( (*(_BYTE *)(v32 + 109) & 0x10) != 0 )
          v33 = *(_QWORD *)(v32 + 208);
        sub_7E1520(v31, "__v_", v33, k[13], a1);
      }
    }
  }
  dword_4F07508[0] = v45;
  LOWORD(dword_4F07508[1]) = v44;
}
