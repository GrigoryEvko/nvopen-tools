// Function: sub_2FF4380
// Address: 0x2ff4380
//
_QWORD *__fastcall sub_2FF4380(__int64 a1, const char *a2, _BYTE *a3, __int64 **a4, __int64 *a5)
{
  int v8; // edx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v11; // r12
  __int64 v12; // rax
  size_t v13; // rax
  __int64 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r9
  _QWORD *v19; // r15
  __int64 v20; // rcx
  __int64 v21; // rsi
  const __m128i *v22; // rbx
  __int64 v23; // rdx
  __int64 v24; // r10
  __int64 v25; // r8
  unsigned __int64 v26; // r11
  __int64 v27; // rcx
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rsi
  unsigned __int64 v30; // rdx
  __m128i *v31; // rcx
  __m128i v32; // xmm1
  __int8 v33; // dl
  __int64 v34; // rdi
  char *v36; // rbx
  __int64 v37; // [rsp+8h] [rbp-88h]
  __int64 v38; // [rsp+10h] [rbp-80h]
  __int64 v40; // [rsp+18h] [rbp-78h]
  _QWORD v41[6]; // [rsp+20h] [rbp-70h] BYREF
  char v42; // [rsp+50h] [rbp-40h]

  *(_QWORD *)a1 = &unk_49DC150;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 12) &= 0x8000u;
  *(_WORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 80) = 0x100000000LL;
  *(_DWORD *)(a1 + 8) = v8;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = a1 + 128;
  *(_QWORD *)(a1 + 112) = 1;
  *(_DWORD *)(a1 + 120) = 0;
  *(_BYTE *)(a1 + 124) = 1;
  v11 = sub_C57470();
  v12 = *(unsigned int *)(a1 + 80);
  if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v12 + 1, 8u, v9, v10);
    v12 = *(unsigned int *)(a1 + 80);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v12) = v11;
  ++*(_DWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_BYTE *)(a1 + 160) = 0;
  *(_QWORD *)a1 = &unk_4A2D8E8;
  *(_QWORD *)(a1 + 192) = a1 + 208;
  *(_QWORD *)(a1 + 200) = 0x800000000LL;
  *(_QWORD *)(a1 + 144) = &unk_4A2D840;
  *(_QWORD *)(a1 + 184) = a1;
  *(_QWORD *)(a1 + 176) = &unk_4A2D898;
  *(_QWORD *)(a1 + 168) = &unk_4A2D860;
  *(_QWORD *)(a1 + 680) = nullsub_1702;
  *(_QWORD *)(a1 + 672) = sub_2FEDEB0;
  v13 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v13);
  *(_BYTE *)(a1 + 12) = (32 * (*a3 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v14 = *a4;
  v15 = **a4;
  *(_BYTE *)(a1 + 160) = 1;
  *(_QWORD *)(a1 + 136) = v15;
  v16 = *a5;
  *(_QWORD *)(a1 + 152) = *v14;
  v17 = a5[1];
  *(_QWORD *)(a1 + 40) = v16;
  *(_QWORD *)(a1 + 48) = v17;
  sub_C53130(a1);
  v19 = (_QWORD *)qword_5023860[0];
  v40 = a1 + 192;
  if ( qword_5023860[0] )
  {
    do
    {
      v20 = v19[4];
      v21 = v19[3];
      v41[4] = &unk_4A2D840;
      v22 = (const __m128i *)v41;
      v23 = v19[5];
      v24 = v19[1];
      v42 = 1;
      v25 = v19[2];
      v26 = *(unsigned int *)(a1 + 204);
      v41[3] = v20;
      v27 = *(unsigned int *)(a1 + 200);
      v28 = *(_QWORD *)(a1 + 192);
      v41[2] = v21;
      v41[5] = v23;
      v29 = v27 + 1;
      v41[0] = v24;
      v30 = v27;
      v41[1] = v25;
      if ( v27 + 1 > v26 )
      {
        if ( v28 > (unsigned __int64)v41 )
        {
          v37 = v25;
          v38 = v24;
LABEL_13:
          sub_2FF41A0(v40, v29, v30, v27, v25, v18);
          v27 = *(unsigned int *)(a1 + 200);
          v24 = v38;
          v28 = *(_QWORD *)(a1 + 192);
          v25 = v37;
          LODWORD(v30) = *(_DWORD *)(a1 + 200);
          goto LABEL_5;
        }
        v37 = v25;
        v38 = v24;
        v30 = v28 + 56 * v27;
        if ( (unsigned __int64)v41 >= v30 )
          goto LABEL_13;
        v36 = (char *)v41 - v28;
        sub_2FF41A0(v40, v29, v30, v27, v25, v18);
        v28 = *(_QWORD *)(a1 + 192);
        v27 = *(unsigned int *)(a1 + 200);
        v25 = v37;
        v24 = v38;
        v22 = (const __m128i *)&v36[v28];
        LODWORD(v30) = *(_DWORD *)(a1 + 200);
      }
LABEL_5:
      v31 = (__m128i *)(v28 + 56 * v27);
      if ( v31 )
      {
        v32 = _mm_loadu_si128(v22 + 1);
        *v31 = _mm_loadu_si128(v22);
        v31[1] = v32;
        v31[2].m128i_i64[1] = v22[2].m128i_i64[1];
        v33 = v22[3].m128i_i8[0];
        v31[2].m128i_i64[0] = (__int64)&unk_4A2D840;
        v31[3].m128i_i8[0] = v33;
        LODWORD(v30) = *(_DWORD *)(a1 + 200);
      }
      v34 = *(_QWORD *)(a1 + 184);
      *(_DWORD *)(a1 + 200) = v30 + 1;
      sub_C52F90(v34, v24, v25);
      v19 = (_QWORD *)*v19;
    }
    while ( v19 );
  }
  qword_5023870 = (_QWORD *)(a1 + 168);
  return qword_5023860;
}
