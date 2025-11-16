// Function: sub_2276DC0
// Address: 0x2276dc0
//
unsigned __int64 __fastcall sub_2276DC0(
        __int64 a1,
        const char *a2,
        _DWORD *a3,
        int *a4,
        __int64 *a5,
        _DWORD **a6,
        __int64 *a7)
{
  int v9; // edx
  __int64 *v10; // rbx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  size_t v14; // rax
  int v15; // edx
  char v16; // al
  __int64 v17; // rdx
  __int64 v18; // rax
  _DWORD *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r14
  int v22; // edx
  __int64 v23; // rsi
  const __m128i *v24; // r12
  __int64 v25; // rcx
  __int64 v26; // r9
  __int64 v27; // r8
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rcx
  unsigned __int64 v31; // r11
  int v32; // esi
  __m128i *v33; // rdx
  __m128i v34; // xmm1
  __int8 v35; // cl
  __int64 v36; // rdi
  char *v38; // r12
  __int64 v39; // [rsp+0h] [rbp-80h]
  __int64 v41; // [rsp+8h] [rbp-78h]
  __int64 v43; // [rsp+10h] [rbp-70h]
  __int64 v45; // [rsp+18h] [rbp-68h]
  _QWORD v46[5]; // [rsp+20h] [rbp-60h] BYREF
  int v47; // [rsp+48h] [rbp-38h]
  char v48; // [rsp+4Ch] [rbp-34h]

  *(_QWORD *)a1 = &unk_49DC150;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(), 1u);
  *(_DWORD *)(a1 + 12) &= 0x8000u;
  *(_WORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 80) = 0x100000000LL;
  *(_DWORD *)(a1 + 8) = v9;
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
  v10 = sub_C57470();
  v13 = *(unsigned int *)(a1 + 80);
  if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v13 + 1, 8u, v11, v12);
    v13 = *(unsigned int *)(a1 + 80);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v13) = v10;
  ++*(_DWORD *)(a1 + 80);
  *(_DWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_BYTE *)(a1 + 156) = 0;
  *(_QWORD *)a1 = &unk_4A08650;
  *(_QWORD *)(a1 + 144) = &unk_4A085E0;
  *(_QWORD *)(a1 + 168) = a1;
  *(_QWORD *)(a1 + 160) = &unk_4A08600;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = nullsub_812;
  *(_QWORD *)(a1 + 592) = sub_226E2B0;
  v14 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v14);
  v15 = *a4;
  v43 = a1 + 176;
  v16 = *(_BYTE *)(a1 + 12) & 0x87 | (8 * (v15 & 3)) | (32 * (*a3 & 3));
  v17 = *a5;
  *(_BYTE *)(a1 + 12) = v16;
  v18 = a5[1];
  *(_QWORD *)(a1 + 40) = v17;
  *(_QWORD *)(a1 + 48) = v18;
  v19 = *a6;
  LODWORD(v17) = **a6;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v17;
  v20 = *((unsigned int *)a7 + 2);
  *(_DWORD *)(a1 + 152) = *v19;
  v21 = *a7;
  v45 = *a7 + 40 * v20;
  if ( *a7 != v45 )
  {
    do
    {
      v22 = *(_DWORD *)(v21 + 16);
      v23 = *(_QWORD *)(v21 + 24);
      v46[4] = &unk_4A085E0;
      v24 = (const __m128i *)v46;
      v25 = *(_QWORD *)(v21 + 32);
      v26 = *(_QWORD *)v21;
      v48 = 1;
      v27 = *(_QWORD *)(v21 + 8);
      v28 = *(unsigned int *)(a1 + 188);
      v47 = v22;
      v29 = *(unsigned int *)(a1 + 184);
      v46[2] = v23;
      v46[3] = v25;
      v30 = *(_QWORD *)(a1 + 176);
      v31 = v29 + 1;
      v46[0] = v26;
      v32 = v29;
      v46[1] = v27;
      if ( v29 + 1 > v28 )
      {
        if ( v30 > (unsigned __int64)v46 )
        {
          v39 = v27;
          v41 = v26;
LABEL_13:
          sub_2276D00(v43, v31, v29, v30, v27, v26);
          v29 = *(unsigned int *)(a1 + 184);
          v30 = *(_QWORD *)(a1 + 176);
          v26 = v41;
          v27 = v39;
          v32 = *(_DWORD *)(a1 + 184);
          goto LABEL_5;
        }
        v39 = v27;
        v41 = v26;
        v29 = v30 + 48 * v29;
        if ( (unsigned __int64)v46 >= v29 )
          goto LABEL_13;
        v38 = (char *)v46 - v30;
        sub_2276D00(v43, v31, v29, v30, v27, v26);
        v30 = *(_QWORD *)(a1 + 176);
        v29 = *(unsigned int *)(a1 + 184);
        v27 = v39;
        v26 = v41;
        v24 = (const __m128i *)&v38[v30];
        v32 = *(_DWORD *)(a1 + 184);
      }
LABEL_5:
      v33 = (__m128i *)(v30 + 48 * v29);
      if ( v33 )
      {
        v34 = _mm_loadu_si128(v24 + 1);
        *v33 = _mm_loadu_si128(v24);
        v33[1] = v34;
        v33[2].m128i_i32[2] = v24[2].m128i_i32[2];
        v35 = v24[2].m128i_i8[12];
        v33[2].m128i_i64[0] = (__int64)&unk_4A085E0;
        v33[2].m128i_i8[12] = v35;
        v32 = *(_DWORD *)(a1 + 184);
      }
      v36 = *(_QWORD *)(a1 + 168);
      v21 += 40;
      *(_DWORD *)(a1 + 184) = v32 + 1;
      sub_C52F90(v36, v26, v27);
    }
    while ( v45 != v21 );
  }
  return sub_C53130(a1);
}
