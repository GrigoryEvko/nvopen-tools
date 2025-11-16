// Function: sub_30AB120
// Address: 0x30ab120
//
unsigned __int64 __fastcall sub_30AB120(__int64 a1, const char *a2, int **a3, _BYTE *a4, __int64 *a5, __int64 *a6)
{
  __int64 v8; // r12
  int v9; // edx
  __int64 *v10; // rbx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  size_t v14; // rax
  __int64 v15; // r8
  int *v16; // rax
  __int64 v17; // r15
  int v18; // edx
  __int64 v19; // r12
  int v21; // edx
  __int64 v22; // r10
  const __m128i *v23; // rbx
  __int64 v24; // rsi
  __int64 v25; // rcx
  __int64 v26; // r9
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rcx
  unsigned __int64 v30; // r11
  int v31; // esi
  __m128i *v32; // rdx
  __m128i v33; // xmm1
  __int8 v34; // cl
  __int64 v35; // rdi
  __int64 v36; // rdx
  char *v38; // rbx
  __int64 v39; // [rsp+8h] [rbp-88h]
  __int64 v40; // [rsp+10h] [rbp-80h]
  __int64 v42; // [rsp+20h] [rbp-70h]
  __int64 v44; // [rsp+28h] [rbp-68h]
  _QWORD v45[5]; // [rsp+30h] [rbp-60h] BYREF
  int v46; // [rsp+58h] [rbp-38h]
  char v47; // [rsp+5Ch] [rbp-34h]

  v8 = a1;
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
  *(_QWORD *)(a1 + 144) = &unk_4A32078;
  *(_QWORD *)a1 = &unk_4A320E8;
  *(_BYTE *)(a1 + 156) = 0;
  *(_QWORD *)(a1 + 168) = a1;
  *(_QWORD *)(a1 + 160) = &unk_4A32098;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = nullsub_1745;
  *(_QWORD *)(a1 + 592) = sub_30A66D0;
  v14 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v14);
  v16 = *a3;
  v17 = *a5;
  v18 = *v16;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v18;
  *(_DWORD *)(a1 + 152) = *v16;
  *(_BYTE *)(a1 + 12) = (32 * (*a4 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v42 = a1 + 176;
  if ( v17 != v17 + 40LL * *((unsigned int *)a5 + 2) )
  {
    v44 = v17 + 40LL * *((unsigned int *)a5 + 2);
    v19 = v17;
    while ( 1 )
    {
      v21 = *(_DWORD *)(v19 + 16);
      v22 = *(_QWORD *)v19;
      v45[4] = &unk_4A32078;
      v23 = (const __m128i *)v45;
      v24 = *(_QWORD *)(v19 + 24);
      v25 = *(_QWORD *)(v19 + 32);
      v47 = 1;
      v26 = *(_QWORD *)(v19 + 8);
      v27 = *(unsigned int *)(a1 + 188);
      v46 = v21;
      v28 = *(unsigned int *)(a1 + 184);
      v45[2] = v24;
      v45[3] = v25;
      v29 = *(_QWORD *)(a1 + 176);
      v30 = v28 + 1;
      v45[0] = v22;
      v31 = v28;
      v45[1] = v26;
      if ( v28 + 1 > v27 )
      {
        if ( v29 > (unsigned __int64)v45 )
        {
          v39 = v26;
          v40 = v22;
LABEL_15:
          sub_30AB060(v42, v30, v28, v29, v15, v26);
          v28 = *(unsigned int *)(a1 + 184);
          v22 = v40;
          v29 = *(_QWORD *)(a1 + 176);
          v26 = v39;
          v31 = *(_DWORD *)(a1 + 184);
          goto LABEL_6;
        }
        v39 = v26;
        v40 = v22;
        v28 = v29 + 48 * v28;
        if ( (unsigned __int64)v45 >= v28 )
          goto LABEL_15;
        v38 = (char *)v45 - v29;
        sub_30AB060(v42, v30, v28, v29, v15, v26);
        v29 = *(_QWORD *)(a1 + 176);
        v28 = *(unsigned int *)(a1 + 184);
        v26 = v39;
        v22 = v40;
        v23 = (const __m128i *)&v38[v29];
        v31 = *(_DWORD *)(a1 + 184);
      }
LABEL_6:
      v32 = (__m128i *)(v29 + 48 * v28);
      if ( v32 )
      {
        v33 = _mm_loadu_si128(v23 + 1);
        *v32 = _mm_loadu_si128(v23);
        v32[1] = v33;
        v32[2].m128i_i32[2] = v23[2].m128i_i32[2];
        v34 = v23[2].m128i_i8[12];
        v32[2].m128i_i64[0] = (__int64)&unk_4A32078;
        v32[2].m128i_i8[12] = v34;
        v31 = *(_DWORD *)(a1 + 184);
      }
      v35 = *(_QWORD *)(a1 + 168);
      v19 += 40;
      *(_DWORD *)(a1 + 184) = v31 + 1;
      sub_C52F90(v35, v22, v26);
      if ( v44 == v19 )
      {
        v8 = a1;
        break;
      }
    }
  }
  v36 = *a6;
  *(_QWORD *)(v8 + 48) = a6[1];
  *(_QWORD *)(v8 + 40) = v36;
  return sub_C53130(v8);
}
