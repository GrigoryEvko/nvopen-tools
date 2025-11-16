// Function: sub_2276520
// Address: 0x2276520
//
unsigned __int64 __fastcall sub_2276520(__int64 a1, const char *a2, int **a3, _DWORD *a4, __int64 *a5, __int64 *a6)
{
  int v9; // edx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 *v12; // r12
  __int64 v13; // rax
  size_t v14; // rax
  int *v15; // rax
  int v16; // edx
  _DWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r14
  int v22; // edx
  __int64 v23; // rsi
  const __m128i *v24; // rbx
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
  char *v38; // rbx
  __int64 v39; // [rsp+0h] [rbp-80h]
  __int64 v40; // [rsp+8h] [rbp-78h]
  __int64 v42; // [rsp+10h] [rbp-70h]
  __int64 v44; // [rsp+18h] [rbp-68h]
  _QWORD v45[5]; // [rsp+20h] [rbp-60h] BYREF
  int v46; // [rsp+48h] [rbp-38h]
  char v47; // [rsp+4Ch] [rbp-34h]

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
  v12 = sub_C57470();
  v13 = *(unsigned int *)(a1 + 80);
  if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 84) )
  {
    sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v13 + 1, 8u, v10, v11);
    v13 = *(unsigned int *)(a1 + 80);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8 * v13) = v12;
  ++*(_DWORD *)(a1 + 80);
  *(_DWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_BYTE *)(a1 + 156) = 0;
  *(_QWORD *)a1 = &unk_4A08830;
  *(_QWORD *)(a1 + 144) = &unk_4A087C0;
  *(_QWORD *)(a1 + 168) = a1;
  *(_QWORD *)(a1 + 160) = &unk_4A087E0;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = nullsub_814;
  *(_QWORD *)(a1 + 592) = sub_226E2F0;
  v14 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v14);
  v15 = *a3;
  v16 = **a3;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v16;
  *(_DWORD *)(a1 + 152) = *v15;
  v17 = a4;
  v42 = a1 + 176;
  v18 = *a5;
  *(_BYTE *)(a1 + 12) = (32 * (*v17 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v19 = a5[1];
  *(_QWORD *)(a1 + 40) = v18;
  v20 = *((unsigned int *)a6 + 2);
  *(_QWORD *)(a1 + 48) = v19;
  v21 = *a6;
  v44 = *a6 + 40 * v20;
  if ( *a6 != v44 )
  {
    do
    {
      v22 = *(_DWORD *)(v21 + 16);
      v23 = *(_QWORD *)(v21 + 24);
      v45[4] = &unk_4A087C0;
      v24 = (const __m128i *)v45;
      v25 = *(_QWORD *)(v21 + 32);
      v26 = *(_QWORD *)v21;
      v47 = 1;
      v27 = *(_QWORD *)(v21 + 8);
      v28 = *(unsigned int *)(a1 + 188);
      v46 = v22;
      v29 = *(unsigned int *)(a1 + 184);
      v45[2] = v23;
      v45[3] = v25;
      v30 = *(_QWORD *)(a1 + 176);
      v31 = v29 + 1;
      v45[0] = v26;
      v32 = v29;
      v45[1] = v27;
      if ( v29 + 1 > v28 )
      {
        if ( v30 > (unsigned __int64)v45 )
        {
          v39 = v27;
          v40 = v26;
LABEL_13:
          sub_2276460(v42, v31, v29, v30, v27, v26);
          v29 = *(unsigned int *)(a1 + 184);
          v30 = *(_QWORD *)(a1 + 176);
          v26 = v40;
          v27 = v39;
          v32 = *(_DWORD *)(a1 + 184);
          goto LABEL_5;
        }
        v39 = v27;
        v40 = v26;
        v29 = v30 + 48 * v29;
        if ( (unsigned __int64)v45 >= v29 )
          goto LABEL_13;
        v38 = (char *)v45 - v30;
        sub_2276460(v42, v31, v29, v30, v27, v26);
        v30 = *(_QWORD *)(a1 + 176);
        v29 = *(unsigned int *)(a1 + 184);
        v27 = v39;
        v26 = v40;
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
        v33[2].m128i_i64[0] = (__int64)&unk_4A087C0;
        v33[2].m128i_i8[12] = v35;
        v32 = *(_DWORD *)(a1 + 184);
      }
      v36 = *(_QWORD *)(a1 + 168);
      v21 += 40;
      *(_DWORD *)(a1 + 184) = v32 + 1;
      sub_C52F90(v36, v26, v27);
    }
    while ( v44 != v21 );
  }
  return sub_C53130(a1);
}
