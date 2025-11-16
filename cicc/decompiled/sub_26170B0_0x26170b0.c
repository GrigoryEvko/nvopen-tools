// Function: sub_26170B0
// Address: 0x26170b0
//
unsigned __int64 __fastcall sub_26170B0(__int64 a1, const char *a2, int **a3, __int64 *a4, __int64 *a5, _BYTE *a6)
{
  __int64 v6; // r15
  int v9; // edx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 *v12; // r12
  __int64 v13; // rax
  size_t v14; // rax
  __int64 v15; // r8
  int *v16; // rax
  int v17; // edx
  __int64 v18; // rdx
  __int64 v19; // r15
  int v21; // edx
  __int64 v22; // rsi
  const __m128i *v23; // rbx
  __int64 v24; // rcx
  __int64 v25; // r10
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
  __int64 v39; // [rsp+0h] [rbp-90h]
  __int64 v40; // [rsp+8h] [rbp-88h]
  __int64 v43; // [rsp+20h] [rbp-70h]
  __int64 v44; // [rsp+28h] [rbp-68h]
  _QWORD v45[5]; // [rsp+30h] [rbp-60h] BYREF
  int v46; // [rsp+58h] [rbp-38h]
  char v47; // [rsp+5Ch] [rbp-34h]

  v6 = a1;
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
  *(_QWORD *)a1 = &unk_4A1F670;
  *(_QWORD *)(a1 + 144) = &unk_4A1F600;
  *(_QWORD *)(a1 + 168) = a1;
  *(_QWORD *)(a1 + 160) = &unk_4A1F620;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = nullsub_1540;
  *(_QWORD *)(a1 + 592) = sub_260FCA0;
  v14 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v14);
  v16 = *a3;
  v43 = a1 + 176;
  v17 = **a3;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v17;
  v18 = *((unsigned int *)a4 + 2);
  *(_DWORD *)(a1 + 152) = *v16;
  v44 = *a4 + 40 * v18;
  if ( *a4 != v44 )
  {
    v19 = *a4;
    while ( 1 )
    {
      v21 = *(_DWORD *)(v19 + 16);
      v22 = *(_QWORD *)(v19 + 24);
      v45[4] = &unk_4A1F600;
      v23 = (const __m128i *)v45;
      v24 = *(_QWORD *)(v19 + 32);
      v25 = *(_QWORD *)v19;
      v47 = 1;
      v26 = *(_QWORD *)(v19 + 8);
      v27 = *(unsigned int *)(a1 + 188);
      v46 = v21;
      v28 = *(unsigned int *)(a1 + 184);
      v45[2] = v22;
      v45[3] = v24;
      v29 = *(_QWORD *)(a1 + 176);
      v30 = v28 + 1;
      v45[0] = v25;
      v31 = v28;
      v45[1] = v26;
      if ( v28 + 1 > v27 )
      {
        if ( v29 > (unsigned __int64)v45 )
        {
          v39 = v26;
          v40 = v25;
LABEL_15:
          sub_2616FF0(v43, v30, v28, v29, v15, v26);
          v28 = *(unsigned int *)(a1 + 184);
          v29 = *(_QWORD *)(a1 + 176);
          v25 = v40;
          v26 = v39;
          v31 = *(_DWORD *)(a1 + 184);
          goto LABEL_6;
        }
        v39 = v26;
        v40 = v25;
        v28 = v29 + 48 * v28;
        if ( (unsigned __int64)v45 >= v28 )
          goto LABEL_15;
        v38 = (char *)v45 - v29;
        sub_2616FF0(v43, v30, v28, v29, v15, v26);
        v29 = *(_QWORD *)(a1 + 176);
        v28 = *(unsigned int *)(a1 + 184);
        v26 = v39;
        v25 = v40;
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
        v32[2].m128i_i64[0] = (__int64)&unk_4A1F600;
        v32[2].m128i_i8[12] = v34;
        v31 = *(_DWORD *)(a1 + 184);
      }
      v35 = *(_QWORD *)(a1 + 168);
      v19 += 40;
      *(_DWORD *)(a1 + 184) = v31 + 1;
      sub_C52F90(v35, v25, v26);
      if ( v44 == v19 )
      {
        v6 = a1;
        break;
      }
    }
  }
  v36 = *a5;
  *(_QWORD *)(v6 + 48) = a5[1];
  *(_QWORD *)(v6 + 40) = v36;
  *(_BYTE *)(v6 + 12) = (32 * (*a6 & 3)) | *(_BYTE *)(v6 + 12) & 0x9F;
  return sub_C53130(v6);
}
