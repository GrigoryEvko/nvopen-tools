// Function: sub_3566890
// Address: 0x3566890
//
unsigned __int64 __fastcall sub_3566890(__int64 a1, const char *a2, _DWORD *a3, int **a4, __int64 *a5, __int64 *a6)
{
  size_t v9; // rax
  _DWORD *v10; // r11
  int *v11; // rax
  int v12; // edx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // r14
  int v18; // edx
  __int64 v19; // rsi
  const __m128i *v20; // rbx
  __int64 v21; // rcx
  __int64 v22; // r9
  __int64 v23; // r8
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // r11
  int v28; // esi
  __m128i *v29; // rdx
  __m128i v30; // xmm1
  __int8 v31; // cl
  __int64 v32; // rdi
  char *v34; // rbx
  __int64 v35; // [rsp+0h] [rbp-80h]
  __int64 v36; // [rsp+8h] [rbp-78h]
  __int64 v38; // [rsp+10h] [rbp-70h]
  __int64 v40; // [rsp+18h] [rbp-68h]
  _QWORD v41[5]; // [rsp+20h] [rbp-60h] BYREF
  int v42; // [rsp+48h] [rbp-38h]
  char v43; // [rsp+4Ch] [rbp-34h]

  sub_D95050(a1, 0, 0);
  *(_BYTE *)(a1 + 156) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 144) = &unk_4A390A0;
  *(_QWORD *)(a1 + 168) = a1;
  *(_QWORD *)a1 = &unk_4A39110;
  *(_QWORD *)(a1 + 160) = &unk_4A390C0;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x800000000LL;
  *(_QWORD *)(a1 + 600) = nullsub_1893;
  *(_QWORD *)(a1 + 592) = sub_353D080;
  v9 = strlen(a2);
  sub_C53080(a1, (__int64)a2, v9);
  v10 = a3;
  v38 = a1 + 176;
  *(_BYTE *)(a1 + 12) = (32 * (*v10 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v11 = *a4;
  v12 = **a4;
  *(_BYTE *)(a1 + 156) = 1;
  *(_DWORD *)(a1 + 136) = v12;
  v13 = *a5;
  *(_DWORD *)(a1 + 152) = *v11;
  v14 = a5[1];
  *(_QWORD *)(a1 + 40) = v13;
  v15 = *((unsigned int *)a6 + 2);
  *(_QWORD *)(a1 + 48) = v14;
  v16 = *a6 + 40 * v15;
  v17 = *a6;
  v40 = v16;
  while ( v40 != v17 )
  {
    v18 = *(_DWORD *)(v17 + 16);
    v19 = *(_QWORD *)(v17 + 24);
    v41[4] = &unk_4A390A0;
    v20 = (const __m128i *)v41;
    v21 = *(_QWORD *)(v17 + 32);
    v22 = *(_QWORD *)v17;
    v43 = 1;
    v23 = *(_QWORD *)(v17 + 8);
    v24 = *(unsigned int *)(a1 + 188);
    v42 = v18;
    v25 = *(unsigned int *)(a1 + 184);
    v41[2] = v19;
    v41[3] = v21;
    v26 = *(_QWORD *)(a1 + 176);
    v27 = v25 + 1;
    v41[0] = v22;
    v28 = v25;
    v41[1] = v23;
    if ( v25 + 1 > v24 )
    {
      if ( v26 > (unsigned __int64)v41 )
      {
        v35 = v23;
        v36 = v22;
LABEL_11:
        sub_35667D0(v38, v27, v25, v26, v23, v22);
        v25 = *(unsigned int *)(a1 + 184);
        v26 = *(_QWORD *)(a1 + 176);
        v22 = v36;
        v23 = v35;
        v28 = *(_DWORD *)(a1 + 184);
        goto LABEL_3;
      }
      v35 = v23;
      v36 = v22;
      v25 = v26 + 48 * v25;
      if ( (unsigned __int64)v41 >= v25 )
        goto LABEL_11;
      v34 = (char *)v41 - v26;
      sub_35667D0(v38, v27, v25, v26, v23, v22);
      v26 = *(_QWORD *)(a1 + 176);
      v25 = *(unsigned int *)(a1 + 184);
      v23 = v35;
      v22 = v36;
      v20 = (const __m128i *)&v34[v26];
      v28 = *(_DWORD *)(a1 + 184);
    }
LABEL_3:
    v29 = (__m128i *)(v26 + 48 * v25);
    if ( v29 )
    {
      v30 = _mm_loadu_si128(v20 + 1);
      *v29 = _mm_loadu_si128(v20);
      v29[1] = v30;
      v29[2].m128i_i32[2] = v20[2].m128i_i32[2];
      v31 = v20[2].m128i_i8[12];
      v29[2].m128i_i64[0] = (__int64)&unk_4A390A0;
      v29[2].m128i_i8[12] = v31;
      v28 = *(_DWORD *)(a1 + 184);
    }
    v32 = *(_QWORD *)(a1 + 168);
    v17 += 40;
    *(_DWORD *)(a1 + 184) = v28 + 1;
    sub_C52F90(v32, v22, v23);
  }
  return sub_C53130(a1);
}
