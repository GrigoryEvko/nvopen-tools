// Function: sub_1940280
// Address: 0x1940280
//
__int64 *__fastcall sub_1940280(__int64 a1, const char *a2, _DWORD *a3, int **a4, __int64 *a5, __int64 *a6)
{
  int v11; // edx
  size_t v12; // rax
  int *v13; // rax
  int v14; // edx
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rbx
  unsigned int v19; // r14d
  __int64 v20; // rax
  const char *v21; // r8
  size_t v22; // rdx
  __int32 v23; // ecx
  __int64 v24; // r10
  __int64 v25; // r9
  __m128i *v26; // r13
  __int64 v27; // rsi
  __m128i *v28; // rsi
  __int64 v29; // rdi
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rax
  const __m128i *v36; // rdi
  __m128i *v37; // rax
  const __m128i *v38; // r11
  __int8 v39; // dl
  int v40; // [rsp+Ch] [rbp-74h]
  int v41; // [rsp+10h] [rbp-70h]
  const char *v42; // [rsp+10h] [rbp-70h]
  const char *v43; // [rsp+18h] [rbp-68h]
  size_t v44; // [rsp+18h] [rbp-68h]
  size_t v45; // [rsp+20h] [rbp-60h]
  __int64 v46; // [rsp+20h] [rbp-60h]
  __int64 v47; // [rsp+28h] [rbp-58h]
  size_t v48; // [rsp+28h] [rbp-58h]
  __int64 v49; // [rsp+28h] [rbp-58h]
  int v50; // [rsp+30h] [rbp-50h]
  __int64 v51; // [rsp+38h] [rbp-48h]
  __int64 v52; // [rsp+40h] [rbp-40h]
  unsigned __int64 v53; // [rsp+40h] [rbp-40h]
  __int64 v54; // [rsp+48h] [rbp-38h]

  *(_QWORD *)a1 = &unk_49EED30;
  v11 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  *(_WORD *)(a1 + 12) &= 0xF000u;
  *(_QWORD *)(a1 + 72) = qword_4FA01C0;
  *(_QWORD *)(a1 + 88) = a1 + 120;
  *(_QWORD *)(a1 + 96) = a1 + 120;
  *(_DWORD *)(a1 + 8) = v11;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 168) = &unk_49F3758;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_49F37C8;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 184) = &unk_49F3778;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  v51 = a1 + 216;
  *(_QWORD *)(a1 + 208) = 0x800000000LL;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 104) = 4;
  *(_DWORD *)(a1 + 112) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_DWORD *)(a1 + 160) = 0;
  *(_BYTE *)(a1 + 180) = 1;
  *(_DWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 192) = a1;
  v12 = strlen(a2);
  sub_16B8280(a1, a2, v12);
  *(_BYTE *)(a1 + 12) = (32 * (*a3 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v13 = *a4;
  v14 = **a4;
  *(_BYTE *)(a1 + 180) = 1;
  *(_DWORD *)(a1 + 160) = v14;
  v15 = *a5;
  *(_DWORD *)(a1 + 176) = *v13;
  v16 = a5[1];
  *(_QWORD *)(a1 + 40) = v15;
  v17 = *((unsigned int *)a6 + 2);
  *(_QWORD *)(a1 + 48) = v16;
  v18 = *a6;
  v54 = *a6 + 40 * v17;
  if ( *a6 != v54 )
  {
    do
    {
      v19 = *(_DWORD *)(a1 + 208);
      v20 = *(unsigned int *)(a1 + 212);
      v21 = *(const char **)v18;
      v22 = *(_QWORD *)(v18 + 8);
      v23 = *(_DWORD *)(v18 + 16);
      v24 = *(_QWORD *)(v18 + 24);
      v25 = *(_QWORD *)(v18 + 32);
      if ( v19 >= (unsigned int)v20 )
      {
        v41 = *(_DWORD *)(v18 + 16);
        v43 = *(const char **)v18;
        v45 = *(_QWORD *)(v18 + 8);
        v47 = *(_QWORD *)(v18 + 24);
        v52 = *(_QWORD *)(v18 + 32);
        v31 = ((((unsigned __int64)(v20 + 2) >> 1) | (v20 + 2)) >> 2) | ((unsigned __int64)(v20 + 2) >> 1) | (v20 + 2);
        v32 = (((v31 >> 4) | v31) >> 8) | (v31 >> 4) | v31;
        v33 = (v32 | (v32 >> 16) | HIDWORD(v32)) + 1;
        v34 = 0xFFFFFFFFLL;
        if ( v33 <= 0xFFFFFFFF )
          v34 = v33;
        v50 = v34;
        v35 = malloc(48 * v34);
        v25 = v52;
        v24 = v47;
        v22 = v45;
        v21 = v43;
        v26 = (__m128i *)v35;
        v23 = v41;
        if ( !v35 )
        {
          sub_16BD1C0("Allocation failed", 1u);
          v19 = *(_DWORD *)(a1 + 208);
          v23 = v41;
          v21 = v43;
          v22 = v45;
          v24 = v47;
          v25 = v52;
        }
        v36 = *(const __m128i **)(a1 + 200);
        v37 = v26;
        v27 = 3LL * v19;
        v53 = (unsigned __int64)v36;
        v38 = &v36[v27];
        if ( v36 != &v36[v27] )
        {
          v48 = v22;
          do
          {
            if ( v37 )
            {
              *v37 = _mm_loadu_si128(v36);
              v37[1] = _mm_loadu_si128(v36 + 1);
              v37[2].m128i_i32[2] = v36[2].m128i_i32[2];
              v39 = v36[2].m128i_i8[12];
              v37[2].m128i_i64[0] = (__int64)&unk_49F3758;
              v37[2].m128i_i8[12] = v39;
            }
            v36 += 3;
            v37 += 3;
          }
          while ( v38 != v36 );
          v22 = v48;
        }
        if ( v51 != v53 )
        {
          v40 = v23;
          v42 = v21;
          v44 = v22;
          v46 = v24;
          v49 = v25;
          _libc_free(v53);
          v23 = v40;
          v21 = v42;
          v22 = v44;
          v24 = v46;
          v25 = v49;
          v19 = *(_DWORD *)(a1 + 208);
          v27 = 3LL * v19;
        }
        *(_QWORD *)(a1 + 200) = v26;
        *(_DWORD *)(a1 + 212) = v50;
      }
      else
      {
        v26 = *(__m128i **)(a1 + 200);
        v27 = 3LL * v19;
      }
      v28 = &v26[v27];
      if ( v28 )
      {
        v28->m128i_i64[0] = (__int64)v21;
        v28->m128i_i64[1] = v22;
        v28[1].m128i_i64[0] = v24;
        v28[1].m128i_i64[1] = v25;
        v28[2].m128i_i32[2] = v23;
        v28[2].m128i_i8[12] = 1;
        v28[2].m128i_i64[0] = (__int64)&unk_49F3758;
        v19 = *(_DWORD *)(a1 + 208);
      }
      v29 = *(_QWORD *)(a1 + 192);
      v18 += 40;
      *(_DWORD *)(a1 + 208) = v19 + 1;
      sub_16B7FD0(v29, v21, v22);
    }
    while ( v54 != v18 );
  }
  return sub_16B88A0(a1);
}
