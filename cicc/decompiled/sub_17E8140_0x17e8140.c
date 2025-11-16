// Function: sub_17E8140
// Address: 0x17e8140
//
__int64 *__fastcall sub_17E8140(__int64 a1, const char *a2, _DWORD *a3, __int64 *a4, __int64 *a5)
{
  int v9; // edx
  size_t v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rbx
  unsigned int v15; // r14d
  __int64 v16; // rax
  const char *v17; // r8
  size_t v18; // rdx
  __int32 v19; // ecx
  __int64 v20; // r10
  __int64 v21; // r9
  __m128i *v22; // r13
  __int64 v23; // rsi
  __m128i *v24; // rsi
  __int64 v25; // rdi
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rax
  const __m128i *v32; // rdi
  __m128i *v33; // rax
  const __m128i *v34; // r11
  __int8 v35; // dl
  int v36; // [rsp+Ch] [rbp-74h]
  int v37; // [rsp+10h] [rbp-70h]
  const char *v38; // [rsp+10h] [rbp-70h]
  const char *v39; // [rsp+18h] [rbp-68h]
  size_t v40; // [rsp+18h] [rbp-68h]
  size_t v41; // [rsp+20h] [rbp-60h]
  __int64 v42; // [rsp+20h] [rbp-60h]
  __int64 v43; // [rsp+28h] [rbp-58h]
  size_t v44; // [rsp+28h] [rbp-58h]
  __int64 v45; // [rsp+28h] [rbp-58h]
  int v46; // [rsp+30h] [rbp-50h]
  __int64 v47; // [rsp+38h] [rbp-48h]
  __int64 v48; // [rsp+40h] [rbp-40h]
  unsigned __int64 v49; // [rsp+40h] [rbp-40h]
  __int64 v50; // [rsp+48h] [rbp-38h]

  *(_QWORD *)a1 = &unk_49EED30;
  v9 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  *(_WORD *)(a1 + 12) &= 0xF000u;
  *(_QWORD *)(a1 + 72) = qword_4FA01C0;
  *(_QWORD *)(a1 + 88) = a1 + 120;
  *(_QWORD *)(a1 + 96) = a1 + 120;
  *(_DWORD *)(a1 + 8) = v9;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 168) = &unk_49E8888;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_49E88F8;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 184) = &unk_49E88A8;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  v47 = a1 + 216;
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
  *(_QWORD *)(a1 + 208) = 0x800000000LL;
  v10 = strlen(a2);
  sub_16B8280(a1, a2, v10);
  v11 = *a4;
  *(_BYTE *)(a1 + 12) = (32 * (*a3 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v12 = a4[1];
  *(_QWORD *)(a1 + 40) = v11;
  v13 = *((unsigned int *)a5 + 2);
  *(_QWORD *)(a1 + 48) = v12;
  v14 = *a5;
  v50 = *a5 + 40 * v13;
  if ( *a5 != v50 )
  {
    do
    {
      v15 = *(_DWORD *)(a1 + 208);
      v16 = *(unsigned int *)(a1 + 212);
      v17 = *(const char **)v14;
      v18 = *(_QWORD *)(v14 + 8);
      v19 = *(_DWORD *)(v14 + 16);
      v20 = *(_QWORD *)(v14 + 24);
      v21 = *(_QWORD *)(v14 + 32);
      if ( v15 >= (unsigned int)v16 )
      {
        v37 = *(_DWORD *)(v14 + 16);
        v39 = *(const char **)v14;
        v41 = *(_QWORD *)(v14 + 8);
        v43 = *(_QWORD *)(v14 + 24);
        v48 = *(_QWORD *)(v14 + 32);
        v27 = ((((unsigned __int64)(v16 + 2) >> 1) | (v16 + 2)) >> 2) | ((unsigned __int64)(v16 + 2) >> 1) | (v16 + 2);
        v28 = (((v27 >> 4) | v27) >> 8) | (v27 >> 4) | v27;
        v29 = (v28 | (v28 >> 16) | HIDWORD(v28)) + 1;
        v30 = 0xFFFFFFFFLL;
        if ( v29 <= 0xFFFFFFFF )
          v30 = v29;
        v46 = v30;
        v31 = malloc(48 * v30);
        v21 = v48;
        v20 = v43;
        v18 = v41;
        v17 = v39;
        v22 = (__m128i *)v31;
        v19 = v37;
        if ( !v31 )
        {
          sub_16BD1C0("Allocation failed", 1u);
          v15 = *(_DWORD *)(a1 + 208);
          v19 = v37;
          v17 = v39;
          v18 = v41;
          v20 = v43;
          v21 = v48;
        }
        v32 = *(const __m128i **)(a1 + 200);
        v33 = v22;
        v23 = 3LL * v15;
        v49 = (unsigned __int64)v32;
        v34 = &v32[v23];
        if ( v32 != &v32[v23] )
        {
          v44 = v18;
          do
          {
            if ( v33 )
            {
              *v33 = _mm_loadu_si128(v32);
              v33[1] = _mm_loadu_si128(v32 + 1);
              v33[2].m128i_i32[2] = v32[2].m128i_i32[2];
              v35 = v32[2].m128i_i8[12];
              v33[2].m128i_i64[0] = (__int64)&unk_49E8888;
              v33[2].m128i_i8[12] = v35;
            }
            v32 += 3;
            v33 += 3;
          }
          while ( v34 != v32 );
          v18 = v44;
        }
        if ( v47 != v49 )
        {
          v36 = v19;
          v38 = v17;
          v40 = v18;
          v42 = v20;
          v45 = v21;
          _libc_free(v49);
          v19 = v36;
          v17 = v38;
          v18 = v40;
          v20 = v42;
          v21 = v45;
          v15 = *(_DWORD *)(a1 + 208);
          v23 = 3LL * v15;
        }
        *(_QWORD *)(a1 + 200) = v22;
        *(_DWORD *)(a1 + 212) = v46;
      }
      else
      {
        v22 = *(__m128i **)(a1 + 200);
        v23 = 3LL * v15;
      }
      v24 = &v22[v23];
      if ( v24 )
      {
        v24->m128i_i64[0] = (__int64)v17;
        v24->m128i_i64[1] = v18;
        v24[1].m128i_i64[0] = v20;
        v24[1].m128i_i64[1] = v21;
        v24[2].m128i_i32[2] = v19;
        v24[2].m128i_i8[12] = 1;
        v24[2].m128i_i64[0] = (__int64)&unk_49E8888;
        v15 = *(_DWORD *)(a1 + 208);
      }
      v25 = *(_QWORD *)(a1 + 192);
      v14 += 40;
      *(_DWORD *)(a1 + 208) = v15 + 1;
      sub_16B7FD0(v25, v17, v18);
    }
    while ( v50 != v14 );
  }
  return sub_16B88A0(a1);
}
