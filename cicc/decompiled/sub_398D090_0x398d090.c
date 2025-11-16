// Function: sub_398D090
// Address: 0x398d090
//
__int64 *__fastcall sub_398D090(__int64 a1, const char *a2, _DWORD *a3, __int64 *a4, __int64 *a5, int **a6)
{
  int v10; // edx
  size_t v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  unsigned int v16; // r14d
  __int64 v17; // rax
  const char *v18; // r8
  size_t v19; // r15
  __int32 v20; // ecx
  __int64 v21; // r10
  __int64 v22; // r9
  __m128i *v23; // r13
  __int64 v24; // rdx
  __m128i *v25; // rdx
  int *v26; // rax
  int v27; // edx
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int64 v34; // r11
  __m128i *v35; // rax
  const __m128i *v36; // rsi
  __int8 v37; // cl
  int v38; // [rsp+Ch] [rbp-74h]
  int v39; // [rsp+Ch] [rbp-74h]
  const char *v40; // [rsp+10h] [rbp-70h]
  const char *v41; // [rsp+10h] [rbp-70h]
  __int64 v42; // [rsp+18h] [rbp-68h]
  int v43; // [rsp+18h] [rbp-68h]
  __int64 v44; // [rsp+18h] [rbp-68h]
  __int64 v45; // [rsp+20h] [rbp-60h]
  __int64 v46; // [rsp+20h] [rbp-60h]
  int v47; // [rsp+28h] [rbp-58h]
  __int64 v48; // [rsp+30h] [rbp-50h]
  __int64 v50; // [rsp+48h] [rbp-38h]

  *(_QWORD *)a1 = &unk_49EED30;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  *(_WORD *)(a1 + 12) &= 0xF000u;
  *(_QWORD *)(a1 + 72) = qword_4FA01C0;
  *(_QWORD *)(a1 + 88) = a1 + 120;
  *(_QWORD *)(a1 + 96) = a1 + 120;
  *(_DWORD *)(a1 + 8) = v10;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 168) = &unk_4A3FA78;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_4A3FAE8;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 184) = &unk_4A3FA98;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  v48 = a1 + 216;
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
  v11 = strlen(a2);
  sub_16B8280(a1, a2, v11);
  v12 = *a4;
  *(_BYTE *)(a1 + 12) = (32 * (*a3 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v13 = a4[1];
  *(_QWORD *)(a1 + 40) = v12;
  v14 = *((unsigned int *)a5 + 2);
  *(_QWORD *)(a1 + 48) = v13;
  v15 = *a5;
  v50 = *a5 + 40 * v14;
  if ( *a5 != v50 )
  {
    do
    {
      v16 = *(_DWORD *)(a1 + 208);
      v17 = *(unsigned int *)(a1 + 212);
      v18 = *(const char **)v15;
      v19 = *(_QWORD *)(v15 + 8);
      v20 = *(_DWORD *)(v15 + 16);
      v21 = *(_QWORD *)(v15 + 24);
      v22 = *(_QWORD *)(v15 + 32);
      if ( v16 >= (unsigned int)v17 )
      {
        v38 = *(_DWORD *)(v15 + 16);
        v40 = *(const char **)v15;
        v42 = *(_QWORD *)(v15 + 24);
        v45 = *(_QWORD *)(v15 + 32);
        v29 = ((((unsigned __int64)(v17 + 2) >> 1) | (v17 + 2)) >> 2) | ((unsigned __int64)(v17 + 2) >> 1) | (v17 + 2);
        v30 = (((v29 >> 4) | v29) >> 8) | (v29 >> 4) | v29;
        v31 = (v30 | (v30 >> 16) | HIDWORD(v30)) + 1;
        v32 = 0xFFFFFFFFLL;
        if ( v31 <= 0xFFFFFFFF )
          v32 = v31;
        v47 = v32;
        v33 = malloc(48 * v32);
        v22 = v45;
        v21 = v42;
        v18 = v40;
        v20 = v38;
        v23 = (__m128i *)v33;
        if ( !v33 )
        {
          sub_16BD1C0("Allocation failed", 1u);
          v16 = *(_DWORD *)(a1 + 208);
          v20 = v38;
          v18 = v40;
          v21 = v42;
          v22 = v45;
        }
        v34 = *(_QWORD *)(a1 + 200);
        v35 = v23;
        v24 = 3LL * v16;
        v36 = (const __m128i *)v34;
        if ( v34 != v34 + v24 * 16 )
        {
          v43 = v20;
          do
          {
            if ( v35 )
            {
              *v35 = _mm_loadu_si128(v36);
              v35[1] = _mm_loadu_si128(v36 + 1);
              v35[2].m128i_i32[2] = v36[2].m128i_i32[2];
              v37 = v36[2].m128i_i8[12];
              v35[2].m128i_i64[0] = (__int64)&unk_4A3FA78;
              v35[2].m128i_i8[12] = v37;
            }
            v36 += 3;
            v35 += 3;
          }
          while ( (const __m128i *)(v34 + v24 * 16) != v36 );
          v24 = 3LL * v16;
          v20 = v43;
        }
        if ( v48 != v34 )
        {
          v39 = v20;
          v41 = v18;
          v44 = v21;
          v46 = v22;
          _libc_free(v34);
          v20 = v39;
          v18 = v41;
          v21 = v44;
          v22 = v46;
          v16 = *(_DWORD *)(a1 + 208);
          v24 = 3LL * v16;
        }
        *(_QWORD *)(a1 + 200) = v23;
        *(_DWORD *)(a1 + 212) = v47;
      }
      else
      {
        v23 = *(__m128i **)(a1 + 200);
        v24 = 3LL * v16;
      }
      v25 = &v23[v24];
      if ( v25 )
      {
        v25->m128i_i64[0] = (__int64)v18;
        v25->m128i_i64[1] = v19;
        v25[1].m128i_i64[0] = v21;
        v25[1].m128i_i64[1] = v22;
        v25[2].m128i_i32[2] = v20;
        v25[2].m128i_i8[12] = 1;
        v25[2].m128i_i64[0] = (__int64)&unk_4A3FA78;
        v16 = *(_DWORD *)(a1 + 208);
      }
      v15 += 40;
      *(_DWORD *)(a1 + 208) = v16 + 1;
      sub_16B7FD0(*(_QWORD *)(a1 + 192), v18, v19);
    }
    while ( v50 != v15 );
  }
  v26 = *a6;
  v27 = **a6;
  *(_BYTE *)(a1 + 180) = 1;
  *(_DWORD *)(a1 + 160) = v27;
  *(_DWORD *)(a1 + 176) = *v26;
  return sub_16B88A0(a1);
}
