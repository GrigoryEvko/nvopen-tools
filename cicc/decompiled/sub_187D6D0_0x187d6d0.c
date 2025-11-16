// Function: sub_187D6D0
// Address: 0x187d6d0
//
__int64 *__fastcall sub_187D6D0(__int64 a1, const char *a2, _QWORD *a3, __int64 *a4, _BYTE *a5)
{
  int v8; // edx
  size_t v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rbx
  unsigned int v14; // r14d
  __int64 v15; // rax
  const char *v16; // r8
  size_t v17; // r15
  __int32 v18; // ecx
  __int64 v19; // r10
  __int64 v20; // r9
  __m128i *v21; // r13
  __int64 v22; // rdx
  __m128i *v23; // rdx
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int64 v30; // r11
  __m128i *v31; // rax
  const __m128i *v32; // rsi
  __int8 v33; // cl
  int v34; // [rsp+Ch] [rbp-74h]
  int v35; // [rsp+Ch] [rbp-74h]
  const char *v36; // [rsp+10h] [rbp-70h]
  const char *v37; // [rsp+10h] [rbp-70h]
  __int64 v38; // [rsp+18h] [rbp-68h]
  int v39; // [rsp+18h] [rbp-68h]
  __int64 v40; // [rsp+18h] [rbp-68h]
  __int64 v41; // [rsp+20h] [rbp-60h]
  __int64 v42; // [rsp+20h] [rbp-60h]
  int v43; // [rsp+28h] [rbp-58h]
  __int64 v44; // [rsp+30h] [rbp-50h]
  __int64 v46; // [rsp+48h] [rbp-38h]

  *(_QWORD *)a1 = &unk_49EED30;
  v8 = _InterlockedExchangeAdd64((volatile signed __int64 *)&unk_4FA0230, 1u);
  *(_WORD *)(a1 + 12) &= 0xF000u;
  *(_QWORD *)(a1 + 72) = qword_4FA01C0;
  *(_QWORD *)(a1 + 88) = a1 + 120;
  *(_QWORD *)(a1 + 96) = a1 + 120;
  *(_DWORD *)(a1 + 8) = v8;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 168) = &unk_49F1B50;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_49F1BC0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 184) = &unk_49F1B70;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  v44 = a1 + 216;
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
  v9 = strlen(a2);
  sub_16B8280(a1, a2, v9);
  v10 = a3[1];
  *(_QWORD *)(a1 + 40) = *a3;
  v11 = *((unsigned int *)a4 + 2);
  *(_QWORD *)(a1 + 48) = v10;
  v12 = *a4 + 40 * v11;
  v13 = *a4;
  v46 = v12;
  while ( v46 != v13 )
  {
    v14 = *(_DWORD *)(a1 + 208);
    v15 = *(unsigned int *)(a1 + 212);
    v16 = *(const char **)v13;
    v17 = *(_QWORD *)(v13 + 8);
    v18 = *(_DWORD *)(v13 + 16);
    v19 = *(_QWORD *)(v13 + 24);
    v20 = *(_QWORD *)(v13 + 32);
    if ( v14 >= (unsigned int)v15 )
    {
      v34 = *(_DWORD *)(v13 + 16);
      v36 = *(const char **)v13;
      v38 = *(_QWORD *)(v13 + 24);
      v41 = *(_QWORD *)(v13 + 32);
      v25 = ((((unsigned __int64)(v15 + 2) >> 1) | (v15 + 2)) >> 2) | ((unsigned __int64)(v15 + 2) >> 1) | (v15 + 2);
      v26 = (((v25 >> 4) | v25) >> 8) | (v25 >> 4) | v25;
      v27 = (v26 | (v26 >> 16) | HIDWORD(v26)) + 1;
      v28 = 0xFFFFFFFFLL;
      if ( v27 <= 0xFFFFFFFF )
        v28 = v27;
      v43 = v28;
      v29 = malloc(48 * v28);
      v20 = v41;
      v19 = v38;
      v16 = v36;
      v18 = v34;
      v21 = (__m128i *)v29;
      if ( !v29 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v14 = *(_DWORD *)(a1 + 208);
        v18 = v34;
        v16 = v36;
        v19 = v38;
        v20 = v41;
      }
      v30 = *(_QWORD *)(a1 + 200);
      v31 = v21;
      v22 = 3LL * v14;
      v32 = (const __m128i *)v30;
      if ( v30 != v30 + v22 * 16 )
      {
        v39 = v18;
        do
        {
          if ( v31 )
          {
            *v31 = _mm_loadu_si128(v32);
            v31[1] = _mm_loadu_si128(v32 + 1);
            v31[2].m128i_i32[2] = v32[2].m128i_i32[2];
            v33 = v32[2].m128i_i8[12];
            v31[2].m128i_i64[0] = (__int64)&unk_49F1B50;
            v31[2].m128i_i8[12] = v33;
          }
          v32 += 3;
          v31 += 3;
        }
        while ( (const __m128i *)(v30 + v22 * 16) != v32 );
        v22 = 3LL * v14;
        v18 = v39;
      }
      if ( v44 != v30 )
      {
        v35 = v18;
        v37 = v16;
        v40 = v19;
        v42 = v20;
        _libc_free(v30);
        v18 = v35;
        v16 = v37;
        v19 = v40;
        v20 = v42;
        v14 = *(_DWORD *)(a1 + 208);
        v22 = 3LL * v14;
      }
      *(_QWORD *)(a1 + 200) = v21;
      *(_DWORD *)(a1 + 212) = v43;
    }
    else
    {
      v21 = *(__m128i **)(a1 + 200);
      v22 = 3LL * v14;
    }
    v23 = &v21[v22];
    if ( v23 )
    {
      v23->m128i_i64[0] = (__int64)v16;
      v23->m128i_i64[1] = v17;
      v23[1].m128i_i64[0] = v19;
      v23[1].m128i_i64[1] = v20;
      v23[2].m128i_i32[2] = v18;
      v23[2].m128i_i8[12] = 1;
      v23[2].m128i_i64[0] = (__int64)&unk_49F1B50;
      v14 = *(_DWORD *)(a1 + 208);
    }
    v13 += 40;
    *(_DWORD *)(a1 + 208) = v14 + 1;
    sub_16B7FD0(*(_QWORD *)(a1 + 192), v16, v17);
  }
  *(_BYTE *)(a1 + 12) = (32 * (*a5 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  return sub_16B88A0(a1);
}
