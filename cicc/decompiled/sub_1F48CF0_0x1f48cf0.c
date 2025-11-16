// Function: sub_1F48CF0
// Address: 0x1f48cf0
//
void __fastcall sub_1F48CF0(__int64 a1, const char *a2, int **a3, _DWORD *a4, _QWORD *a5, __int64 *a6)
{
  size_t v9; // rax
  int *v10; // rax
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r15
  unsigned int v15; // r13d
  __int64 v16; // rax
  const char *v17; // r8
  size_t v18; // r14
  __int32 v19; // ecx
  __int64 v20; // r10
  __int64 v21; // r9
  __m128i *v22; // r12
  __int64 v23; // rdx
  __m128i *v24; // rdx
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rax
  const __m128i *v30; // rsi
  __m128i *v31; // rax
  const __m128i *i; // r11
  int v33; // [rsp+Ch] [rbp-64h]
  int v34; // [rsp+10h] [rbp-60h]
  const char *v35; // [rsp+10h] [rbp-60h]
  const char *v36; // [rsp+18h] [rbp-58h]
  __int64 v37; // [rsp+18h] [rbp-58h]
  __int64 v38; // [rsp+20h] [rbp-50h]
  __int64 v39; // [rsp+20h] [rbp-50h]
  int v40; // [rsp+28h] [rbp-48h]
  __int64 v41; // [rsp+30h] [rbp-40h]
  unsigned __int64 v42; // [rsp+30h] [rbp-40h]
  __int64 v44; // [rsp+38h] [rbp-38h]

  v9 = strlen(a2);
  sub_16B8280(a1, a2, v9);
  v10 = *a3;
  v11 = **a3;
  *(_BYTE *)(a1 + 180) = 1;
  *(_DWORD *)(a1 + 160) = v11;
  *(_DWORD *)(a1 + 176) = *v10;
  *(_BYTE *)(a1 + 12) = (32 * (*a4 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v12 = a5[1];
  *(_QWORD *)(a1 + 40) = *a5;
  *(_QWORD *)(a1 + 48) = v12;
  v13 = *a6;
  v44 = *a6 + 40LL * *((unsigned int *)a6 + 2);
  if ( v13 != v44 )
  {
    v14 = v13;
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
        v34 = *(_DWORD *)(v14 + 16);
        v36 = *(const char **)v14;
        v38 = *(_QWORD *)(v14 + 24);
        v41 = *(_QWORD *)(v14 + 32);
        v25 = ((((unsigned __int64)(v16 + 2) >> 1) | (v16 + 2)) >> 2) | ((unsigned __int64)(v16 + 2) >> 1) | (v16 + 2);
        v26 = (((v25 >> 4) | v25) >> 8) | (v25 >> 4) | v25;
        v27 = (v26 | (v26 >> 16) | HIDWORD(v26)) + 1;
        v28 = 0xFFFFFFFFLL;
        if ( v27 <= 0xFFFFFFFF )
          v28 = v27;
        v40 = v28;
        v29 = malloc(48 * v28);
        v21 = v41;
        v20 = v38;
        v17 = v36;
        v19 = v34;
        v22 = (__m128i *)v29;
        if ( !v29 )
        {
          sub_16BD1C0("Allocation failed", 1u);
          v15 = *(_DWORD *)(a1 + 208);
          v19 = v34;
          v17 = v36;
          v20 = v38;
          v21 = v41;
        }
        v30 = *(const __m128i **)(a1 + 200);
        v31 = v22;
        v23 = 3LL * v15;
        v42 = (unsigned __int64)v30;
        for ( i = &v30[v23]; i != v30; v31 += 3 )
        {
          if ( v31 )
          {
            *v31 = _mm_loadu_si128(v30);
            v31[1] = _mm_loadu_si128(v30 + 1);
            v31[2].m128i_i32[2] = v30[2].m128i_i32[2];
            v31[2].m128i_i8[12] = v30[2].m128i_i8[12];
            v31[2].m128i_i64[0] = (__int64)&unk_49FF3D8;
          }
          v30 += 3;
        }
        if ( v42 != a1 + 216 )
        {
          v33 = v19;
          v35 = v17;
          v37 = v20;
          v39 = v21;
          _libc_free(v42);
          v19 = v33;
          v17 = v35;
          v20 = v37;
          v21 = v39;
          v15 = *(_DWORD *)(a1 + 208);
          v23 = 3LL * v15;
        }
        *(_QWORD *)(a1 + 200) = v22;
        *(_DWORD *)(a1 + 212) = v40;
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
        v24[2].m128i_i64[0] = (__int64)&unk_49FF3D8;
        v15 = *(_DWORD *)(a1 + 208);
      }
      v14 += 40;
      *(_DWORD *)(a1 + 208) = v15 + 1;
      sub_16B7FD0(*(_QWORD *)(a1 + 192), v17, v18);
    }
    while ( v44 != v14 );
  }
}
