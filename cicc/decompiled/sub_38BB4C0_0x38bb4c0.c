// Function: sub_38BB4C0
// Address: 0x38bb4c0
//
__int64 __fastcall sub_38BB4C0(__int64 a1, const char *a2, _DWORD *a3, _QWORD *a4, __int64 *a5, unsigned int **a6)
{
  size_t v9; // rax
  __int64 v10; // rax
  __int64 v11; // rbx
  unsigned int v12; // r13d
  __int64 v13; // rax
  const char *v14; // r8
  size_t v15; // r14
  __int32 v16; // ecx
  __int64 v17; // r10
  __int64 v18; // r9
  __m128i *v19; // r12
  __int64 v20; // rdx
  __m128i *v21; // rdx
  unsigned int *v22; // rax
  unsigned int v23; // edx
  __int64 result; // rax
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rax
  const __m128i *v30; // rsi
  __m128i *v31; // rax
  const __m128i *i; // r11
  int v33; // [rsp+4h] [rbp-6Ch]
  int v34; // [rsp+8h] [rbp-68h]
  const char *v35; // [rsp+8h] [rbp-68h]
  const char *v36; // [rsp+10h] [rbp-60h]
  __int64 v37; // [rsp+10h] [rbp-60h]
  __int64 v38; // [rsp+18h] [rbp-58h]
  __int64 v39; // [rsp+18h] [rbp-58h]
  int v40; // [rsp+20h] [rbp-50h]
  __int64 v42; // [rsp+30h] [rbp-40h]
  unsigned __int64 v43; // [rsp+30h] [rbp-40h]
  __int64 v44; // [rsp+38h] [rbp-38h]

  v9 = strlen(a2);
  sub_16B8280(a1, a2, v9);
  *(_BYTE *)(a1 + 12) = (32 * (*a3 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v10 = a4[1];
  *(_QWORD *)(a1 + 40) = *a4;
  *(_QWORD *)(a1 + 48) = v10;
  v11 = *a5;
  v44 = *a5 + 40LL * *((unsigned int *)a5 + 2);
  if ( *a5 != v44 )
  {
    do
    {
      v12 = *(_DWORD *)(a1 + 208);
      v13 = *(unsigned int *)(a1 + 212);
      v14 = *(const char **)v11;
      v15 = *(_QWORD *)(v11 + 8);
      v16 = *(_DWORD *)(v11 + 16);
      v17 = *(_QWORD *)(v11 + 24);
      v18 = *(_QWORD *)(v11 + 32);
      if ( v12 >= (unsigned int)v13 )
      {
        v34 = *(_DWORD *)(v11 + 16);
        v36 = *(const char **)v11;
        v38 = *(_QWORD *)(v11 + 24);
        v42 = *(_QWORD *)(v11 + 32);
        v25 = ((((unsigned __int64)(v13 + 2) >> 1) | (v13 + 2)) >> 2) | ((unsigned __int64)(v13 + 2) >> 1) | (v13 + 2);
        v26 = (((v25 >> 4) | v25) >> 8) | (v25 >> 4) | v25;
        v27 = (v26 | (v26 >> 16) | HIDWORD(v26)) + 1;
        v28 = 0xFFFFFFFFLL;
        if ( v27 <= 0xFFFFFFFF )
          v28 = v27;
        v40 = v28;
        v29 = malloc(48 * v28);
        v18 = v42;
        v17 = v38;
        v14 = v36;
        v16 = v34;
        v19 = (__m128i *)v29;
        if ( !v29 )
        {
          sub_16BD1C0("Allocation failed", 1u);
          v12 = *(_DWORD *)(a1 + 208);
          v16 = v34;
          v14 = v36;
          v17 = v38;
          v18 = v42;
        }
        v30 = *(const __m128i **)(a1 + 200);
        v31 = v19;
        v20 = 3LL * v12;
        v43 = (unsigned __int64)v30;
        for ( i = &v30[v20]; i != v30; v31 += 3 )
        {
          if ( v31 )
          {
            *v31 = _mm_loadu_si128(v30);
            v31[1] = _mm_loadu_si128(v30 + 1);
            v31[2].m128i_i32[2] = v30[2].m128i_i32[2];
            v31[2].m128i_i8[12] = v30[2].m128i_i8[12];
            v31[2].m128i_i64[0] = (__int64)&unk_4A3DEB8;
          }
          v30 += 3;
        }
        if ( v43 != a1 + 216 )
        {
          v33 = v16;
          v35 = v14;
          v37 = v17;
          v39 = v18;
          _libc_free(v43);
          v16 = v33;
          v14 = v35;
          v17 = v37;
          v18 = v39;
          v12 = *(_DWORD *)(a1 + 208);
          v20 = 3LL * v12;
        }
        *(_QWORD *)(a1 + 200) = v19;
        *(_DWORD *)(a1 + 212) = v40;
      }
      else
      {
        v19 = *(__m128i **)(a1 + 200);
        v20 = 3LL * v12;
      }
      v21 = &v19[v20];
      if ( v21 )
      {
        v21->m128i_i64[0] = (__int64)v14;
        v21->m128i_i64[1] = v15;
        v21[1].m128i_i64[0] = v17;
        v21[1].m128i_i64[1] = v18;
        v21[2].m128i_i32[2] = v16;
        v21[2].m128i_i8[12] = 1;
        v21[2].m128i_i64[0] = (__int64)&unk_4A3DEB8;
        v12 = *(_DWORD *)(a1 + 208);
      }
      v11 += 40;
      *(_DWORD *)(a1 + 208) = v12 + 1;
      sub_16B7FD0(*(_QWORD *)(a1 + 192), v14, v15);
    }
    while ( v44 != v11 );
  }
  v22 = *a6;
  v23 = **a6;
  *(_BYTE *)(a1 + 180) = 1;
  *(_DWORD *)(a1 + 160) = v23;
  result = *v22;
  *(_DWORD *)(a1 + 176) = result;
  return result;
}
