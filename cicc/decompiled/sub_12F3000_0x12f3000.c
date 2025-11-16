// Function: sub_12F3000
// Address: 0x12f3000
//
__int64 __fastcall sub_12F3000(__int64 a1, const char *a2, _QWORD *a3, int **a4, __int64 a5)
{
  size_t v8; // rax
  __int64 v9; // rax
  int *v10; // rax
  int v11; // edx
  __int64 result; // rax
  __int64 v13; // r15
  unsigned int v14; // r13d
  __int64 v15; // rax
  const char *v16; // r8
  __int64 v17; // r14
  __int64 v18; // rcx
  __int64 v19; // r10
  __int64 v20; // r9
  __m128i *v21; // r12
  __int64 v22; // rdx
  __m128i *v23; // rdx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rax
  const __m128i *v30; // rsi
  __m128i *v31; // rax
  const __m128i *i; // r11
  unsigned int v33; // [rsp+Ch] [rbp-64h]
  unsigned int v34; // [rsp+10h] [rbp-60h]
  const char *v35; // [rsp+10h] [rbp-60h]
  const char *v36; // [rsp+18h] [rbp-58h]
  __int64 v37; // [rsp+18h] [rbp-58h]
  __int64 v38; // [rsp+20h] [rbp-50h]
  __int64 v39; // [rsp+20h] [rbp-50h]
  int v40; // [rsp+28h] [rbp-48h]
  __int64 v41; // [rsp+30h] [rbp-40h]
  const __m128i *v42; // [rsp+30h] [rbp-40h]
  __int64 v43; // [rsp+38h] [rbp-38h]

  v8 = strlen(a2);
  sub_16B8280(a1, a2, v8);
  v9 = a3[1];
  *(_QWORD *)(a1 + 40) = *a3;
  *(_QWORD *)(a1 + 48) = v9;
  v10 = *a4;
  v11 = **a4;
  *(_BYTE *)(a1 + 180) = 1;
  *(_DWORD *)(a1 + 160) = v11;
  *(_DWORD *)(a1 + 176) = *v10;
  result = *(_QWORD *)a5;
  v43 = *(_QWORD *)a5 + 40LL * *(unsigned int *)(a5 + 8);
  if ( *(_QWORD *)a5 != v43 )
  {
    v13 = *(_QWORD *)a5;
    do
    {
      v14 = *(_DWORD *)(a1 + 208);
      v15 = *(unsigned int *)(a1 + 212);
      v16 = *(const char **)v13;
      v17 = *(_QWORD *)(v13 + 8);
      v18 = *(unsigned int *)(v13 + 16);
      v19 = *(_QWORD *)(v13 + 24);
      v20 = *(_QWORD *)(v13 + 32);
      if ( v14 >= (unsigned int)v15 )
      {
        v34 = *(_DWORD *)(v13 + 16);
        v36 = *(const char **)v13;
        v38 = *(_QWORD *)(v13 + 24);
        v41 = *(_QWORD *)(v13 + 32);
        v24 = ((((unsigned __int64)(v15 + 2) >> 1) | (v15 + 2)) >> 2) | ((unsigned __int64)(v15 + 2) >> 1) | (v15 + 2);
        v25 = (((v24 >> 4) | v24) >> 8) | (v24 >> 4) | v24;
        v26 = v25 | (v25 >> 16);
        v27 = (v26 | HIDWORD(v25)) + 1;
        v28 = 0xFFFFFFFFLL;
        if ( v27 <= 0xFFFFFFFF )
          v28 = v27;
        v40 = v28;
        v29 = malloc(48 * v28, a2, v26, v27, v16, v20);
        v20 = v41;
        v19 = v38;
        v16 = v36;
        v18 = v34;
        v21 = (__m128i *)v29;
        if ( !v29 )
        {
          sub_16BD1C0("Allocation failed");
          v14 = *(_DWORD *)(a1 + 208);
          v18 = v34;
          v16 = v36;
          v19 = v38;
          v20 = v41;
        }
        v30 = *(const __m128i **)(a1 + 200);
        v31 = v21;
        v22 = 3LL * v14;
        v42 = v30;
        for ( i = &v30[v22]; i != v30; v31 += 3 )
        {
          if ( v31 )
          {
            *v31 = _mm_loadu_si128(v30);
            v31[1] = _mm_loadu_si128(v30 + 1);
            v31[2].m128i_i32[2] = v30[2].m128i_i32[2];
            v31[2].m128i_i8[12] = v30[2].m128i_i8[12];
            v31[2].m128i_i64[0] = (__int64)&unk_49E7D08;
          }
          v30 += 3;
        }
        if ( v42 != (const __m128i *)(a1 + 216) )
        {
          v33 = v18;
          v35 = v16;
          v37 = v19;
          v39 = v20;
          _libc_free(v42, v30);
          v18 = v33;
          v16 = v35;
          v19 = v37;
          v20 = v39;
          v14 = *(_DWORD *)(a1 + 208);
          v22 = 3LL * v14;
        }
        *(_QWORD *)(a1 + 200) = v21;
        *(_DWORD *)(a1 + 212) = v40;
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
        v23[2].m128i_i64[0] = (__int64)&unk_49E7D08;
        v14 = *(_DWORD *)(a1 + 208);
      }
      a2 = v16;
      v13 += 40;
      *(_DWORD *)(a1 + 208) = v14 + 1;
      result = sub_16B7FD0(*(_QWORD *)(a1 + 192), v16, v17, v18);
    }
    while ( v43 != v13 );
  }
  return result;
}
