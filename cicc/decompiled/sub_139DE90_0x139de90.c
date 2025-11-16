// Function: sub_139DE90
// Address: 0x139de90
//
__int64 __fastcall sub_139DE90(__int64 a1, const char *a2, _QWORD *a3, int **a4, __int64 a5)
{
  size_t v8; // rax
  __int64 v9; // rax
  int *v10; // rax
  int v11; // edx
  __int64 result; // rax
  __int64 v13; // r15
  unsigned int v14; // r13d
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // r14
  __int64 v18; // rcx
  __int64 v19; // r10
  __int64 v20; // r9
  __m128i *v21; // r12
  __int64 v22; // rdx
  __m128i *v23; // rdx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rax
  const __m128i *v29; // rsi
  __m128i *v30; // rax
  const __m128i *i; // r11
  unsigned int v32; // [rsp+Ch] [rbp-64h]
  unsigned int v33; // [rsp+10h] [rbp-60h]
  __int64 v34; // [rsp+10h] [rbp-60h]
  __int64 v35; // [rsp+18h] [rbp-58h]
  __int64 v36; // [rsp+18h] [rbp-58h]
  __int64 v37; // [rsp+20h] [rbp-50h]
  __int64 v38; // [rsp+20h] [rbp-50h]
  int v39; // [rsp+28h] [rbp-48h]
  __int64 v40; // [rsp+30h] [rbp-40h]
  unsigned __int64 v41; // [rsp+30h] [rbp-40h]
  __int64 v42; // [rsp+38h] [rbp-38h]

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
  v42 = *(_QWORD *)a5 + 40LL * *(unsigned int *)(a5 + 8);
  if ( *(_QWORD *)a5 != v42 )
  {
    v13 = *(_QWORD *)a5;
    do
    {
      v14 = *(_DWORD *)(a1 + 208);
      v15 = *(unsigned int *)(a1 + 212);
      v16 = *(_QWORD *)v13;
      v17 = *(_QWORD *)(v13 + 8);
      v18 = *(unsigned int *)(v13 + 16);
      v19 = *(_QWORD *)(v13 + 24);
      v20 = *(_QWORD *)(v13 + 32);
      if ( v14 >= (unsigned int)v15 )
      {
        v33 = *(_DWORD *)(v13 + 16);
        v35 = *(_QWORD *)v13;
        v37 = *(_QWORD *)(v13 + 24);
        v40 = *(_QWORD *)(v13 + 32);
        v24 = ((((unsigned __int64)(v15 + 2) >> 1) | (v15 + 2)) >> 2) | ((unsigned __int64)(v15 + 2) >> 1) | (v15 + 2);
        v25 = (((v24 >> 4) | v24) >> 8) | (v24 >> 4) | v24;
        v26 = (v25 | (v25 >> 16) | HIDWORD(v25)) + 1;
        v27 = 0xFFFFFFFFLL;
        if ( v26 <= 0xFFFFFFFF )
          v27 = v26;
        v39 = v27;
        v28 = malloc(48 * v27);
        v20 = v40;
        v19 = v37;
        v16 = v35;
        v18 = v33;
        v21 = (__m128i *)v28;
        if ( !v28 )
        {
          sub_16BD1C0("Allocation failed");
          v14 = *(_DWORD *)(a1 + 208);
          v18 = v33;
          v16 = v35;
          v19 = v37;
          v20 = v40;
        }
        v29 = *(const __m128i **)(a1 + 200);
        v30 = v21;
        v22 = 3LL * v14;
        v41 = (unsigned __int64)v29;
        for ( i = &v29[v22]; i != v29; v30 += 3 )
        {
          if ( v30 )
          {
            *v30 = _mm_loadu_si128(v29);
            v30[1] = _mm_loadu_si128(v29 + 1);
            v30[2].m128i_i32[2] = v29[2].m128i_i32[2];
            v30[2].m128i_i8[12] = v29[2].m128i_i8[12];
            v30[2].m128i_i64[0] = (__int64)&unk_49E93C8;
          }
          v29 += 3;
        }
        if ( v41 != a1 + 216 )
        {
          v32 = v18;
          v34 = v16;
          v36 = v19;
          v38 = v20;
          _libc_free(v41);
          v18 = v32;
          v16 = v34;
          v19 = v36;
          v20 = v38;
          v14 = *(_DWORD *)(a1 + 208);
          v22 = 3LL * v14;
        }
        *(_QWORD *)(a1 + 200) = v21;
        *(_DWORD *)(a1 + 212) = v39;
      }
      else
      {
        v21 = *(__m128i **)(a1 + 200);
        v22 = 3LL * v14;
      }
      v23 = &v21[v22];
      if ( v23 )
      {
        v23->m128i_i64[0] = v16;
        v23->m128i_i64[1] = v17;
        v23[1].m128i_i64[0] = v19;
        v23[1].m128i_i64[1] = v20;
        v23[2].m128i_i32[2] = v18;
        v23[2].m128i_i8[12] = 1;
        v23[2].m128i_i64[0] = (__int64)&unk_49E93C8;
        v14 = *(_DWORD *)(a1 + 208);
      }
      v13 += 40;
      *(_DWORD *)(a1 + 208) = v14 + 1;
      result = sub_16B7FD0(*(_QWORD *)(a1 + 192), v16, v17, v18);
    }
    while ( v42 != v13 );
  }
  return result;
}
