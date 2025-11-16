// Function: sub_1444680
// Address: 0x1444680
//
__int64 __fastcall sub_1444680(__int64 a1, const char *a2, _DWORD **a3, _DWORD *a4, _QWORD *a5, __int64 a6)
{
  size_t v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 result; // rax
  __int64 v14; // r15
  unsigned int v15; // r13d
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // r14
  __int64 v19; // rcx
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
  _DWORD *v33; // rax
  unsigned int v34; // [rsp+Ch] [rbp-84h]
  unsigned int v35; // [rsp+10h] [rbp-80h]
  __int64 v36; // [rsp+10h] [rbp-80h]
  __int64 v37; // [rsp+18h] [rbp-78h]
  __int64 v38; // [rsp+18h] [rbp-78h]
  __int64 v39; // [rsp+20h] [rbp-70h]
  __int64 v40; // [rsp+20h] [rbp-70h]
  int v41; // [rsp+28h] [rbp-68h]
  __int64 v42; // [rsp+30h] [rbp-60h]
  unsigned __int64 v43; // [rsp+30h] [rbp-60h]
  __int64 v45; // [rsp+38h] [rbp-58h]
  const char *v46; // [rsp+40h] [rbp-50h] BYREF
  char v47; // [rsp+50h] [rbp-40h]
  char v48; // [rsp+51h] [rbp-3Fh]

  v9 = strlen(a2);
  sub_16B8280(a1, a2, v9);
  if ( *(_QWORD *)(a1 + 160) )
  {
    v11 = sub_16E8CB0(a1, a2, v10);
    v48 = 1;
    v46 = "cl::location(x) specified more than once!";
    v47 = 3;
    sub_16B1F90(a1, &v46, 0, 0, v11);
  }
  else
  {
    v33 = *a3;
    *(_BYTE *)(a1 + 180) = 1;
    *(_QWORD *)(a1 + 160) = v33;
    *(_DWORD *)(a1 + 176) = *v33;
  }
  *(_BYTE *)(a1 + 12) = (32 * (*a4 & 3)) | *(_BYTE *)(a1 + 12) & 0x9F;
  v12 = a5[1];
  *(_QWORD *)(a1 + 40) = *a5;
  *(_QWORD *)(a1 + 48) = v12;
  result = *(_QWORD *)a6;
  v14 = result;
  v45 = *(_QWORD *)a6 + 40LL * *(unsigned int *)(a6 + 8);
  if ( result != v45 )
  {
    do
    {
      v15 = *(_DWORD *)(a1 + 208);
      v16 = *(unsigned int *)(a1 + 212);
      v17 = *(_QWORD *)v14;
      v18 = *(_QWORD *)(v14 + 8);
      v19 = *(unsigned int *)(v14 + 16);
      v20 = *(_QWORD *)(v14 + 24);
      v21 = *(_QWORD *)(v14 + 32);
      if ( v15 >= (unsigned int)v16 )
      {
        v35 = *(_DWORD *)(v14 + 16);
        v37 = *(_QWORD *)v14;
        v39 = *(_QWORD *)(v14 + 24);
        v42 = *(_QWORD *)(v14 + 32);
        v25 = ((((unsigned __int64)(v16 + 2) >> 1) | (v16 + 2)) >> 2) | ((unsigned __int64)(v16 + 2) >> 1) | (v16 + 2);
        v26 = (((v25 >> 4) | v25) >> 8) | (v25 >> 4) | v25;
        v27 = (v26 | (v26 >> 16) | HIDWORD(v26)) + 1;
        v28 = 0xFFFFFFFFLL;
        if ( v27 <= 0xFFFFFFFF )
          v28 = v27;
        v41 = v28;
        v29 = malloc(48 * v28);
        v21 = v42;
        v20 = v39;
        v17 = v37;
        v19 = v35;
        v22 = (__m128i *)v29;
        if ( !v29 )
        {
          sub_16BD1C0("Allocation failed");
          v15 = *(_DWORD *)(a1 + 208);
          v19 = v35;
          v17 = v37;
          v20 = v39;
          v21 = v42;
        }
        v30 = *(const __m128i **)(a1 + 200);
        v31 = v22;
        v23 = 3LL * v15;
        v43 = (unsigned __int64)v30;
        for ( i = &v30[v23]; i != v30; v31 += 3 )
        {
          if ( v31 )
          {
            *v31 = _mm_loadu_si128(v30);
            v31[1] = _mm_loadu_si128(v30 + 1);
            v31[2].m128i_i32[2] = v30[2].m128i_i32[2];
            v31[2].m128i_i8[12] = v30[2].m128i_i8[12];
            v31[2].m128i_i64[0] = (__int64)&unk_49EBB00;
          }
          v30 += 3;
        }
        if ( v43 != a1 + 216 )
        {
          v34 = v19;
          v36 = v17;
          v38 = v20;
          v40 = v21;
          _libc_free(v43);
          v19 = v34;
          v17 = v36;
          v20 = v38;
          v21 = v40;
          v15 = *(_DWORD *)(a1 + 208);
          v23 = 3LL * v15;
        }
        *(_QWORD *)(a1 + 200) = v22;
        *(_DWORD *)(a1 + 212) = v41;
      }
      else
      {
        v22 = *(__m128i **)(a1 + 200);
        v23 = 3LL * v15;
      }
      v24 = &v22[v23];
      if ( v24 )
      {
        v24->m128i_i64[0] = v17;
        v24->m128i_i64[1] = v18;
        v24[1].m128i_i64[0] = v20;
        v24[1].m128i_i64[1] = v21;
        v24[2].m128i_i32[2] = v19;
        v24[2].m128i_i8[12] = 1;
        v24[2].m128i_i64[0] = (__int64)&unk_49EBB00;
        v15 = *(_DWORD *)(a1 + 208);
      }
      v14 += 40;
      *(_DWORD *)(a1 + 208) = v15 + 1;
      result = sub_16B7FD0(*(_QWORD *)(a1 + 192), v17, v18, v19);
    }
    while ( v45 != v14 );
  }
  return result;
}
