// Function: sub_2E94D10
// Address: 0x2e94d10
//
__int64 __fastcall sub_2E94D10(__int64 *a1, unsigned int a2, __int64 a3, _QWORD *a4)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  int v11; // ecx
  unsigned int v12; // esi
  int *v13; // rdx
  int v14; // r8d
  __int64 v15; // rdi
  __int64 v16; // rax
  int v17; // ecx
  __int64 v18; // rsi
  int v19; // ecx
  unsigned int v20; // edx
  int *v21; // rax
  int v22; // edi
  __int64 result; // rax
  _QWORD *v24; // rbx
  _QWORD *i; // r13
  int v26; // ecx
  __m128i *v27; // rsi
  int v28; // edx
  int v29; // r9d
  int v30; // eax
  int v31; // r8d
  __m128i v32; // [rsp+0h] [rbp-50h] BYREF
  __int64 v33; // [rsp+10h] [rbp-40h]

  *(_QWORD *)(*a4 + 8LL * (*(_DWORD *)(a3 + 24) >> 6)) |= 1LL << *(_DWORD *)(a3 + 24);
  v8 = 80LL * *(int *)(a3 + 24);
  v9 = v8 + a1[5];
  if ( (*(_BYTE *)(v9 + 8) & 1) != 0 )
  {
    v10 = v9 + 16;
    v11 = 3;
  }
  else
  {
    v26 = *(_DWORD *)(v9 + 24);
    v10 = *(_QWORD *)(v9 + 16);
    if ( !v26 )
      goto LABEL_5;
    v11 = v26 - 1;
  }
  v12 = v11 & (37 * a2);
  v13 = (int *)(v10 + 16LL * (v11 & (37 * a2)));
  v14 = *v13;
  if ( a2 == *v13 )
  {
LABEL_4:
    v15 = *((_QWORD *)v13 + 1);
    if ( v15 )
      return sub_2E8D6E0(v15, a2, *a1);
  }
  else
  {
    v28 = 1;
    while ( v14 != -1 )
    {
      v29 = v28 + 1;
      v12 = v11 & (v28 + v12);
      v13 = (int *)(v10 + 16LL * v12);
      v14 = *v13;
      if ( a2 == *v13 )
        goto LABEL_4;
      v28 = v29;
    }
  }
LABEL_5:
  v16 = a1[2] + v8;
  if ( (*(_BYTE *)(v16 + 8) & 1) != 0 )
  {
    v18 = v16 + 16;
    v19 = 3;
  }
  else
  {
    v17 = *(_DWORD *)(v16 + 24);
    v18 = *(_QWORD *)(v16 + 16);
    if ( !v17 )
      goto LABEL_11;
    v19 = v17 - 1;
  }
  v20 = v19 & (37 * a2);
  v21 = (int *)(v18 + 16LL * v20);
  v22 = *v21;
  if ( a2 == *v21 )
  {
LABEL_9:
    result = *((_QWORD *)v21 + 1);
    if ( result && a3 == *(_QWORD *)(result + 24) )
      return result;
  }
  else
  {
    v30 = 1;
    while ( v22 != -1 )
    {
      v31 = v30 + 1;
      v20 = v19 & (v30 + v20);
      v21 = (int *)(v18 + 16LL * v20);
      v22 = *v21;
      if ( a2 == *v21 )
        goto LABEL_9;
      v30 = v31;
    }
  }
LABEL_11:
  if ( !(unsigned __int8)sub_2E31DD0(a3, a2, -1, -1) )
  {
    v32.m128i_i64[1] = -1;
    v27 = *(__m128i **)(a3 + 192);
    v32.m128i_i32[0] = (unsigned __int16)a2;
    v33 = -1;
    if ( v27 == *(__m128i **)(a3 + 200) )
    {
      sub_2E341F0((unsigned __int64 *)(a3 + 184), v27, &v32);
    }
    else
    {
      if ( v27 )
      {
        *v27 = _mm_loadu_si128(&v32);
        v27[1].m128i_i64[0] = v33;
        v27 = *(__m128i **)(a3 + 192);
      }
      *(_QWORD *)(a3 + 192) = (char *)v27 + 24;
    }
  }
  v24 = *(_QWORD **)(a3 + 64);
  result = *(unsigned int *)(a3 + 72);
  for ( i = &v24[result]; i != v24; ++v24 )
  {
    result = *(_QWORD *)(*a4 + 8LL * (*(_DWORD *)(*v24 + 24LL) >> 6)) & (1LL << *(_DWORD *)(*v24 + 24LL));
    if ( !result )
      result = sub_2E94D10(a1, a2, *v24, a4, 1, *v24, v32.m128i_i32[0], v32.m128i_i64[1]);
  }
  return result;
}
