// Function: sub_E0D940
// Address: 0xe0d940
//
_BYTE *__fastcall sub_E0D940(unsigned __int64 a1, _WORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rcx
  __m128i *v10; // rsi
  __m128i v11; // rax
  __int64 v12; // r12
  int v13; // ecx
  _BYTE *v14; // rdi
  unsigned __int64 v16; // rsi
  _BYTE *v17; // rdi
  unsigned __int64 v18; // rdx
  _WORD *v19; // rax
  __int64 v20; // [rsp-A0h] [rbp-A0h] BYREF
  __m128i v21; // [rsp-98h] [rbp-98h] BYREF
  __m128i v22; // [rsp-88h] [rbp-88h] BYREF
  unsigned __int64 v23; // [rsp-78h] [rbp-78h] BYREF
  _WORD *v24; // [rsp-70h] [rbp-70h]
  int v25; // [rsp-68h] [rbp-68h]
  _BYTE *v26; // [rsp-58h] [rbp-58h] BYREF
  unsigned __int64 v27; // [rsp-50h] [rbp-50h]
  unsigned __int64 v28; // [rsp-48h] [rbp-48h]
  __int64 v29; // [rsp-40h] [rbp-40h]
  int v30; // [rsp-38h] [rbp-38h]

  if ( a1 <= 1 || *a2 != 17503 )
    return 0;
  v6 = a1;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = -1;
  v30 = 1;
  if ( a1 == 6 && *(_DWORD *)a2 == 1634550879 && a2[2] == 28265 )
  {
    v28 = 998;
    v19 = (_WORD *)malloc(998, a2, a3, a4, a5, a6);
    v26 = v19;
    v14 = v19;
    if ( v19 )
    {
      *(_DWORD *)v19 = 1634541636;
      v19[2] = 28265;
      goto LABEL_42;
    }
    goto LABEL_50;
  }
  v23 = a1;
  v7 = 0;
  v24 = a2;
  v25 = a1;
  v8 = a1 - 2;
  v21.m128i_i64[1] = (__int64)(a2 + 1);
  for ( v21.m128i_i64[0] = v8; ; v8 = v21.m128i_i64[0] )
  {
    if ( v8 )
    {
      v9 = v21.m128i_i64[1];
      if ( *(_BYTE *)v21.m128i_i64[1] == 48 )
      {
        v10 = (__m128i *)(v21.m128i_i64[1] + v8);
        v11.m128i_i64[1] = v21.m128i_i64[1];
        while ( 1 )
        {
          ++v11.m128i_i64[1];
          v11.m128i_i64[0] = v8 + v9 - v11.m128i_i64[1];
          v21 = v11;
          if ( (__m128i *)v11.m128i_i64[1] == v10 )
            break;
          if ( *(_BYTE *)v11.m128i_i64[1] != 48 )
            goto LABEL_10;
        }
LABEL_15:
        _libc_free(v26, v10);
        return 0;
      }
    }
    if ( v7 )
    {
      v16 = v27;
      v17 = v26;
      v18 = v27 + 1;
      if ( v27 + 1 > v28 )
      {
        if ( v27 + 993 <= 2 * v28 )
          v28 *= 2LL;
        else
          v28 = v27 + 993;
        v26 = (_BYTE *)realloc(v26);
        v17 = v26;
        if ( !v26 )
          goto LABEL_50;
        v16 = v27;
        v18 = v27 + 1;
      }
      v27 = v18;
      v17[v16] = 46;
    }
    v10 = (__m128i *)&v26;
    sub_E0D760((__int64)&v23, (void **)&v26, (unsigned __int64 *)&v21);
    v11.m128i_i64[0] = v21.m128i_i64[0];
    if ( !v21.m128i_i64[0] )
      goto LABEL_15;
    v7 = 1;
LABEL_10:
    v22 = _mm_loadu_si128(&v21);
    v12 = v22.m128i_i64[1];
    v13 = *(char *)v22.m128i_i64[1];
    if ( (unsigned int)(v13 - 48) > 9 )
    {
      if ( (_BYTE)v13 != 81 )
        goto LABEL_12;
      ++v22.m128i_i64[1];
      v22.m128i_i64[0] = v11.m128i_i64[0] - 1;
      if ( v11.m128i_i64[0] == 1 )
        goto LABEL_12;
      v10 = (__m128i *)&v20;
      if ( !(unsigned __int8)sub_E0CFB0(v22.m128i_i64, &v20)
        || v12 - (__int64)v24 < v20
        || (unsigned int)(*(char *)(v12 - v20) - 48) > 9 )
      {
        break;
      }
    }
  }
  if ( !v21.m128i_i64[0] )
    goto LABEL_15;
LABEL_12:
  if ( *(_BYTE *)v21.m128i_i64[1] != 90 )
  {
    v10 = &v21;
    if ( (unsigned __int8)sub_E0D140((__int64)&v23, v21.m128i_i64) && v21.m128i_i64[1] && !*(_BYTE *)v21.m128i_i64[1] )
      goto LABEL_14;
    goto LABEL_15;
  }
  if ( *(_BYTE *)(v21.m128i_i64[1] + 1) )
    goto LABEL_15;
LABEL_14:
  v6 = v27;
  v14 = v26;
  if ( !v27 )
    goto LABEL_15;
  if ( v27 + 1 > v28 )
  {
    if ( v27 + 993 > 2 * v28 )
      v28 = v27 + 993;
    else
      v28 *= 2LL;
    v26 = (_BYTE *)realloc(v26);
    v14 = v26;
    if ( v26 )
    {
      v6 = v27;
      goto LABEL_42;
    }
LABEL_50:
    abort();
  }
LABEL_42:
  v14[v6] = 0;
  return v26;
}
