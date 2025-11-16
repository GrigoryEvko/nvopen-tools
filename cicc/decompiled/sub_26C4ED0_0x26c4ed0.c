// Function: sub_26C4ED0
// Address: 0x26c4ed0
//
__m128i *__fastcall sub_26C4ED0(__int64 a1, int *a2, size_t a3, int *a4, size_t a5, __int64 a6)
{
  int *v6; // r11
  __m128i *result; // rax
  __int64 v12; // rcx
  int v13; // esi
  unsigned int v14; // edx
  __int8 *v15; // r15
  __int64 v16; // rdi
  __int64 *v17; // rdx
  __int64 v18; // rax
  unsigned int v19; // eax
  __int64 *v20; // rdi
  __int64 v21; // r8
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rbx
  char v25; // dl
  int *v26; // rax
  int v27; // edi
  int v28; // r9d
  int v29; // r10d
  size_t v30; // [rsp+8h] [rbp-E8h]
  int *v31; // [rsp+8h] [rbp-E8h]
  _QWORD v32[2]; // [rsp+10h] [rbp-E0h] BYREF
  __m128i v33; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v34; // [rsp+30h] [rbp-C0h]

  v6 = a2;
  if ( a4 )
  {
    v30 = a5;
    sub_C7D030(&v33);
    sub_C7D280(v33.m128i_i32, a4, v30);
    sub_C7D290(&v33, v32);
    a5 = v32[0];
    v6 = a2;
  }
  result = (__m128i *)*(unsigned int *)(a1 + 112);
  v12 = *(_QWORD *)(a1 + 96);
  if ( (_DWORD)result )
  {
    v13 = (_DWORD)result - 1;
    v14 = ((_DWORD)result - 1) & (((0xBF58476D1CE4E5B9LL * a5) >> 31) ^ (484763065 * a5));
    v15 = (__int8 *)(v12 + 16LL * v14);
    v16 = *(_QWORD *)v15;
    if ( a5 == *(_QWORD *)v15 )
    {
LABEL_5:
      result = (__m128i *)(16LL * (_QWORD)result);
      v17 = (__int64 *)((char *)result->m128i_i64 + v12);
      if ( &result->m128i_i8[v12] == v15 )
        return result;
      if ( v6 )
      {
        v31 = v6;
        sub_C7D030(&v33);
        sub_C7D280(v33.m128i_i32, v31, a3);
        sub_C7D290(&v33, v32);
        v12 = *(_QWORD *)(a1 + 96);
        a3 = v32[0];
        v18 = *(unsigned int *)(a1 + 112);
        v17 = (__int64 *)(v12 + 16 * v18);
        if ( !(_DWORD)v18 )
          goto LABEL_16;
        v13 = v18 - 1;
      }
      v19 = v13 & (((0xBF58476D1CE4E5B9LL * a3) >> 31) ^ (484763065 * a3));
      v20 = (__int64 *)(v12 + 16LL * v19);
      v21 = *v20;
      if ( a3 == *v20 )
      {
LABEL_10:
        if ( v17 != v20 )
          goto LABEL_11;
      }
      else
      {
        v27 = 1;
        while ( v21 != -1 )
        {
          v29 = v27 + 1;
          v19 = v13 & (v27 + v19);
          v20 = (__int64 *)(v12 + 16LL * v19);
          v21 = *v20;
          if ( a3 == *v20 )
            goto LABEL_10;
          v27 = v29;
        }
      }
LABEL_16:
      v20 = v17;
LABEL_11:
      v22 = v20[1];
      v23 = *((_QWORD *)v15 + 1);
      v34 = a6;
      v33.m128i_i64[0] = v22;
      v33.m128i_i64[1] = v23;
      v24 = v20[1];
      result = sub_26C4E00(v24 + 16, &v33);
      if ( !v25 )
      {
        v34 += result[3].m128i_i64[0];
        v26 = sub_220F330(result->m128i_i32, (_QWORD *)(v24 + 24));
        j_j___libc_free_0((unsigned __int64)v26);
        --*(_QWORD *)(v24 + 56);
        return sub_26C4E00(v24 + 16, &v33);
      }
      return result;
    }
    v28 = 1;
    while ( v16 != -1 )
    {
      v14 = v13 & (v28 + v14);
      v15 = (__int8 *)(v12 + 16LL * v14);
      v16 = *(_QWORD *)v15;
      if ( a5 == *(_QWORD *)v15 )
        goto LABEL_5;
      ++v28;
    }
  }
  return result;
}
