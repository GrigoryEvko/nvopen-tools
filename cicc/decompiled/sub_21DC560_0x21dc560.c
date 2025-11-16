// Function: sub_21DC560
// Address: 0x21dc560
//
__int64 __fastcall sub_21DC560(__int64 a1, char *a2)
{
  __int64 v3; // rbx
  __int64 v4; // rdx
  __m128i *v5; // r9
  __int64 v6; // rax
  size_t v7; // rdx
  int v8; // eax
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __m128i *v15; // rax
  __m128i *v16; // rdi
  __m128i *v17; // [rsp+8h] [rbp-68h]
  __int64 v18; // [rsp+10h] [rbp-60h]
  unsigned int v19; // [rsp+1Ch] [rbp-54h]
  void *s2; // [rsp+20h] [rbp-50h] BYREF
  __int64 v21; // [rsp+28h] [rbp-48h]
  __m128i v22[4]; // [rsp+30h] [rbp-40h] BYREF

  if ( *(_DWORD *)(a1 + 16) )
  {
    v18 = *(unsigned int *)(a1 + 16);
    v3 = 0;
    while ( 1 )
    {
      v19 = v3;
      v4 = -1;
      s2 = v22;
      if ( a2 )
        v4 = (__int64)&a2[strlen(a2)];
      sub_21DBEB0((__int64 *)&s2, a2, v4);
      v5 = (__m128i *)s2;
      v6 = *(_QWORD *)(a1 + 8) + 32 * v3;
      v7 = *(_QWORD *)(v6 + 8);
      if ( v7 == v21 )
      {
        if ( !v7 )
          break;
        v17 = (__m128i *)s2;
        v8 = memcmp(*(const void **)v6, s2, v7);
        v5 = v17;
        if ( !v8 )
          break;
      }
      if ( v5 != v22 )
        j_j___libc_free_0(v5, v22[0].m128i_i64[0] + 1);
      if ( v18 == ++v3 )
        goto LABEL_14;
    }
    if ( v5 != v22 )
      j_j___libc_free_0(v5, v22[0].m128i_i64[0] + 1);
  }
  else
  {
LABEL_14:
    s2 = v22;
    v10 = -1;
    if ( a2 )
      v10 = (__int64)&a2[strlen(a2)];
    sub_21DBEB0((__int64 *)&s2, a2, v10);
    v19 = *(_DWORD *)(a1 + 16);
    if ( v19 >= *(_DWORD *)(a1 + 20) )
    {
      sub_12BE710(a1 + 8, 0, v11, v12, v13, v14);
      v19 = *(_DWORD *)(a1 + 16);
    }
    v15 = (__m128i *)(*(_QWORD *)(a1 + 8) + 32LL * v19);
    if ( v15 )
    {
      v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
      if ( s2 == v22 )
      {
        v15[1] = _mm_load_si128(v22);
      }
      else
      {
        v15->m128i_i64[0] = (__int64)s2;
        v15[1].m128i_i64[0] = v22[0].m128i_i64[0];
      }
      v15->m128i_i64[1] = v21;
      v19 = *(_DWORD *)(a1 + 16);
      *(_DWORD *)(a1 + 16) = v19 + 1;
    }
    else
    {
      v16 = (__m128i *)s2;
      *(_DWORD *)(a1 + 16) = v19 + 1;
      if ( v16 != v22 )
      {
        j_j___libc_free_0(v16, v22[0].m128i_i64[0] + 1);
        return (unsigned int)(*(_DWORD *)(a1 + 16) - 1);
      }
    }
  }
  return v19;
}
