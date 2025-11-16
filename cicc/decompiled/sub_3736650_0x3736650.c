// Function: sub_3736650
// Address: 0x3736650
//
void __fastcall sub_3736650(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, unsigned __int8 *a5)
{
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // r10
  __m128i *v12; // rax
  __m128i *v13; // rcx
  __m128i *v14; // rdx
  int v15; // eax
  unsigned int v16; // r8d
  _QWORD *v17; // r9
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned int v20; // r8d
  _QWORD *v21; // r9
  _QWORD *v22; // rcx
  __int64 *v23; // rdx
  _QWORD *v24; // [rsp+0h] [rbp-B0h]
  _QWORD *v25; // [rsp+8h] [rbp-A8h]
  unsigned int v26; // [rsp+14h] [rbp-9Ch]
  __m128i *src; // [rsp+20h] [rbp-90h]
  size_t n; // [rsp+28h] [rbp-88h]
  __m128i v30; // [rsp+30h] [rbp-80h] BYREF
  __m128i v31; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v32[2]; // [rsp+50h] [rbp-60h] BYREF
  char *v33; // [rsp+60h] [rbp-50h] BYREF
  size_t v34; // [rsp+68h] [rbp-48h]
  _QWORD v35[8]; // [rsp+70h] [rbp-40h] BYREF

  if ( sub_37365C0((_QWORD *)a1) )
  {
    if ( a2 )
    {
      v33 = (char *)v35;
      sub_3735130((__int64 *)&v33, a2, (__int64)&a2[a3]);
    }
    else
    {
      v34 = 0;
      v33 = (char *)v35;
      LOBYTE(v35[0]) = 0;
    }
    sub_3248550(&v31, (char *)a1, a5, v7, v8, v9);
    v10 = 15;
    v11 = 15;
    if ( (_QWORD *)v31.m128i_i64[0] != v32 )
      v11 = v32[0];
    if ( v31.m128i_i64[1] + v34 <= v11 )
      goto LABEL_11;
    if ( v33 != (char *)v35 )
      v10 = v35[0];
    if ( v31.m128i_i64[1] + v34 <= v10 )
    {
      v12 = (__m128i *)sub_2241130((unsigned __int64 *)&v33, 0, 0, v31.m128i_i64[0], v31.m128i_u64[1]);
      v14 = v12 + 1;
      src = &v30;
      v13 = (__m128i *)v12->m128i_i64[0];
      if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
        goto LABEL_12;
    }
    else
    {
LABEL_11:
      v12 = (__m128i *)sub_2241490((unsigned __int64 *)&v31, v33, v34);
      src = &v30;
      v13 = (__m128i *)v12->m128i_i64[0];
      v14 = v12 + 1;
      if ( (__m128i *)v12->m128i_i64[0] != &v12[1] )
      {
LABEL_12:
        src = v13;
        v30.m128i_i64[0] = v12[1].m128i_i64[0];
        goto LABEL_13;
      }
    }
    v30 = _mm_loadu_si128(v12 + 1);
LABEL_13:
    n = v12->m128i_u64[1];
    v12->m128i_i64[0] = (__int64)v14;
    v12->m128i_i64[1] = 0;
    v12[1].m128i_i8[0] = 0;
    if ( (_QWORD *)v31.m128i_i64[0] != v32 )
      j_j___libc_free_0(v31.m128i_u64[0]);
    if ( v33 != (char *)v35 )
      j_j___libc_free_0((unsigned __int64)v33);
    v15 = sub_C92610();
    v16 = sub_C92740(a1 + 424, src, n, v15);
    v17 = (_QWORD *)(*(_QWORD *)(a1 + 424) + 8LL * v16);
    v18 = *v17;
    if ( *v17 )
    {
      if ( v18 != -8 )
      {
LABEL_19:
        *(_QWORD *)(v18 + 8) = a4;
        if ( src != &v30 )
          j_j___libc_free_0((unsigned __int64)src);
        return;
      }
      --*(_DWORD *)(a1 + 440);
    }
    v25 = v17;
    v26 = v16;
    v19 = sub_C7D670(n + 17, 8);
    v20 = v26;
    v21 = v25;
    v22 = (_QWORD *)v19;
    if ( n )
    {
      v24 = (_QWORD *)v19;
      memcpy((void *)(v19 + 16), src, n);
      v20 = v26;
      v21 = v25;
      v22 = v24;
    }
    *((_BYTE *)v22 + n + 16) = 0;
    *v22 = n;
    v22[1] = 0;
    *v21 = v22;
    ++*(_DWORD *)(a1 + 436);
    v23 = (__int64 *)(*(_QWORD *)(a1 + 424) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a1 + 424), v20));
    v18 = *v23;
    if ( !*v23 || v18 == -8 )
    {
      do
      {
        do
        {
          v18 = v23[1];
          ++v23;
        }
        while ( v18 == -8 );
      }
      while ( !v18 );
    }
    goto LABEL_19;
  }
}
