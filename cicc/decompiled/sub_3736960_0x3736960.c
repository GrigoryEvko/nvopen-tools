// Function: sub_3736960
// Address: 0x3736960
//
void __fastcall sub_3736960(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 *a4)
{
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned __int8 v10; // al
  __int64 v11; // r13
  __int64 v12; // rdi
  _BYTE *v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rdi
  __m128i *v17; // rax
  __m128i *v18; // rcx
  __m128i *v19; // rdx
  int v20; // eax
  unsigned int v21; // r9d
  _QWORD *v22; // r10
  __int64 v23; // rdx
  __int64 v24; // rax
  unsigned int v25; // r9d
  _QWORD *v26; // r10
  _QWORD *v27; // rcx
  __int64 *v28; // rax
  __int64 *v29; // rax
  _QWORD *v30; // [rsp+0h] [rbp-B0h]
  _QWORD *v31; // [rsp+8h] [rbp-A8h]
  unsigned int v32; // [rsp+14h] [rbp-9Ch]
  __m128i *src; // [rsp+20h] [rbp-90h]
  size_t n; // [rsp+28h] [rbp-88h]
  __m128i v35; // [rsp+30h] [rbp-80h] BYREF
  char *v36; // [rsp+40h] [rbp-70h] BYREF
  size_t v37; // [rsp+48h] [rbp-68h]
  _QWORD v38[2]; // [rsp+50h] [rbp-60h] BYREF
  __m128i v39; // [rsp+60h] [rbp-50h] BYREF
  _QWORD v40[8]; // [rsp+70h] [rbp-40h] BYREF

  if ( sub_37365C0((_QWORD *)a1) )
  {
    v10 = *(_BYTE *)(a2 - 16);
    if ( (v10 & 2) != 0 )
      v11 = *(_QWORD *)(a2 - 32);
    else
      v11 = a2 - 16 - 8LL * ((v10 >> 2) & 0xF);
    v12 = *(_QWORD *)(v11 + 16);
    if ( v12 && (v13 = (_BYTE *)sub_B91420(v12)) != 0 )
    {
      v36 = (char *)v38;
      sub_3735130((__int64 *)&v36, v13, (__int64)&v13[v14]);
    }
    else
    {
      v37 = 0;
      v36 = (char *)v38;
      LOBYTE(v38[0]) = 0;
    }
    sub_3248550(&v39, (char *)a1, a4, v7, v8, v9);
    v15 = 15;
    v16 = 15;
    if ( (_QWORD *)v39.m128i_i64[0] != v40 )
      v16 = v40[0];
    if ( v39.m128i_i64[1] + v37 <= v16 )
      goto LABEL_14;
    if ( v36 != (char *)v38 )
      v15 = v38[0];
    if ( v39.m128i_i64[1] + v37 <= v15 )
    {
      v17 = (__m128i *)sub_2241130((unsigned __int64 *)&v36, 0, 0, v39.m128i_i64[0], v39.m128i_u64[1]);
      v19 = v17 + 1;
      src = &v35;
      v18 = (__m128i *)v17->m128i_i64[0];
      if ( (__m128i *)v17->m128i_i64[0] != &v17[1] )
        goto LABEL_15;
    }
    else
    {
LABEL_14:
      v17 = (__m128i *)sub_2241490((unsigned __int64 *)&v39, v36, v37);
      src = &v35;
      v18 = (__m128i *)v17->m128i_i64[0];
      v19 = v17 + 1;
      if ( (__m128i *)v17->m128i_i64[0] != &v17[1] )
      {
LABEL_15:
        src = v18;
        v35.m128i_i64[0] = v17[1].m128i_i64[0];
        goto LABEL_16;
      }
    }
    v35 = _mm_loadu_si128(v17 + 1);
LABEL_16:
    n = v17->m128i_u64[1];
    v17->m128i_i64[0] = (__int64)v19;
    v17->m128i_i64[1] = 0;
    v17[1].m128i_i8[0] = 0;
    if ( (_QWORD *)v39.m128i_i64[0] != v40 )
      j_j___libc_free_0(v39.m128i_u64[0]);
    if ( v36 != (char *)v38 )
      j_j___libc_free_0((unsigned __int64)v36);
    v20 = sub_C92610();
    v21 = sub_C92740(a1 + 448, src, n, v20);
    v22 = (_QWORD *)(*(_QWORD *)(a1 + 448) + 8LL * v21);
    v23 = *v22;
    if ( *v22 )
    {
      if ( v23 != -8 )
      {
LABEL_22:
        *(_QWORD *)(v23 + 8) = a3;
        if ( src != &v35 )
          j_j___libc_free_0((unsigned __int64)src);
        return;
      }
      --*(_DWORD *)(a1 + 464);
    }
    v31 = v22;
    v32 = v21;
    v24 = sub_C7D670(n + 17, 8);
    v25 = v32;
    v26 = v31;
    v27 = (_QWORD *)v24;
    if ( n )
    {
      v30 = (_QWORD *)v24;
      memcpy((void *)(v24 + 16), src, n);
      v25 = v32;
      v26 = v31;
      v27 = v30;
    }
    *((_BYTE *)v27 + n + 16) = 0;
    *v27 = n;
    v27[1] = 0;
    *v26 = v27;
    ++*(_DWORD *)(a1 + 460);
    v28 = (__int64 *)(*(_QWORD *)(a1 + 448) + 8LL * (unsigned int)sub_C929D0((__int64 *)(a1 + 448), v25));
    v23 = *v28;
    if ( *v28 == -8 || !v23 )
    {
      v29 = v28 + 1;
      do
      {
        do
          v23 = *v29++;
        while ( v23 == -8 );
      }
      while ( !v23 );
    }
    goto LABEL_22;
  }
}
