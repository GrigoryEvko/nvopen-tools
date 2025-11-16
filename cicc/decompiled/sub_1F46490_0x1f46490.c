// Function: sub_1F46490
// Address: 0x1f46490
//
__int64 __fastcall sub_1F46490(__int64 a1, _QWORD *a2, char a3, char a4, unsigned __int8 a5)
{
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 result; // rax
  bool v12; // zf
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rax
  unsigned int v17; // r14d
  __int64 v18; // r13
  __int64 v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rdx
  _BYTE *v23; // rsi
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rcx
  __m128i *v26; // rax
  _OWORD *v27; // rsi
  __m128i *v28; // rdx
  _BYTE *v29; // rdi
  size_t v30; // rdx
  __int64 v31; // r8
  __int64 v32; // rax
  unsigned int v33; // [rsp+18h] [rbp-C8h]
  unsigned int v34; // [rsp+2Ch] [rbp-B4h]
  unsigned int v35; // [rsp+2Ch] [rbp-B4h]
  void *dest; // [rsp+30h] [rbp-B0h] BYREF
  size_t v37; // [rsp+38h] [rbp-A8h]
  _QWORD v38[2]; // [rsp+40h] [rbp-A0h] BYREF
  _QWORD *v39; // [rsp+50h] [rbp-90h] BYREF
  __int64 v40; // [rsp+58h] [rbp-88h]
  _QWORD v41[2]; // [rsp+60h] [rbp-80h] BYREF
  const char *v42; // [rsp+70h] [rbp-70h] BYREF
  __int64 v43; // [rsp+78h] [rbp-68h]
  _QWORD v44[2]; // [rsp+80h] [rbp-60h] BYREF
  _OWORD *v45; // [rsp+90h] [rbp-50h]
  size_t n; // [rsp+98h] [rbp-48h]
  _OWORD src[4]; // [rsp+A0h] [rbp-40h] BYREF

  v9 = a2[2];
  v10 = *(_QWORD *)(a1 + 184);
  if ( *(_QWORD *)(a1 + 168) == v9 )
  {
    *(_BYTE *)(a1 + 200) = 1;
    if ( v9 != v10 )
      goto LABEL_14;
LABEL_10:
    *(_BYTE *)(a1 + 201) = 1;
    goto LABEL_4;
  }
  if ( v9 == v10 )
    goto LABEL_10;
  if ( !*(_BYTE *)(a1 + 200) )
  {
LABEL_4:
    (*(void (__fastcall **)(_QWORD *))(*a2 + 8LL))(a2);
    goto LABEL_5;
  }
LABEL_14:
  if ( *(_BYTE *)(a1 + 201) )
    goto LABEL_4;
  v12 = *(_BYTE *)(a1 + 202) == 0;
  v37 = 0;
  dest = v38;
  LOBYTE(v38[0]) = 0;
  v34 = a5;
  if ( v12 )
    goto LABEL_16;
  if ( a4 || a3 )
  {
    v23 = (_BYTE *)(*(__int64 (__fastcall **)(_QWORD *))(*a2 + 16LL))(a2);
    if ( v23 )
    {
      v42 = (const char *)v44;
      sub_1F450A0((__int64 *)&v42, v23, (__int64)&v23[v22]);
    }
    else
    {
      v43 = 0;
      v42 = (const char *)v44;
      LOBYTE(v44[0]) = 0;
    }
    v39 = v41;
    sub_1F450A0((__int64 *)&v39, "After ", (__int64)"");
    v24 = 15;
    v25 = 15;
    if ( v39 != v41 )
      v25 = v41[0];
    if ( v40 + v43 <= v25 )
      goto LABEL_42;
    if ( v42 != (const char *)v44 )
      v24 = v44[0];
    if ( v40 + v43 <= v24 )
    {
      v26 = (__m128i *)sub_2241130(&v42, 0, 0, v39, v40);
      v45 = src;
      v27 = (_OWORD *)v26->m128i_i64[0];
      v28 = v26 + 1;
      if ( (__m128i *)v26->m128i_i64[0] != &v26[1] )
        goto LABEL_43;
    }
    else
    {
LABEL_42:
      v26 = (__m128i *)sub_2241490(&v39, v42, v43, v25, v40);
      v45 = src;
      v27 = (_OWORD *)v26->m128i_i64[0];
      v28 = v26 + 1;
      if ( (__m128i *)v26->m128i_i64[0] != &v26[1] )
      {
LABEL_43:
        v45 = v27;
        *(_QWORD *)&src[0] = v26[1].m128i_i64[0];
        goto LABEL_44;
      }
    }
    src[0] = _mm_loadu_si128(v26 + 1);
LABEL_44:
    n = v26->m128i_u64[1];
    v26->m128i_i64[0] = (__int64)v28;
    v26->m128i_i64[1] = 0;
    v26[1].m128i_i8[0] = 0;
    v29 = dest;
    v30 = n;
    if ( v45 == src )
    {
      if ( n )
      {
        if ( n == 1 )
          *(_BYTE *)dest = src[0];
        else
          memcpy(dest, src, n);
        v30 = n;
        v29 = dest;
      }
      v37 = v30;
      v29[v30] = 0;
      v29 = v45;
      goto LABEL_48;
    }
    if ( dest == v38 )
    {
      dest = v45;
      v37 = n;
      v38[0] = *(_QWORD *)&src[0];
    }
    else
    {
      v31 = v38[0];
      dest = v45;
      v37 = n;
      v38[0] = *(_QWORD *)&src[0];
      if ( v29 )
      {
        v45 = v29;
        *(_QWORD *)&src[0] = v31;
LABEL_48:
        n = 0;
        *v29 = 0;
        if ( v45 != src )
          j_j___libc_free_0(v45, *(_QWORD *)&src[0] + 1LL);
        if ( v39 != v41 )
          j_j___libc_free_0(v39, v41[0] + 1LL);
        if ( v42 != (const char *)v44 )
          j_j___libc_free_0(v42, v44[0] + 1LL);
LABEL_16:
        (*(void (__fastcall **)(_QWORD, _QWORD *, _QWORD))(**(_QWORD **)(a1 + 160) + 16LL))(
          *(_QWORD *)(a1 + 160),
          a2,
          v34);
        if ( *(_BYTE *)(a1 + 202) )
        {
          if ( a4 )
            sub_1F463C0(a1, (__int64)&dest, v13);
          if ( a3 )
            sub_1F46420(a1, (__int64)&dest);
        }
        goto LABEL_21;
      }
    }
    v45 = src;
    v29 = src;
    goto LABEL_48;
  }
  (*(void (__fastcall **)(_QWORD, _QWORD *, _QWORD))(**(_QWORD **)(a1 + 160) + 16LL))(*(_QWORD *)(a1 + 160), a2, a5);
LABEL_21:
  v14 = *(_QWORD *)(a1 + 216);
  v15 = *(_QWORD *)(v14 + 32);
  v16 = 32LL * *(unsigned int *)(v14 + 40);
  if ( v15 + v16 != v15 )
  {
    v17 = v34;
    v18 = v15 + v16;
    do
    {
      if ( *(_QWORD *)v15 == v9 )
      {
        v19 = *(_QWORD *)(v15 + 8);
        v20 = *(unsigned __int8 *)(v15 + 25);
        v21 = *(unsigned __int8 *)(v15 + 24);
        if ( !*(_BYTE *)(v15 + 16) )
        {
          v33 = *(unsigned __int8 *)(v15 + 24);
          v35 = *(unsigned __int8 *)(v15 + 25);
          v32 = sub_16369A0(*(_QWORD *)(v15 + 8), v19);
          v21 = v33;
          v20 = v35;
          v19 = v32;
        }
        sub_1F46490(a1, v19, v21, v20, v17);
      }
      v15 += 32;
    }
    while ( v18 != v15 );
  }
  if ( dest != v38 )
    j_j___libc_free_0(dest, v38[0] + 1LL);
LABEL_5:
  result = *(_QWORD *)(a1 + 176);
  if ( *(_QWORD *)(a1 + 192) != v9 )
  {
    if ( v9 != result )
    {
      if ( !*(_BYTE *)(a1 + 201) )
        return result;
      goto LABEL_12;
    }
LABEL_27:
    *(_BYTE *)(a1 + 200) = 1;
    return result;
  }
  *(_BYTE *)(a1 + 201) = 1;
  if ( v9 == result )
    goto LABEL_27;
LABEL_12:
  if ( !*(_BYTE *)(a1 + 200) )
    sub_16BD130("Cannot stop compilation after pass that is not run", 1u);
  return result;
}
