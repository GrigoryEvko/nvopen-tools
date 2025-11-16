// Function: sub_31BBC70
// Address: 0x31bbc70
//
__int64 __fastcall sub_31BBC70(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v4; // r12
  __int64 v5; // r15
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 *v9; // r12
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 *v12; // rdx
  __int64 *v13; // r12
  __int64 *v14; // r14
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r15
  __int64 v19; // r14
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 j; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 *v27; // r15
  __int64 v28; // r14
  __int64 i; // r12
  __int64 v30; // rax
  __int64 v31; // [rsp-C0h] [rbp-C0h]
  __int64 v32; // [rsp-B8h] [rbp-B8h]
  __int64 *v33; // [rsp-B0h] [rbp-B0h]
  __int64 *v34; // [rsp-A8h] [rbp-A8h]
  __int64 *v35; // [rsp-A0h] [rbp-A0h]
  __int64 *v36; // [rsp-A0h] [rbp-A0h]
  __int64 v37; // [rsp-A0h] [rbp-A0h]
  __m128i v38; // [rsp-98h] [rbp-98h] BYREF
  __m128i v39; // [rsp-88h] [rbp-88h] BYREF
  __m128i v40; // [rsp-78h] [rbp-78h] BYREF
  __int64 *v41; // [rsp-68h] [rbp-68h] BYREF
  __int64 v42; // [rsp-60h] [rbp-60h]
  __int64 v43; // [rsp-58h] [rbp-58h] BYREF

  if ( !a3 )
    return 0;
  v4 = a2 + 1;
  v35 = &a2[a3];
  v5 = *a2;
  if ( v35 == a2 + 1 )
  {
    v8 = *(_QWORD *)(a1 + 32);
    v9 = (__int64 *)(a1 + 32);
    if ( !v8 )
    {
      v6 = *a2;
      goto LABEL_11;
    }
    if ( !v5 )
    {
      v6 = *(_QWORD *)(a1 + 40);
      v5 = *(_QWORD *)(a1 + 32);
      goto LABEL_11;
    }
    v6 = *a2;
    goto LABEL_25;
  }
  v6 = *a2;
  do
  {
    while ( 1 )
    {
      v7 = *v4;
      if ( !sub_B445A0(*(_QWORD *)(*v4 + 16), *(_QWORD *)(v5 + 16)) )
        break;
      v5 = v7;
      if ( v35 == ++v4 )
        goto LABEL_10;
    }
    if ( sub_B445A0(*(_QWORD *)(v6 + 16), *(_QWORD *)(v7 + 16)) )
      v6 = v7;
    ++v4;
  }
  while ( v35 != v4 );
LABEL_10:
  v8 = *(_QWORD *)(a1 + 32);
  v9 = (__int64 *)(a1 + 32);
  if ( v8 )
  {
LABEL_25:
    if ( sub_B445A0(*(_QWORD *)(v8 + 16), *(_QWORD *)(v5 + 16)) )
      v5 = *(_QWORD *)(a1 + 32);
    if ( !sub_B445A0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 16LL), *(_QWORD *)(v6 + 16)) )
      v6 = *(_QWORD *)(a1 + 40);
  }
LABEL_11:
  v38.m128i_i64[1] = v6;
  v38.m128i_i64[0] = v5;
  sub_31BA920((__int64)&v41, &v38, v9);
  v10 = *v41;
  v11 = v41[1];
  if ( v41 != &v43 )
    _libc_free((unsigned __int64)v41);
  v39.m128i_i64[0] = v10;
  v39.m128i_i64[1] = v11;
  if ( v10 )
  {
    sub_31BB1F0(a1, v39.m128i_i64);
    v31 = sub_31B9080(v9, a1);
    v13 = (__int64 *)v31;
    v34 = v12;
    if ( !v31 )
    {
      v40 = _mm_loadu_si128(&v39);
      v25 = sub_31B9080(v40.m128i_i64, a1);
      v27 = (__int64 *)v25;
      if ( v25 )
      {
        v28 = v26;
        if ( v26 )
          v28 = *(_QWORD *)(v26 + 48);
        for ( i = *(_QWORD *)(v25 + 48); i != v28; i = *(_QWORD *)(i + 48) )
        {
          v30 = *(_QWORD *)(i + 40);
          v41 = v27;
          v42 = v30;
          sub_31BB4E0(a1, i, (__int64 *)&v41);
        }
      }
      goto LABEL_22;
    }
    v14 = v12;
    if ( !sub_B445A0(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 16LL), *(_QWORD *)(v39.m128i_i64[0] + 16)) )
    {
      if ( !sub_B445A0(*(_QWORD *)(v39.m128i_i64[1] + 16), *(_QWORD *)(*(_QWORD *)(a1 + 32) + 16LL)) )
        BUG();
      v40 = _mm_loadu_si128(&v39);
      v33 = (__int64 *)sub_31B9080(v40.m128i_i64, a1);
      if ( v33 )
      {
        v32 = v22;
        if ( v22 )
          v32 = *(_QWORD *)(v22 + 48);
        for ( j = v33[6]; j != v32; j = *(_QWORD *)(v37 + 48) )
        {
          v37 = j;
          v42 = *(_QWORD *)(j + 40);
          v41 = v33;
          sub_31BB4E0(a1, j, (__int64 *)&v41);
        }
      }
      v41 = (__int64 *)sub_31B9080(v39.m128i_i64, a1);
      v42 = v24;
      if ( !v34 || (v14 = (__int64 *)v34[6], v14 != (__int64 *)v31) )
      {
        do
        {
          sub_31BB4E0(a1, (__int64)v13, (__int64 *)&v41);
          v13 = (__int64 *)v13[6];
        }
        while ( v13 != v14 );
      }
      goto LABEL_22;
    }
    v15 = sub_31B9080(v39.m128i_i64, a1);
    v17 = v15;
    v18 = v16;
    v19 = v15;
    if ( v15 )
    {
      v36 = (__int64 *)v15;
      if ( !sub_B445A0(*(_QWORD *)(*(_QWORD *)(v31 + 8) + 16LL), *(_QWORD *)(*(_QWORD *)(v15 + 8) + 16LL)) )
        v13 = v36;
      sub_B445A0(*(_QWORD *)(v34[1] + 16), *(_QWORD *)(*(_QWORD *)(v18 + 8) + 16LL));
      v17 = (__int64)v36;
    }
    else if ( !v16 )
    {
LABEL_22:
      result = v39.m128i_i64[0];
      *(__m128i *)(a1 + 32) = _mm_loadu_si128(&v38);
      return result;
    }
    v20 = *(_QWORD *)(v18 + 48);
    if ( v20 != v17 )
    {
      do
      {
        v21 = *(_QWORD *)(v19 + 40);
        v41 = v13;
        v42 = v21;
        sub_31BB4E0(a1, v19, (__int64 *)&v41);
        v19 = *(_QWORD *)(v19 + 48);
      }
      while ( v20 != v19 );
    }
    goto LABEL_22;
  }
  return 0;
}
