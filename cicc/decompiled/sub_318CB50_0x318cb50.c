// Function: sub_318CB50
// Address: 0x318cb50
//
void __fastcall sub_318CB50(_QWORD *a1, __int64 a2, __int64 a3)
{
  __m128i v6; // xmm1
  __int64 v7; // r14
  __int64 v8; // r14
  unsigned __int64 *v9; // r12
  __int64 v10; // rax
  __m128i *v11; // rdi
  _QWORD **v12; // r13
  _QWORD **v13; // rbx
  _QWORD *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rcx
  __m128i *v20; // rdx
  unsigned __int64 v21; // rsi
  __int64 v22; // r9
  int v23; // eax
  _QWORD *v24; // rcx
  __int64 v25; // rdi
  __int64 v26; // [rsp+0h] [rbp-80h]
  __int64 v27; // [rsp+0h] [rbp-80h]
  __int64 v28; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+8h] [rbp-78h]
  __int8 *v30; // [rsp+8h] [rbp-78h]
  __m128i v31; // [rsp+10h] [rbp-70h] BYREF
  __m128i v32; // [rsp+20h] [rbp-60h] BYREF
  __m128i v33; // [rsp+30h] [rbp-50h] BYREF
  __m128i v34; // [rsp+40h] [rbp-40h] BYREF

  sub_318B480((__int64)&v31, (__int64)a1);
  v6 = _mm_loadu_si128(&v32);
  v33 = _mm_loadu_si128(&v31);
  v34 = v6;
  sub_371B2F0(&v33);
  if ( v33.m128i_i64[1] == *(_QWORD *)(a3 + 8) )
    return;
  sub_3187170(a1[3], a1, a3);
  v7 = a1[3];
  if ( *(_DWORD *)(v7 + 72) == 1 )
  {
    v15 = sub_22077B0(0x18u);
    v18 = v15;
    if ( v15 )
    {
      v28 = v15;
      sub_318DEC0(v15, a1, v16, v17, v15);
      v18 = v28;
    }
    v33.m128i_i64[0] = v18;
    v19 = *(unsigned int *)(v7 + 16);
    v20 = &v33;
    v21 = *(_QWORD *)(v7 + 8);
    v22 = v19 + 1;
    v23 = *(_DWORD *)(v7 + 16);
    if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(v7 + 20) )
    {
      v25 = v7 + 8;
      if ( v21 > (unsigned __int64)&v33 )
      {
        v26 = v18;
        sub_31878D0(v25, v19 + 1, (__int64)&v33, v19, v18, v22);
        v19 = *(unsigned int *)(v7 + 16);
        v21 = *(_QWORD *)(v7 + 8);
        v20 = &v33;
        v18 = v26;
        v23 = *(_DWORD *)(v7 + 16);
      }
      else
      {
        if ( (unsigned __int64)&v33 < v21 + 8 * v19 )
        {
          v27 = v18;
          v30 = &v33.m128i_i8[-v21];
          sub_31878D0(v25, v19 + 1, (__int64)&v33, v19, v18, v22);
          v21 = *(_QWORD *)(v7 + 8);
          v19 = *(unsigned int *)(v7 + 16);
          v18 = v27;
          v20 = (__m128i *)&v30[v21];
        }
        else
        {
          v29 = v18;
          sub_31878D0(v25, v19 + 1, (__int64)&v33, v19, v18, v22);
          v19 = *(unsigned int *)(v7 + 16);
          v21 = *(_QWORD *)(v7 + 8);
          v18 = v29;
          v20 = &v33;
        }
        v23 = *(_DWORD *)(v7 + 16);
      }
    }
    v24 = (_QWORD *)(v21 + 8 * v19);
    if ( v24 )
    {
      *v24 = v20->m128i_i64[0];
      v20->m128i_i64[0] = 0;
      v18 = v33.m128i_i64[0];
      v23 = *(_DWORD *)(v7 + 16);
    }
    *(_DWORD *)(v7 + 16) = v23 + 1;
    if ( v18 )
    {
      (*(void (__fastcall **)(__int64, unsigned __int64, __m128i *))(*(_QWORD *)v18 + 24LL))(v18, v21, v20);
      v8 = *(_QWORD *)(a2 + 16);
      v9 = (unsigned __int64 *)(v8 + 48);
      if ( *(_QWORD *)(a3 + 8) == v8 + 48 )
        goto LABEL_5;
      goto LABEL_4;
    }
  }
  v8 = *(_QWORD *)(a2 + 16);
  v9 = (unsigned __int64 *)(v8 + 48);
  if ( *(_QWORD *)(a3 + 8) != v8 + 48 )
  {
LABEL_4:
    v10 = sub_371B3B0(a3, *(_QWORD *)(a3 + 8), *(_QWORD *)(a3 + 16));
    v9 = (unsigned __int64 *)(sub_318B5C0(v10) + 24);
  }
LABEL_5:
  (*(void (__fastcall **)(__m128i *, _QWORD *))(*a1 + 80LL))(&v33, a1);
  v11 = (__m128i *)v33.m128i_i64[0];
  v12 = (_QWORD **)(v33.m128i_i64[0] + 8LL * v33.m128i_u32[2]);
  if ( v12 != (_QWORD **)v33.m128i_i64[0] )
  {
    v13 = (_QWORD **)v33.m128i_i64[0];
    do
    {
      v14 = *v13++;
      sub_B44550(v14, v8, v9, 0);
    }
    while ( v12 != v13 );
    v11 = (__m128i *)v33.m128i_i64[0];
  }
  if ( v11 != &v34 )
    _libc_free((unsigned __int64)v11);
}
