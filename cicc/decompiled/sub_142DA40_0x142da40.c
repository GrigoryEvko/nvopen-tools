// Function: sub_142DA40
// Address: 0x142da40
//
_QWORD *__fastcall sub_142DA40(_QWORD *a1, unsigned __int64 *a2, const __m128i *a3)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // rbx
  _QWORD *v8; // r14
  _QWORD *v9; // r8
  __int64 v10; // rsi
  __int64 v11; // r13
  _QWORD *v12; // r9
  __m128i v13; // xmm0
  _QWORD *v14; // r12
  unsigned __int64 v15; // rdx
  _QWORD *v16; // rax
  __int64 v18; // rax
  _BOOL8 v19; // rdi
  _QWORD *v20; // [rsp+18h] [rbp-38h]
  _QWORD *v21; // [rsp+18h] [rbp-38h]

  v5 = sub_22077B0(80);
  v7 = *a2;
  v8 = (_QWORD *)a3[1].m128i_i64[0];
  a3[1].m128i_i64[0] = 0;
  v9 = (_QWORD *)a3[1].m128i_i64[1];
  v10 = a3[2].m128i_i64[0];
  v11 = v5;
  v12 = a1 + 1;
  v13 = _mm_loadu_si128(a3);
  *(_QWORD *)(v5 + 32) = v7;
  a3[2].m128i_i64[0] = 0;
  a3[1].m128i_i64[1] = 0;
  v14 = (_QWORD *)a1[2];
  *(_QWORD *)(v5 + 56) = v8;
  *(_QWORD *)(v5 + 64) = v9;
  *(_QWORD *)(v5 + 72) = v10;
  *(__m128i *)(v5 + 40) = v13;
  if ( v14 )
  {
    while ( 1 )
    {
      v15 = v14[4];
      v16 = (_QWORD *)v14[3];
      if ( v15 > v7 )
        v16 = (_QWORD *)v14[2];
      LOBYTE(v6) = v7 < v15;
      if ( !v16 )
        break;
      v14 = v16;
    }
    if ( v7 >= v15 )
    {
      if ( v15 < v7 )
        goto LABEL_20;
      goto LABEL_9;
    }
    if ( v14 == (_QWORD *)a1[3] )
      goto LABEL_20;
  }
  else
  {
    v14 = a1 + 1;
    if ( v12 == (_QWORD *)a1[3] )
    {
      v19 = 1;
LABEL_22:
      sub_220F040(v19, v11, v14, v12);
      ++a1[5];
      return (_QWORD *)v11;
    }
  }
  v21 = v9;
  v18 = sub_220EF80(v14);
  v9 = v21;
  if ( *(_QWORD *)(v18 + 32) >= v7 )
  {
    v14 = (_QWORD *)v18;
  }
  else
  {
    v12 = a1 + 1;
    if ( v14 )
    {
LABEL_20:
      v19 = 1;
      if ( v12 != v14 )
        v19 = v7 < v14[4];
      goto LABEL_22;
    }
  }
LABEL_9:
  if ( v8 != v9 )
  {
    do
    {
      if ( *v8 )
      {
        v20 = v9;
        (*(void (__fastcall **)(_QWORD, __int64, unsigned __int64, __int64, _QWORD *, _QWORD *))(*(_QWORD *)*v8 + 8LL))(
          *v8,
          v10,
          v15,
          v6,
          v9,
          v12);
        v9 = v20;
      }
      ++v8;
    }
    while ( v9 != v8 );
    v10 = *(_QWORD *)(v11 + 72);
    v9 = *(_QWORD **)(v11 + 56);
  }
  if ( v9 )
    j_j___libc_free_0(v9, v10 - (_QWORD)v9);
  j_j___libc_free_0(v11, 80);
  return v14;
}
