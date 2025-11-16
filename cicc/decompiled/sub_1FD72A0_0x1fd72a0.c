// Function: sub_1FD72A0
// Address: 0x1fd72a0
//
__int64 __fastcall sub_1FD72A0(
        _QWORD *a1,
        __int64 a2,
        unsigned int a3,
        unsigned int a4,
        __int64 a5,
        char a6,
        __int64 a7)
{
  __m128i *v10; // rsi
  int v11; // ecx
  _QWORD **v12; // rdx
  unsigned int v13; // eax
  __int64 v14; // rdi
  __int64 v15; // rsi
  const __m128i *v16; // rax
  __m128i *v17; // rax
  const __m128i *v18; // rax
  unsigned int v19; // r12d
  unsigned int v24; // [rsp+28h] [rbp-98h]
  unsigned int v25; // [rsp+2Ch] [rbp-94h]
  __int64 i; // [rsp+38h] [rbp-88h] BYREF
  const __m128i *v27; // [rsp+40h] [rbp-80h] BYREF
  __m128i *v28; // [rsp+48h] [rbp-78h]
  const __m128i *v29; // [rsp+50h] [rbp-70h]
  __m128i v30; // [rsp+60h] [rbp-60h] BYREF
  __m128i v31; // [rsp+70h] [rbp-50h] BYREF
  __int64 v32; // [rsp+80h] [rbp-40h]

  v24 = a4;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  sub_1FD3FA0(&v27, a4);
  v25 = a4 + a3;
  for ( i = a2 | 4; v25 != a3; v28 = (__m128i *)((char *)v10 + 40) )
  {
    while ( 1 )
    {
      LODWORD(v32) = v32 & 0xFC00;
      v11 = *(_DWORD *)(a2 + 20);
      v31.m128i_i64[1] = 0;
      v30.m128i_i64[1] = 0;
      v31.m128i_i32[0] = 0;
      v30.m128i_i64[0] = *(_QWORD *)(a2 + 24 * (a3 - (unsigned __int64)(v11 & 0xFFFFFFF)));
      v31.m128i_i64[1] = *(_QWORD *)v30.m128i_i64[0];
      sub_20A1C00(&v30, &i, a3);
      v10 = v28;
      if ( v28 != v29 )
        break;
      ++a3;
      sub_1D27190(&v27, v28, &v30);
      if ( v25 == a3 )
        goto LABEL_8;
    }
    if ( v28 )
    {
      *v28 = _mm_loadu_si128(&v30);
      v10[1] = _mm_loadu_si128(&v31);
      v10[2].m128i_i64[0] = v32;
      v10 = v28;
    }
    ++a3;
  }
LABEL_8:
  v12 = *(_QWORD ***)a2;
  if ( a6 )
    v12 = (_QWORD **)sub_1643270(*v12);
  v13 = *(unsigned __int16 *)(a2 + 18);
  v14 = *(_QWORD *)(a7 + 40);
  *(_QWORD *)a7 = v12;
  v15 = *(_QWORD *)(a7 + 56);
  *(_QWORD *)(a7 + 24) = a5;
  *(_DWORD *)(a7 + 16) = (v13 >> 2) & 0x3FFFDFFF;
  v16 = v27;
  v27 = 0;
  *(_QWORD *)(a7 + 40) = v16;
  v17 = v28;
  v28 = 0;
  *(_QWORD *)(a7 + 48) = v17;
  v18 = v29;
  v29 = 0;
  *(_QWORD *)(a7 + 56) = v18;
  if ( v14 )
    j_j___libc_free_0(v14, v15 - v14);
  if ( v24 == -1 )
    v24 = -858993459 * ((__int64)(*(_QWORD *)(a7 + 48) - *(_QWORD *)(a7 + 40)) >> 3);
  *(_DWORD *)(a7 + 12) = v24;
  v19 = sub_1FD6490(a1, a7);
  if ( v27 )
    j_j___libc_free_0(v27, (char *)v29 - (char *)v27);
  return v19;
}
