// Function: sub_3744400
// Address: 0x3744400
//
__int64 __fastcall sub_3744400(
        _QWORD *a1,
        __int64 a2,
        unsigned int a3,
        unsigned int a4,
        __int64 a5,
        char a6,
        __int64 a7)
{
  unsigned int v7; // r15d
  unsigned int v9; // r13d
  __m128i *v10; // rsi
  int v11; // edx
  _QWORD **v12; // rdx
  unsigned __int16 v13; // ax
  unsigned __int64 v14; // rdi
  const __m128i *v15; // rax
  __m128i *v16; // rax
  const __m128i *v17; // rax
  unsigned int v18; // r12d
  unsigned int v23; // [rsp+1Ch] [rbp-84h]
  const __m128i *v24; // [rsp+20h] [rbp-80h] BYREF
  __m128i *v25; // [rsp+28h] [rbp-78h]
  const __m128i *v26; // [rsp+30h] [rbp-70h]
  __m128i v27; // [rsp+40h] [rbp-60h] BYREF
  __m128i v28; // [rsp+50h] [rbp-50h] BYREF
  __m128i v29; // [rsp+60h] [rbp-40h] BYREF

  v7 = a3;
  v9 = a4 + a3;
  v23 = a4;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  sub_3375F60(&v24, a4);
  for ( ; v9 != v7; v25 = v10 + 3 )
  {
    while ( 1 )
    {
      v29.m128i_i32[0] = v29.m128i_i16[0] & 0x8000;
      v11 = *(_DWORD *)(a2 + 4);
      v28.m128i_i64[1] = 0;
      v27.m128i_i64[1] = 0;
      v28.m128i_i32[0] = 0;
      v29.m128i_i64[1] = 0;
      v27.m128i_i64[0] = *(_QWORD *)(a2 + 32 * (v7 - (unsigned __int64)(v11 & 0x7FFFFFF)));
      v28.m128i_i64[1] = *(_QWORD *)(v27.m128i_i64[0] + 8);
      sub_34470B0((__int64)&v27, a2, v7);
      v10 = v25;
      if ( v25 != v26 )
        break;
      ++v7;
      sub_332CDC0((unsigned __int64 *)&v24, v25, &v27);
      if ( v9 == v7 )
        goto LABEL_8;
    }
    if ( v25 )
    {
      *v25 = _mm_loadu_si128(&v27);
      v10[1] = _mm_loadu_si128(&v28);
      v10[2] = _mm_loadu_si128(&v29);
      v10 = v25;
    }
    ++v7;
  }
LABEL_8:
  v12 = *(_QWORD ***)(a2 + 8);
  if ( a6 )
    v12 = (_QWORD **)sub_BCB120(*v12);
  v13 = *(_WORD *)(a2 + 2);
  v14 = *(_QWORD *)(a7 + 40);
  *(_QWORD *)a7 = v12;
  *(_QWORD *)(a7 + 24) = a5;
  *(_DWORD *)(a7 + 16) = (v13 >> 2) & 0x3FF;
  v15 = v24;
  v24 = 0;
  *(_QWORD *)(a7 + 40) = v15;
  v16 = v25;
  v25 = 0;
  *(_QWORD *)(a7 + 48) = v16;
  v17 = v26;
  v26 = 0;
  *(_QWORD *)(a7 + 56) = v17;
  if ( v14 )
    j_j___libc_free_0(v14);
  if ( v23 == -1 )
    v23 = -1431655765 * ((__int64)(*(_QWORD *)(a7 + 48) - *(_QWORD *)(a7 + 40)) >> 4);
  *(_DWORD *)(a7 + 12) = v23;
  v18 = sub_3743390(a1, (__int64 ***)a7);
  if ( v24 )
    j_j___libc_free_0((unsigned __int64)v24);
  return v18;
}
