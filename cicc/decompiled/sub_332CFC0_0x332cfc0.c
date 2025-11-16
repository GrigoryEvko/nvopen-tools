// Function: sub_332CFC0
// Address: 0x332cfc0
//
__m128i *__fastcall sub_332CFC0(
        __m128i *a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 (__fastcall *a5)(__int64 a1, __int64 a2, unsigned int a3),
        __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rcx
  unsigned int *v8; // rbx
  char v9; // r12
  unsigned int *v10; // r13
  __m128i *v12; // rsi
  unsigned __int16 *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rsi
  char v19; // al
  unsigned int v24; // [rsp+1Ch] [rbp-94h]
  __int16 v25; // [rsp+20h] [rbp-90h] BYREF
  __int64 v26; // [rsp+28h] [rbp-88h]
  unsigned __int64 v27; // [rsp+30h] [rbp-80h] BYREF
  __m128i *v28; // [rsp+38h] [rbp-78h]
  const __m128i *v29; // [rsp+40h] [rbp-70h]
  __m128i v30; // [rsp+50h] [rbp-60h] BYREF
  __m128i v31; // [rsp+60h] [rbp-50h] BYREF
  __m128i v32[4]; // [rsp+70h] [rbp-40h] BYREF

  v6 = *(unsigned int *)(a4 + 64);
  v7 = *(_QWORD *)(a4 + 40);
  v8 = (unsigned int *)(v7 + 40 * v6);
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0u;
  v31 = 0u;
  v32[0] = 0u;
  v24 = (unsigned __int8)a5;
  if ( (unsigned int *)v7 == v8 )
  {
    sub_33264E0(a1, a2, a3, a4, &v27, (unsigned __int8)a5);
  }
  else
  {
    v9 = (char)a5;
    v10 = (unsigned int *)v7;
    do
    {
      while ( 1 )
      {
        v13 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v10 + 48LL) + 16LL * v10[2]);
        v14 = *v13;
        v26 = *((_QWORD *)v13 + 1);
        v15 = *(_QWORD *)(a2 + 16);
        v25 = v14;
        v16 = sub_3007410((__int64)&v25, *(__int64 **)(v15 + 64), v14, v7, (__int64)a5, a6);
        v17 = *(_QWORD *)(a2 + 8);
        v18 = v16;
        v30.m128i_i64[1] = *(_QWORD *)v10;
        LODWORD(v16) = v10[2];
        v31.m128i_i64[1] = v18;
        v31.m128i_i32[0] = v16;
        a5 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v17 + 1128LL);
        v19 = v9;
        if ( a5 != sub_2FE3330 )
          v19 = a5(v17, v18, v24);
        v12 = v28;
        v32[0].m128i_i8[0] = v19 & 1 | v32[0].m128i_i8[0] & 0xFC | (2 * ((v19 ^ 1) & 1));
        if ( v28 != v29 )
          break;
        v10 += 10;
        sub_332CDC0(&v27, v28, &v30);
        if ( v8 == v10 )
          goto LABEL_10;
      }
      if ( v28 )
      {
        *v28 = _mm_loadu_si128(&v30);
        v12[1] = _mm_loadu_si128(&v31);
        v12[2] = _mm_loadu_si128(v32);
        v12 = v28;
      }
      v10 += 10;
      v28 = v12 + 3;
    }
    while ( v8 != v10 );
LABEL_10:
    sub_33264E0(a1, a2, a3, a4, &v27, v24);
  }
  if ( v27 )
    j_j___libc_free_0(v27);
  return a1;
}
