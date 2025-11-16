// Function: sub_108C9C0
// Address: 0x108c9c0
//
__int64 __fastcall sub_108C9C0(__int64 a1, _BYTE *a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 *v10; // r15
  __m128i *v11; // rsi
  __m128i *v12; // rsi
  _QWORD *v13; // r14
  _QWORD *v14; // rdi
  __m128i *v15; // rdi
  __int64 result; // rax
  __m128i *v17; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v18; // [rsp+18h] [rbp-A8h]
  __m128i v19; // [rsp+20h] [rbp-A0h] BYREF
  __m128i *v20; // [rsp+30h] [rbp-90h] BYREF
  __int64 v21; // [rsp+38h] [rbp-88h]
  __m128i v22; // [rsp+40h] [rbp-80h] BYREF
  __m128i *v23; // [rsp+50h] [rbp-70h]
  __int64 v24; // [rsp+58h] [rbp-68h]
  __m128i v25; // [rsp+60h] [rbp-60h] BYREF
  __m128i *v26; // [rsp+70h] [rbp-50h]
  __int64 v27; // [rsp+78h] [rbp-48h]
  _OWORD v28[4]; // [rsp+80h] [rbp-40h] BYREF

  if ( a4 )
  {
    v20 = &v22;
    sub_108B380((__int64 *)&v20, a4, (__int64)&a4[a5]);
  }
  else
  {
    v22.m128i_i8[0] = 0;
    v20 = &v22;
    v21 = 0;
  }
  if ( a2 )
  {
    v17 = &v19;
    sub_108B380((__int64 *)&v17, a2, (__int64)&a2[a3]);
    v26 = (__m128i *)v28;
    v7 = v18;
    if ( v17 != &v19 )
    {
      v26 = v17;
      *(_QWORD *)&v28[0] = v19.m128i_i64[0];
      goto LABEL_6;
    }
  }
  else
  {
    v19.m128i_i8[0] = 0;
    v7 = 0;
    v26 = (__m128i *)v28;
  }
  v28[0] = _mm_load_si128(&v19);
LABEL_6:
  v27 = v7;
  v23 = &v25;
  v17 = &v19;
  v18 = 0;
  v19.m128i_i8[0] = 0;
  if ( v20 == &v22 )
  {
    v25 = _mm_load_si128(&v22);
  }
  else
  {
    v23 = v20;
    v25.m128i_i64[0] = v22.m128i_i64[0];
  }
  v8 = v21;
  v20 = &v22;
  v21 = 0;
  v24 = v8;
  v22.m128i_i8[0] = 0;
  v9 = (__int64 *)sub_22077B0(72);
  v10 = v9;
  if ( v9 )
  {
    v11 = v26;
    *v9 = (__int64)(v9 + 2);
    sub_108B590(v9, v11, (__int64)v11->m128i_i64 + v27);
    v12 = v23;
    v10[4] = (__int64)(v10 + 6);
    sub_108B590(v10 + 4, v12, (__int64)v12->m128i_i64 + v24);
  }
  if ( v23 != &v25 )
    j_j___libc_free_0(v23, v25.m128i_i64[0] + 1);
  if ( v26 != (__m128i *)v28 )
    j_j___libc_free_0(v26, *(_QWORD *)&v28[0] + 1LL);
  v13 = *(_QWORD **)(a1 + 2024);
  *(_QWORD *)(a1 + 2024) = v10;
  if ( v13 )
  {
    v14 = (_QWORD *)v13[4];
    if ( v14 != v13 + 6 )
      j_j___libc_free_0(v14, v13[6] + 1LL);
    if ( (_QWORD *)*v13 != v13 + 2 )
      j_j___libc_free_0(*v13, v13[2] + 1LL);
    j_j___libc_free_0(v13, 72);
    v10 = *(__int64 **)(a1 + 2024);
  }
  v10[8] = 4;
  v15 = v17;
  result = 4
         * ((*(_QWORD *)(*(_QWORD *)(a1 + 2024) + 40LL) != 0)
          + (unsigned int)((*(_QWORD *)(*(_QWORD *)(a1 + 2024) + 40LL)
                          - (unsigned __int64)(*(_QWORD *)(*(_QWORD *)(a1 + 2024) + 40LL) != 0)) >> 2))
         + 4;
  *(_QWORD *)(a1 + 1984) += result;
  if ( v15 != &v19 )
    result = j_j___libc_free_0(v15, v19.m128i_i64[0] + 1);
  if ( v20 != &v22 )
    return j_j___libc_free_0(v20, v22.m128i_i64[0] + 1);
  return result;
}
