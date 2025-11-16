// Function: sub_2E901B0
// Address: 0x2e901b0
//
__int64 __fastcall sub_2E901B0(__int64 a1, __int64 *a2, __int64 a3, __int32 a4, int a5)
{
  __int64 v7; // rax
  unsigned __int8 *v8; // rsi
  __int64 v9; // r15
  _QWORD *v10; // r12
  __int64 v11; // r15
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  const __m128i *v16; // r14
  const __m128i *v17; // rbx
  const __m128i *i; // r14
  __int64 v19; // [rsp+8h] [rbp-88h]
  unsigned __int8 *v22; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int8 *v23; // [rsp+28h] [rbp-68h] BYREF
  __m128i v24; // [rsp+30h] [rbp-60h] BYREF
  __m128i v25; // [rsp+40h] [rbp-50h]
  __int64 v26; // [rsp+50h] [rbp-40h]

  v7 = sub_2E894A0(a3, a5);
  v8 = *(unsigned __int8 **)(a3 + 56);
  v9 = *(_QWORD *)(a3 + 16);
  v19 = v7;
  v22 = v8;
  if ( !v8 )
  {
    v24.m128i_i64[0] = 0;
    goto LABEL_20;
  }
  sub_B96E90((__int64)&v22, (__int64)v8, 1);
  v24.m128i_i64[0] = (__int64)v22;
  if ( !v22 )
  {
LABEL_20:
    v24.m128i_i64[1] = 0;
    v25.m128i_i64[0] = 0;
    v23 = 0;
    v10 = *(_QWORD **)(a1 + 32);
    goto LABEL_5;
  }
  sub_B976B0((__int64)&v22, v22, (__int64)&v24);
  v22 = 0;
  v24.m128i_i64[1] = 0;
  v25.m128i_i64[0] = 0;
  v10 = *(_QWORD **)(a1 + 32);
  v23 = (unsigned __int8 *)v24.m128i_i64[0];
  if ( v24.m128i_i64[0] )
    sub_B96E90((__int64)&v23, v24.m128i_i64[0], 1);
LABEL_5:
  v11 = (__int64)sub_2E7B380(v10, v9, &v23, 0);
  if ( v23 )
    sub_B91220((__int64)&v23, (__int64)v23);
  sub_2E31040((__int64 *)(a1 + 40), v11);
  v12 = *a2;
  v13 = *(_QWORD *)v11;
  *(_QWORD *)(v11 + 8) = a2;
  v12 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v11 = v12 | v13 & 7;
  *(_QWORD *)(v12 + 8) = v11;
  v14 = v24.m128i_i64[1];
  *a2 = v11 | *a2 & 7;
  if ( v14 )
    sub_2E882B0(v11, (__int64)v10, v14);
  if ( v25.m128i_i64[0] )
    sub_2E88680(v11, (__int64)v10, v25.m128i_i64[0]);
  if ( v24.m128i_i64[0] )
    sub_B91220((__int64)&v24, v24.m128i_i64[0]);
  if ( v22 )
    sub_B91220((__int64)&v22, (__int64)v22);
  if ( *(_WORD *)(a3 + 68) == 14 )
  {
    v24.m128i_i64[0] = 5;
    v25.m128i_i64[0] = 0;
    v25.m128i_i32[2] = a4;
    sub_2E8EAD0(v11, (__int64)v10, &v24);
    v24.m128i_i64[0] = 1;
    v25 = 0u;
    sub_2E8EAD0(v11, (__int64)v10, &v24);
  }
  v25.m128i_i64[1] = sub_2E89170(a3);
  v24.m128i_i64[0] = 14;
  v25.m128i_i64[0] = 0;
  sub_2E8EAD0(v11, (__int64)v10, &v24);
  v24.m128i_i64[0] = 14;
  v25.m128i_i64[0] = 0;
  v25.m128i_i64[1] = v19;
  sub_2E8EAD0(v11, (__int64)v10, &v24);
  if ( *(_WORD *)(a3 + 68) == 15 )
  {
    v16 = *(const __m128i **)(a3 + 32);
    v17 = (const __m128i *)((char *)v16 + 40 * (*(_DWORD *)(a3 + 40) & 0xFFFFFF));
    for ( i = v16 + 5; v17 != i; i = (const __m128i *)((char *)i + 40) )
    {
      if ( i->m128i_i8[0] || a5 != i->m128i_i32[2] )
      {
        v24 = _mm_loadu_si128(i);
        v25 = _mm_loadu_si128(i + 1);
        v26 = i[2].m128i_i64[0];
        sub_2E8EAD0(v11, (__int64)v10, &v24);
      }
      else
      {
        v24.m128i_i64[0] = 5;
        v25.m128i_i64[0] = 0;
        v25.m128i_i32[2] = a4;
        sub_2E8EAD0(v11, (__int64)v10, &v24);
      }
    }
  }
  return v11;
}
