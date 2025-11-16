// Function: sub_2E904C0
// Address: 0x2e904c0
//
__int64 __fastcall sub_2E904C0(__int64 a1, __int64 *a2, __int64 a3, __int32 a4, __int64 a5)
{
  __int64 v7; // rax
  unsigned __int8 *v8; // rsi
  __int64 v9; // r12
  _QWORD *v10; // r13
  __int64 v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  const __m128i *v16; // r15
  const __m128i *v17; // rbx
  const __m128i *v18; // r15
  const __m128i **v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // rsi
  __int64 v22; // rcx
  __int64 v23; // rdx
  const __m128i **v24; // rdx
  __int64 v25; // [rsp+10h] [rbp-90h]
  unsigned __int8 *v28; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int8 *v29; // [rsp+38h] [rbp-68h] BYREF
  __m128i v30; // [rsp+40h] [rbp-60h] BYREF
  __m128i v31; // [rsp+50h] [rbp-50h]
  __int64 v32; // [rsp+60h] [rbp-40h]

  v7 = sub_2E893B0(a3, a5);
  v8 = *(unsigned __int8 **)(a3 + 56);
  v9 = *(_QWORD *)(a3 + 16);
  v25 = v7;
  v28 = v8;
  if ( !v8 )
  {
    v30.m128i_i64[0] = 0;
    goto LABEL_20;
  }
  sub_B96E90((__int64)&v28, (__int64)v8, 1);
  v30.m128i_i64[0] = (__int64)v28;
  if ( !v28 )
  {
LABEL_20:
    v30.m128i_i64[1] = 0;
    v31.m128i_i64[0] = 0;
    v29 = 0;
    v10 = *(_QWORD **)(a1 + 32);
    goto LABEL_5;
  }
  sub_B976B0((__int64)&v28, v28, (__int64)&v30);
  v28 = 0;
  v30.m128i_i64[1] = 0;
  v31.m128i_i64[0] = 0;
  v10 = *(_QWORD **)(a1 + 32);
  v29 = (unsigned __int8 *)v30.m128i_i64[0];
  if ( v30.m128i_i64[0] )
    sub_B96E90((__int64)&v29, v30.m128i_i64[0], 1);
LABEL_5:
  v11 = (__int64)sub_2E7B380(v10, v9, &v29, 0);
  if ( v29 )
    sub_B91220((__int64)&v29, (__int64)v29);
  sub_2E31040((__int64 *)(a1 + 40), v11);
  v12 = *a2;
  v13 = *(_QWORD *)v11;
  *(_QWORD *)(v11 + 8) = a2;
  v12 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v11 = v12 | v13 & 7;
  *(_QWORD *)(v12 + 8) = v11;
  v14 = v30.m128i_i64[1];
  *a2 = v11 | *a2 & 7;
  if ( v14 )
    sub_2E882B0(v11, (__int64)v10, v14);
  if ( v31.m128i_i64[0] )
    sub_2E88680(v11, (__int64)v10, v31.m128i_i64[0]);
  if ( v30.m128i_i64[0] )
    sub_B91220((__int64)&v30, v30.m128i_i64[0]);
  if ( v28 )
    sub_B91220((__int64)&v28, (__int64)v28);
  if ( *(_WORD *)(a3 + 68) == 14 )
  {
    v30.m128i_i64[0] = 5;
    v31.m128i_i64[0] = 0;
    v31.m128i_i32[2] = a4;
    sub_2E8EAD0(v11, (__int64)v10, &v30);
    v30.m128i_i64[0] = 1;
    v31 = 0u;
    sub_2E8EAD0(v11, (__int64)v10, &v30);
  }
  v31.m128i_i64[1] = sub_2E89170(a3);
  v30.m128i_i64[0] = 14;
  v31.m128i_i64[0] = 0;
  sub_2E8EAD0(v11, (__int64)v10, &v30);
  v30.m128i_i64[0] = 14;
  v31.m128i_i64[0] = 0;
  v31.m128i_i64[1] = v25;
  sub_2E8EAD0(v11, (__int64)v10, &v30);
  if ( *(_WORD *)(a3 + 68) == 15 )
  {
    v16 = *(const __m128i **)(a3 + 32);
    v17 = (const __m128i *)((char *)v16 + 40 * (*(_DWORD *)(a3 + 40) & 0xFFFFFF));
    v18 = v16 + 5;
    if ( v17 != v18 )
    {
      while ( 1 )
      {
        v19 = *(const __m128i ***)a5;
        v20 = 8LL * *(unsigned int *)(a5 + 8);
        v21 = (_QWORD *)(*(_QWORD *)a5 + v20);
        v22 = v20 >> 3;
        v23 = v20 >> 5;
        if ( v23 )
        {
          v24 = &v19[4 * v23];
          while ( v18 != *v19 )
          {
            if ( v18 == v19[1] )
            {
              ++v19;
              goto LABEL_29;
            }
            if ( v18 == v19[2] )
            {
              v19 += 2;
              goto LABEL_29;
            }
            if ( v18 == v19[3] )
            {
              v19 += 3;
              goto LABEL_29;
            }
            v19 += 4;
            if ( v24 == v19 )
            {
              v22 = (const __m128i **)v21 - v19;
              goto LABEL_34;
            }
          }
          goto LABEL_29;
        }
LABEL_34:
        if ( v22 == 2 )
          goto LABEL_40;
        if ( v22 == 3 )
          break;
        if ( v22 != 1 )
          goto LABEL_37;
LABEL_42:
        if ( *v19 != v18 )
        {
LABEL_37:
          v30 = _mm_loadu_si128(v18);
          v31 = _mm_loadu_si128(v18 + 1);
          v32 = v18[2].m128i_i64[0];
          sub_2E8EAD0(v11, (__int64)v10, &v30);
          goto LABEL_31;
        }
LABEL_29:
        if ( v21 == v19 )
          goto LABEL_37;
        v30.m128i_i64[0] = 5;
        v31.m128i_i64[0] = 0;
        v31.m128i_i32[2] = a4;
        sub_2E8EAD0(v11, (__int64)v10, &v30);
LABEL_31:
        v18 = (const __m128i *)((char *)v18 + 40);
        if ( v17 == v18 )
          return v11;
      }
      if ( *v19 == v18 )
        goto LABEL_29;
      ++v19;
LABEL_40:
      if ( *v19 == v18 )
        goto LABEL_29;
      ++v19;
      goto LABEL_42;
    }
  }
  return v11;
}
