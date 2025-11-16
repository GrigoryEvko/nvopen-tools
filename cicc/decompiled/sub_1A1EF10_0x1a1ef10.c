// Function: sub_1A1EF10
// Address: 0x1a1ef10
//
__int64 __fastcall sub_1A1EF10(__int64 a1, __int64 a2, __int64 **a3, const __m128i *a4, unsigned int a5, _QWORD *a6)
{
  unsigned __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 *v14; // rcx
  __m128i v15; // xmm0
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rsi
  __int64 v20; // rdx
  unsigned __int8 *v21; // rsi
  const void *v22; // r15
  __int64 *v23; // r13
  __int64 result; // rax
  _QWORD *v25; // rax
  _QWORD *v26; // r11
  __int64 *v27; // [rsp+8h] [rbp-D8h]
  __int64 v28; // [rsp+10h] [rbp-D0h]
  __int64 *v29; // [rsp+10h] [rbp-D0h]
  __int64 v30; // [rsp+10h] [rbp-D0h]
  __int64 v31; // [rsp+10h] [rbp-D0h]
  __int64 v33; // [rsp+18h] [rbp-C8h]
  __m128i v35; // [rsp+30h] [rbp-B0h] BYREF
  char v36; // [rsp+40h] [rbp-A0h]
  char v37; // [rsp+41h] [rbp-9Fh]
  __m128i v38; // [rsp+50h] [rbp-90h] BYREF
  __int64 v39; // [rsp+60h] [rbp-80h]
  __m128i v40; // [rsp+70h] [rbp-70h] BYREF
  __int64 v41; // [rsp+80h] [rbp-60h]
  __m128i v42; // [rsp+90h] [rbp-50h] BYREF
  __int16 v43; // [rsp+A0h] [rbp-40h]

  if ( a5 )
  {
    v8 = (*a6 >> 3) % (unsigned __int64)a5;
    if ( v8 )
      *a6 += 8 * (a5 - (unsigned int)v8);
  }
  v9 = sub_157EB90(*(_QWORD *)(a1 + 8));
  v10 = sub_1632FA0(v9);
  *a6 += sub_127FA20(v10, a2);
  v40.m128i_i64[0] = (__int64)".gep";
  LOWORD(v41) = 259;
  sub_14EC200(&v42, a4, &v40);
  v28 = sub_1A1D720((__int64 *)a1, *(_BYTE **)(a1 + 184), *(__int64 ***)(a1 + 136), *(unsigned int *)(a1 + 144), &v42);
  v35.m128i_i64[0] = (__int64)".load";
  v37 = 1;
  v36 = 3;
  sub_14EC200(&v38, a4, &v35);
  v11 = sub_1648A60(64, 1u);
  v12 = (__int64)v11;
  if ( v11 )
    sub_15F9210((__int64)v11, *(_QWORD *)(*(_QWORD *)v28 + 24LL), v28, 0, 0, 0);
  v13 = *(_QWORD *)(a1 + 8);
  v14 = *(__int64 **)(a1 + 16);
  if ( (unsigned __int8)v39 > 1u )
  {
    v27 = *(__int64 **)(a1 + 16);
    v43 = 260;
    v42.m128i_i64[0] = a1 + 64;
    v31 = v13;
    sub_14EC200(&v40, &v42, &v38);
    v13 = v31;
    v14 = v27;
  }
  else
  {
    v15 = _mm_loadu_si128(&v38);
    v41 = v39;
    v40 = v15;
  }
  v29 = v14;
  if ( v13 )
  {
    sub_157E9D0(v13 + 40, v12);
    v16 = *v29;
    v17 = *(_QWORD *)(v12 + 24) & 7LL;
    *(_QWORD *)(v12 + 32) = v29;
    v16 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v12 + 24) = v16 | v17;
    *(_QWORD *)(v16 + 8) = v12 + 24;
    *v29 = *v29 & 7 | (v12 + 24);
  }
  sub_164B780(v12, v40.m128i_i64);
  v18 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v42.m128i_i64[0] = *(_QWORD *)a1;
    sub_1623A60((__int64)&v42, v18, 2);
    v19 = *(_QWORD *)(v12 + 48);
    v20 = v12 + 48;
    if ( v19 )
    {
      sub_161E7C0(v12 + 48, v19);
      v20 = v12 + 48;
    }
    v21 = (unsigned __int8 *)v42.m128i_i64[0];
    *(_QWORD *)(v12 + 48) = v42.m128i_i64[0];
    if ( v21 )
      sub_1623210((__int64)&v42, v21, v20);
  }
  sub_15F8F50(v12, a5);
  LOWORD(v39) = 259;
  v38.m128i_i64[0] = (__int64)".insert";
  sub_14EC200(&v40, a4, &v38);
  v22 = *(const void **)(a1 + 104);
  v23 = *a3;
  if ( *((_BYTE *)*a3 + 16) > 0x10u || *(_BYTE *)(v12 + 16) > 0x10u )
  {
    v30 = *(unsigned int *)(a1 + 112);
    v43 = 257;
    v25 = sub_1648A60(88, 2u);
    v26 = v25;
    if ( v25 )
    {
      v33 = (__int64)v25;
      sub_15F1EA0((__int64)v25, *v23, 63, (__int64)(v25 - 6), 2, 0);
      *(_QWORD *)(v33 + 56) = v33 + 72;
      *(_QWORD *)(v33 + 64) = 0x400000000LL;
      sub_15FAD90(v33, (__int64)v23, v12, v22, v30, (__int64)&v42);
      v26 = (_QWORD *)v33;
    }
    result = (__int64)sub_1A1C7B0((__int64 *)a1, v26, &v40);
  }
  else
  {
    result = sub_15A3A20(*a3, (__int64 *)v12, *(_DWORD **)(a1 + 104), *(unsigned int *)(a1 + 112), 0);
  }
  *a3 = (__int64 *)result;
  return result;
}
