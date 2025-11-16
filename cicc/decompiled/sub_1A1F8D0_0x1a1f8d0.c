// Function: sub_1A1F8D0
// Address: 0x1a1f8d0
//
__int64 __fastcall sub_1A1F8D0(__int64 a1, __int64 a2, __int64 *a3, const __m128i *a4, unsigned int a5, _QWORD *a6)
{
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // r9
  _QWORD *v14; // rax
  _QWORD *v15; // r13
  __int64 v16; // rax
  unsigned __int64 *v17; // rcx
  __m128i v18; // xmm0
  unsigned __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rsi
  unsigned __int8 *v23; // rsi
  _QWORD *v25; // rax
  _QWORD *v26; // r11
  unsigned int *v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  unsigned int *v32; // [rsp+8h] [rbp-B8h]
  __int64 v33; // [rsp+18h] [rbp-A8h]
  unsigned __int64 *v34; // [rsp+18h] [rbp-A8h]
  unsigned int *v35; // [rsp+18h] [rbp-A8h]
  __int64 v36; // [rsp+18h] [rbp-A8h]
  __int64 v37; // [rsp+20h] [rbp-A0h]
  __int64 v38; // [rsp+20h] [rbp-A0h]
  unsigned __int64 *v39; // [rsp+20h] [rbp-A0h]
  __int64 v40; // [rsp+20h] [rbp-A0h]
  __m128i v42; // [rsp+30h] [rbp-90h] BYREF
  __int64 v43; // [rsp+40h] [rbp-80h]
  __m128i v44; // [rsp+50h] [rbp-70h] BYREF
  __int64 v45; // [rsp+60h] [rbp-60h]
  __m128i v46; // [rsp+70h] [rbp-50h] BYREF
  __int16 v47; // [rsp+80h] [rbp-40h]

  if ( a5 )
  {
    v9 = (*a6 >> 3) % (unsigned __int64)a5;
    if ( v9 )
      *a6 += 8 * (a5 - (unsigned int)v9);
  }
  v10 = sub_157EB90(*(_QWORD *)(a1 + 8));
  v11 = sub_1632FA0(v10);
  *a6 += sub_127FA20(v11, a2);
  v42.m128i_i64[0] = (__int64)".extract";
  LOWORD(v43) = 259;
  sub_14EC200(&v44, a4, &v42);
  v12 = *a3;
  v37 = *(unsigned int *)(a1 + 112);
  if ( *(_BYTE *)(v12 + 16) > 0x10u )
  {
    v35 = *(unsigned int **)(a1 + 104);
    v47 = 257;
    v25 = sub_1648A60(88, 1u);
    v26 = v25;
    if ( v25 )
    {
      v27 = v35;
      v32 = v35;
      v36 = (__int64)v25;
      v28 = sub_15FB2A0(*(_QWORD *)v12, v27, v37);
      sub_15F1EA0(v36, v28, 62, v36 - 24, 1, 0);
      if ( *(_QWORD *)(v36 - 24) )
      {
        v29 = *(_QWORD *)(v36 - 16);
        v30 = *(_QWORD *)(v36 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v30 = v29;
        if ( v29 )
          *(_QWORD *)(v29 + 16) = *(_QWORD *)(v29 + 16) & 3LL | v30;
      }
      *(_QWORD *)(v36 - 24) = v12;
      v31 = *(_QWORD *)(v12 + 8);
      *(_QWORD *)(v36 - 16) = v31;
      if ( v31 )
        *(_QWORD *)(v31 + 16) = (v36 - 16) | *(_QWORD *)(v31 + 16) & 3LL;
      *(_QWORD *)(v36 - 8) = (v12 + 8) | *(_QWORD *)(v36 - 8) & 3LL;
      *(_QWORD *)(v12 + 8) = v36 - 24;
      *(_QWORD *)(v36 + 56) = v36 + 72;
      *(_QWORD *)(v36 + 64) = 0x400000000LL;
      sub_15FB110(v36, v32, v37, (__int64)&v46);
      v26 = (_QWORD *)v36;
    }
    v13 = (__int64)sub_1A1C7B0((__int64 *)a1, v26, &v44);
  }
  else
  {
    v13 = sub_15A3AE0((_QWORD *)v12, *(unsigned int **)(a1 + 104), *(unsigned int *)(a1 + 112), 0);
  }
  v33 = v13;
  v44.m128i_i64[0] = (__int64)".gep";
  LOWORD(v45) = 259;
  sub_14EC200(&v46, a4, &v44);
  v38 = sub_1A1D720((__int64 *)a1, *(_BYTE **)(a1 + 184), *(__int64 ***)(a1 + 136), *(unsigned int *)(a1 + 144), &v46);
  LOWORD(v43) = 257;
  v14 = sub_1648A60(64, 2u);
  v15 = v14;
  if ( v14 )
    sub_15F9650((__int64)v14, v33, v38, 0, 0);
  v16 = *(_QWORD *)(a1 + 8);
  v17 = *(unsigned __int64 **)(a1 + 16);
  if ( (unsigned __int8)v43 > 1u )
  {
    v40 = *(_QWORD *)(a1 + 8);
    v46.m128i_i64[0] = a1 + 64;
    v34 = v17;
    v47 = 260;
    sub_14EC200(&v44, &v46, &v42);
    v16 = v40;
    v17 = v34;
  }
  else
  {
    v18 = _mm_loadu_si128(&v42);
    v45 = v43;
    v44 = v18;
  }
  v39 = v17;
  if ( v16 )
  {
    sub_157E9D0(v16 + 40, (__int64)v15);
    v19 = *v39;
    v20 = v15[3] & 7LL;
    v15[4] = v39;
    v19 &= 0xFFFFFFFFFFFFFFF8LL;
    v15[3] = v19 | v20;
    *(_QWORD *)(v19 + 8) = v15 + 3;
    *v39 = *v39 & 7 | (unsigned __int64)(v15 + 3);
  }
  sub_164B780((__int64)v15, v44.m128i_i64);
  v21 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v46.m128i_i64[0] = *(_QWORD *)a1;
    sub_1623A60((__int64)&v46, v21, 2);
    v22 = v15[6];
    if ( v22 )
      sub_161E7C0((__int64)(v15 + 6), v22);
    v23 = (unsigned __int8 *)v46.m128i_i64[0];
    v15[6] = v46.m128i_i64[0];
    if ( v23 )
      sub_1623210((__int64)&v46, v23, (__int64)(v15 + 6));
  }
  return sub_15F9450((__int64)v15, a5);
}
