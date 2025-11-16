// Function: sub_1A202F0
// Address: 0x1a202f0
//
_QWORD *__fastcall sub_1A202F0(
        _BYTE *a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        unsigned __int64 a5,
        const __m128i *a6,
        double a7,
        double a8,
        double a9)
{
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // r14
  unsigned __int64 v12; // r15
  __int32 v13; // eax
  unsigned __int32 v14; // edi
  __int64 v15; // rsi
  unsigned __int64 v16; // rdx
  __int64 v17; // r13
  unsigned __int8 v18; // al
  __int64 v19; // r14
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r15
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // r15
  _QWORD *v30; // rax
  _QWORD *v31; // rax
  _QWORD *v32; // rax
  unsigned int v33; // esi
  __int64 **v34; // [rsp+8h] [rbp-C8h]
  _QWORD *v37; // [rsp+18h] [rbp-B8h]
  __int64 v39; // [rsp+20h] [rbp-B0h]
  int v40; // [rsp+20h] [rbp-B0h]
  unsigned __int64 v42; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v43; // [rsp+38h] [rbp-98h]
  __m128i v44; // [rsp+40h] [rbp-90h] BYREF
  char v45; // [rsp+50h] [rbp-80h]
  char v46; // [rsp+51h] [rbp-7Fh]
  __m128i v47; // [rsp+60h] [rbp-70h] BYREF
  char v48; // [rsp+70h] [rbp-60h]
  char v49; // [rsp+71h] [rbp-5Fh]
  __m128i v50; // [rsp+80h] [rbp-50h] BYREF
  __int16 v51; // [rsp+90h] [rbp-40h]

  v9 = (__int64)a4;
  v10 = *a4;
  v11 = *(_QWORD *)a3;
  v12 = (unsigned __int64)(sub_127FA20((__int64)a1, *a4) + 7) >> 3;
  if ( (unsigned __int64)(sub_127FA20((__int64)a1, v11) + 7) >> 3 == 2 * v12 && (!a5 || v12 == a5) )
  {
    v49 = 1;
    v34 = (__int64 **)sub_16463B0((__int64 *)v10, 2u);
    v47.m128i_i64[0] = (__int64)".castvec";
    v48 = 3;
    sub_14EC200(&v50, a6, &v47);
    v37 = sub_1A1C8D0((__int64 *)a2, 47, a3, v34, &v50);
    v23 = sub_1643350(*(_QWORD **)(a2 + 24));
    v24 = sub_159C470(v23, (unsigned int)(a5 / v12), 0);
    v49 = 1;
    v25 = v24;
    v48 = 3;
    v47.m128i_i64[0] = (__int64)".insert";
    sub_14EC200(&v50, a6, &v47);
    v26 = sub_1A1D560((__int64 *)a2, (__int64)v37, v9, v25, &v50);
    v49 = 1;
    v27 = v26;
    v48 = 3;
    v47.m128i_i64[0] = (__int64)".castback";
    sub_14EC200(&v50, a6, &v47);
    return sub_1A1C8D0((__int64 *)a2, 47, v27, (__int64 **)v11, &v50);
  }
  if ( v11 != v10 )
  {
    v49 = 1;
    v47.m128i_i64[0] = (__int64)".ext";
    v48 = 3;
    sub_14EC200(&v50, a6, &v47);
    v9 = (__int64)sub_1A1C8D0((__int64 *)a2, 37, v9, (__int64 **)v11, &v50);
  }
  if ( *a1 )
  {
    v29 = sub_127FA20((__int64)a1, v11);
    v39 = 8 * (((unsigned __int64)(v29 + 7) >> 3) - a5 - ((unsigned __int64)(sub_127FA20((__int64)a1, v10) + 7) >> 3));
  }
  else
  {
    v39 = 8 * a5;
  }
  if ( v39 )
  {
    v46 = 1;
    v45 = 3;
    v44.m128i_i64[0] = (__int64)".shift";
    sub_14EC200(&v47, a6, &v44);
    v28 = sub_15A0680(*(_QWORD *)v9, v39, 0);
    if ( *(_BYTE *)(v9 + 16) > 0x10u || *(_BYTE *)(v28 + 16) > 0x10u )
    {
      v51 = 257;
      v32 = (_QWORD *)sub_15FB440(23, (__int64 *)v9, v28, (__int64)&v50, 0);
      v9 = (__int64)sub_1A1C7B0((__int64 *)a2, v32, &v47);
    }
    else
    {
      v9 = sub_15A2D50((__int64 *)v9, v28, 0, 0, a7, a8, a9);
    }
  }
  else if ( *(_DWORD *)(v10 + 8) >> 8 >= *(_DWORD *)(v11 + 8) >> 8 )
  {
    return (_QWORD *)v9;
  }
  sub_1643380((__int64)&v44, v10);
  sub_16A5C50((__int64)&v47, (const void **)&v44, *(_DWORD *)(v11 + 8) >> 8);
  v13 = v47.m128i_i32[2];
  v50.m128i_i32[2] = v47.m128i_i32[2];
  if ( v47.m128i_i32[2] <= 0x40u )
  {
    v14 = v47.m128i_u32[2];
    v50.m128i_i64[0] = v47.m128i_i64[0];
    goto LABEL_10;
  }
  sub_16A4FD0((__int64)&v50, (const void **)&v47);
  v13 = v50.m128i_i32[2];
  if ( v50.m128i_i32[2] <= 0x40u )
  {
    v14 = v47.m128i_u32[2];
LABEL_10:
    v15 = -1;
    if ( (_DWORD)v39 != v13 )
      v15 = ~((v50.m128i_i64[0] << v39) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v13));
    goto LABEL_12;
  }
  sub_16A7DC0(v50.m128i_i64, v39);
  v13 = v50.m128i_i32[2];
  if ( v50.m128i_i32[2] > 0x40u )
  {
    sub_16A8F40(v50.m128i_i64);
    v13 = v50.m128i_i32[2];
    v16 = v50.m128i_i64[0];
    v14 = v47.m128i_u32[2];
    goto LABEL_13;
  }
  v14 = v47.m128i_u32[2];
  v15 = ~v50.m128i_i64[0];
LABEL_12:
  v16 = v15 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v13);
LABEL_13:
  v43 = v13;
  v42 = v16;
  if ( v14 > 0x40 && v47.m128i_i64[0] )
    j_j___libc_free_0_0(v47.m128i_i64[0]);
  if ( v44.m128i_i32[2] > 0x40u && v44.m128i_i64[0] )
    j_j___libc_free_0_0(v44.m128i_i64[0]);
  v44.m128i_i64[0] = (__int64)".mask";
  v46 = 1;
  v45 = 3;
  sub_14EC200(&v47, a6, &v44);
  v17 = sub_15A1070(*(_QWORD *)a3, (__int64)&v42);
  v18 = *(_BYTE *)(v17 + 16);
  if ( v18 > 0x10u )
  {
LABEL_39:
    v51 = 257;
    v30 = (_QWORD *)sub_15FB440(26, (__int64 *)a3, v17, (__int64)&v50, 0);
    v19 = (__int64)sub_1A1C7B0((__int64 *)a2, v30, &v47);
    goto LABEL_23;
  }
  if ( v18 != 13 )
  {
LABEL_21:
    if ( *(_BYTE *)(a3 + 16) <= 0x10u )
    {
      v19 = sub_15A2CF0((__int64 *)a3, v17, a7, a8, a9);
      goto LABEL_23;
    }
    goto LABEL_39;
  }
  v33 = *(_DWORD *)(v17 + 32);
  if ( v33 <= 0x40 )
  {
    v19 = a3;
    if ( *(_QWORD *)(v17 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v33) )
      goto LABEL_21;
  }
  else
  {
    v40 = *(_DWORD *)(v17 + 32);
    v19 = a3;
    if ( v40 != (unsigned int)sub_16A58F0(v17 + 24) )
      goto LABEL_21;
  }
LABEL_23:
  v46 = 1;
  v44.m128i_i64[0] = (__int64)".insert";
  v45 = 3;
  sub_14EC200(&v47, a6, &v44);
  if ( *(_BYTE *)(v9 + 16) > 0x10u )
    goto LABEL_40;
  if ( !sub_1593BB0(v9, (__int64)a6, v20, v21) )
  {
    if ( *(_BYTE *)(v19 + 16) <= 0x10u )
    {
      v19 = sub_15A2D10((__int64 *)v19, v9, a7, a8, a9);
      goto LABEL_27;
    }
LABEL_40:
    v51 = 257;
    v31 = (_QWORD *)sub_15FB440(27, (__int64 *)v19, v9, (__int64)&v50, 0);
    v19 = (__int64)sub_1A1C7B0((__int64 *)a2, v31, &v47);
  }
LABEL_27:
  if ( v43 > 0x40 && v42 )
    j_j___libc_free_0_0(v42);
  return (_QWORD *)v19;
}
