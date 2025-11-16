// Function: sub_1A20950
// Address: 0x1a20950
//
_QWORD *__fastcall sub_1A20950(
        _BYTE *a1,
        __int64 a2,
        __int64 *a3,
        __int64 *a4,
        unsigned __int64 a5,
        const __m128i *a6,
        double a7,
        double a8,
        double a9)
{
  __int64 v9; // r14
  __int64 v12; // r15
  __int64 v13; // rbx
  _QWORD *result; // rax
  __int64 **v15; // r12
  _QWORD *v16; // r12
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // rax
  _QWORD *v20; // rax
  _QWORD *v21; // rax
  _QWORD *v22; // r13
  unsigned __int64 v24; // [rsp+18h] [rbp-98h]
  __int64 v25; // [rsp+18h] [rbp-98h]
  __m128i v26; // [rsp+20h] [rbp-90h] BYREF
  char v27; // [rsp+30h] [rbp-80h]
  char v28; // [rsp+31h] [rbp-7Fh]
  __m128i v29; // [rsp+40h] [rbp-70h] BYREF
  char v30; // [rsp+50h] [rbp-60h]
  char v31; // [rsp+51h] [rbp-5Fh]
  __m128i v32; // [rsp+60h] [rbp-50h] BYREF
  __int16 v33; // [rsp+70h] [rbp-40h]

  v9 = (__int64)a3;
  v12 = *a3;
  v24 = (unsigned __int64)(sub_127FA20((__int64)a1, (__int64)a4) + 7) >> 3;
  if ( (unsigned __int64)(sub_127FA20((__int64)a1, v12) + 7) >> 3 == 2 * v24 && (!a5 || v24 == a5) )
  {
    v31 = 1;
    v15 = (__int64 **)sub_16463B0(a4, 2u);
    v29.m128i_i64[0] = (__int64)".castvec";
    v30 = 3;
    sub_14EC200(&v32, a6, &v29);
    v16 = sub_1A1C8D0((__int64 *)a2, 47, v9, v15, &v32);
    v17 = sub_1643350(*(_QWORD **)(a2 + 24));
    v28 = 1;
    v18 = sub_159C470(v17, (unsigned int)(a5 / v24), 0);
    v27 = 3;
    v26.m128i_i64[0] = (__int64)".extract";
    sub_14EC200(&v29, a6, &v26);
    if ( *((_BYTE *)v16 + 16) > 0x10u || *(_BYTE *)(v18 + 16) > 0x10u )
    {
      v33 = 257;
      v21 = sub_1648A60(56, 2u);
      v22 = v21;
      if ( v21 )
        sub_15FA320((__int64)v21, v16, v18, (__int64)&v32, 0);
      return sub_1A1C7B0((__int64 *)a2, v22, &v29);
    }
    else
    {
      return (_QWORD *)sub_15A37D0(v16, v18, 0);
    }
  }
  else
  {
    if ( *a1 )
    {
      v25 = sub_127FA20((__int64)a1, v12);
      v13 = 8
          * (((unsigned __int64)(v25 + 7) >> 3)
           - a5
           - ((unsigned __int64)(sub_127FA20((__int64)a1, (__int64)a4) + 7) >> 3));
    }
    else
    {
      v13 = 8 * a5;
    }
    if ( v13 )
    {
      v28 = 1;
      v27 = 3;
      v26.m128i_i64[0] = (__int64)".shift";
      sub_14EC200(&v29, a6, &v26);
      v19 = sub_15A0680(*(_QWORD *)v9, v13, 0);
      if ( *(_BYTE *)(v9 + 16) > 0x10u || *(_BYTE *)(v19 + 16) > 0x10u )
      {
        v33 = 257;
        v20 = (_QWORD *)sub_15FB440(24, (__int64 *)v9, v19, (__int64)&v32, 0);
        v9 = (__int64)sub_1A1C7B0((__int64 *)a2, v20, &v29);
      }
      else
      {
        v9 = sub_15A2D80((__int64 *)v9, v19, 0, a7, a8, a9);
      }
    }
    result = (_QWORD *)v9;
    if ( (__int64 *)v12 != a4 )
    {
      v31 = 1;
      v29.m128i_i64[0] = (__int64)".trunc";
      v30 = 3;
      sub_14EC200(&v32, a6, &v29);
      return sub_1A1C8D0((__int64 *)a2, 36, v9, (__int64 **)a4, &v32);
    }
  }
  return result;
}
