// Function: sub_258FD00
// Address: 0x258fd00
//
__int64 __fastcall sub_258FD00(
        _QWORD *a1,
        __int64 a2,
        unsigned __int8 *a3,
        char *a4,
        unsigned __int8 *a5,
        _BYTE *a6,
        _BYTE *a7)
{
  __int64 v7; // r15
  int v10; // esi
  __int64 v12; // rax
  unsigned __int8 *v13; // r8
  __int64 v14; // rdx
  unsigned __int8 *v15; // r10
  char v16; // al
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r15
  __int64 v21; // rax
  __m128i v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 result; // rax
  __int64 v27; // r15
  unsigned __int8 *v28; // r14
  unsigned int v29; // eax
  __int64 v30; // r11
  unsigned __int8 *v31; // r10
  unsigned __int8 *v32; // rax
  __int64 v33; // r11
  unsigned __int8 *v34; // r10
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // r15
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // r15
  __int64 v43; // rax
  __int64 v44; // rdx
  unsigned __int8 *v45; // rax
  __int64 v46; // [rsp+18h] [rbp-D8h]
  char v47; // [rsp+27h] [rbp-C9h]
  unsigned __int8 *v49; // [rsp+28h] [rbp-C8h]
  unsigned __int8 *v50; // [rsp+28h] [rbp-C8h]
  unsigned __int8 *v52; // [rsp+30h] [rbp-C0h]
  unsigned __int8 *v53; // [rsp+30h] [rbp-C0h]
  unsigned int v54; // [rsp+30h] [rbp-C0h]
  unsigned __int8 *v55; // [rsp+30h] [rbp-C0h]
  unsigned __int8 *v56; // [rsp+30h] [rbp-C0h]
  __int64 v57; // [rsp+38h] [rbp-B8h]
  __int64 v58; // [rsp+38h] [rbp-B8h]
  __int64 v59; // [rsp+38h] [rbp-B8h]
  __int64 v60; // [rsp+38h] [rbp-B8h]
  __int64 v61; // [rsp+38h] [rbp-B8h]
  char v62; // [rsp+47h] [rbp-A9h] BYREF
  __int64 v63; // [rsp+48h] [rbp-A8h] BYREF
  unsigned __int64 v64; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v65; // [rsp+58h] [rbp-98h]
  __int64 v66; // [rsp+60h] [rbp-90h] BYREF
  __int64 v67; // [rsp+68h] [rbp-88h]
  char *v68; // [rsp+70h] [rbp-80h]
  __int64 *v69; // [rsp+78h] [rbp-78h]
  __m128i v70[3]; // [rsp+80h] [rbp-70h] BYREF
  char v71; // [rsp+B0h] [rbp-40h]

  *a7 = 0;
  v7 = *(_QWORD *)(*(_QWORD *)a4 + 8LL);
  if ( *(_BYTE *)(v7 + 8) != 14 )
    return 0;
  v10 = *a5;
  if ( (unsigned int)(v10 - 67) <= 0xC || (v57 = *(_QWORD *)a4, (_BYTE)v10 == 63) )
  {
    *a7 = 1;
    return 0;
  }
  v12 = sub_B43CB0((__int64)a5);
  v47 = 1;
  v13 = a5;
  v14 = v57;
  v15 = a3;
  if ( v12 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
      v7 = **(_QWORD **)(v7 + 16);
    v16 = sub_B2F070(v12, *(_DWORD *)(v7 + 8) >> 8);
    v13 = a5;
    v15 = a3;
    v47 = v16;
    v14 = v57;
  }
  if ( (unsigned __int8)(*v13 - 34) <= 0x33u )
  {
    v17 = 0x8000000000041LL;
    if ( _bittest64(&v17, (unsigned int)*v13 - 34) )
    {
      if ( (v13[7] & 0x80u) == 0 )
        goto LABEL_13;
      v58 = (__int64)v13;
      v18 = sub_BD2BC0((__int64)v13);
      v13 = (unsigned __int8 *)v58;
      v20 = v18 + v19;
      if ( *(char *)(v58 + 7) >= 0 )
        goto LABEL_13;
      v21 = sub_BD2BC0(v58);
      v13 = (unsigned __int8 *)v58;
      if ( !(unsigned int)((v20 - v21) >> 4) )
        goto LABEL_13;
      v54 = *(_DWORD *)(v58 + 4) & 0x7FFFFFF;
      if ( *(char *)(v58 + 7) >= 0 )
        goto LABEL_13;
      v37 = sub_BD2BC0(v58);
      v13 = (unsigned __int8 *)v58;
      v39 = v37 + v38;
      if ( *(char *)(v58 + 7) >= 0 )
      {
        if ( !(unsigned int)(v39 >> 4) )
          goto LABEL_13;
      }
      else
      {
        v40 = sub_BD2BC0(v58);
        v13 = (unsigned __int8 *)v58;
        if ( !(unsigned int)((v39 - v40) >> 4) )
          goto LABEL_13;
        if ( *(char *)(v58 + 7) < 0 )
        {
          v41 = sub_BD2BC0(v58);
          v42 = (__int64)&a4[-(v58 - 32LL * v54)] >> 5;
          v13 = (unsigned __int8 *)v58;
          if ( *(_DWORD *)(v41 + 8) <= (unsigned int)v42 )
          {
            if ( *(char *)(v58 + 7) >= 0 )
              BUG();
            v43 = sub_BD2BC0(v58);
            v13 = (unsigned __int8 *)v58;
            if ( *(_DWORD *)(v43 + v44 - 4) > (unsigned int)v42 )
            {
              v66 = 0x5A0000002BLL;
              sub_CF93C0(v70, a4, &v66, 2);
              result = 0;
              if ( v70[0].m128i_i32[0] )
              {
                result = v70[0].m128i_i64[1];
                *a6 |= !((v70[0].m128i_i32[0] != 43) & (unsigned __int8)v47);
              }
              return result;
            }
          }
LABEL_13:
          if ( a4 == (char *)(v13 - 32) )
          {
            *a6 |= v47 ^ 1;
            return 0;
          }
          else
          {
            v22.m128i_i64[0] = sub_254C9B0(
                                 (__int64)v13,
                                 (a4 - (char *)&v13[-32 * (*((_DWORD *)v13 + 1) & 0x7FFFFFF)]) >> 5);
            v70[0] = v22;
            sub_258F340(a1, a2, v70, 2, &v66, 0, 0);
            v23 = v70[0].m128i_i64[1];
            v24 = v70[0].m128i_i64[0];
            *a6 |= v66;
            v25 = sub_258DCE0((__int64)a1, v24, v23, a2, 2, 0, 1);
            result = 0;
            if ( v25 )
              return *(unsigned int *)(v25 + 104);
          }
          return result;
        }
      }
      BUG();
    }
  }
  v49 = v15;
  v59 = v14;
  v52 = v13;
  v27 = *(_QWORD *)(a1[26] + 104LL);
  sub_D66840(v70, v13);
  if ( !v71 )
    return 0;
  v28 = (unsigned __int8 *)v70[0].m128i_i64[0];
  if ( v70[0].m128i_i64[0] != v59 )
    return 0;
  if ( v70[0].m128i_i64[1] < 0 )
    return 0;
  v46 = v70[0].m128i_i64[1];
  if ( (v70[0].m128i_i64[1] & 0x4000000000000000LL) != 0 || sub_B46560(v52) )
    return 0;
  v29 = sub_AE43F0(v27, *(_QWORD *)(v59 + 8));
  v30 = v46;
  v31 = v49;
  v65 = v29;
  if ( v29 > 0x40 )
  {
    sub_C43690((__int64)&v64, 0, 0);
    v31 = v49;
    v30 = v46;
  }
  else
  {
    v64 = 0;
  }
  v66 = (__int64)a1;
  v67 = a2;
  v68 = &v62;
  v53 = v31;
  v60 = v30;
  LOBYTE(v63) = 1;
  v62 = 0;
  v69 = &v63;
  v32 = sub_BD45C0(
          v28,
          v27,
          (__int64)&v64,
          0,
          1,
          0,
          (unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *))sub_2589870,
          (__int64)&v66);
  v33 = v60;
  v34 = v53;
  if ( v65 > 0x40 )
  {
    v50 = v53;
    v55 = v32;
    v63 = *(_QWORD *)v64;
    j_j___libc_free_0_0(v64);
    v34 = v50;
    v32 = v55;
    v33 = v60;
  }
  else
  {
    v35 = 0;
    if ( v65 )
      v35 = (__int64)(v64 << (64 - (unsigned __int8)v65)) >> (64 - (unsigned __int8)v65);
    v63 = v35;
  }
  if ( v32 && v34 == v32 )
  {
    LOBYTE(v67) = 0;
    v66 = v33 & 0x3FFFFFFFFFFFFFFFLL;
    v36 = sub_CA1930(&v66);
    *a6 |= v47 ^ 1;
    result = v63 + v36;
    if ( result < 0 )
      return 0;
    return result;
  }
  v56 = v34;
  v61 = v33;
  v45 = sub_25536C0((__int64)v28, &v63, v27, 1);
  if ( !v45 || v56 != v45 || v63 )
    return 0;
  LOBYTE(v67) = 0;
  v66 = v61 & 0x3FFFFFFFFFFFFFFFLL;
  result = sub_CA1930(&v66);
  *a6 |= v47 ^ 1;
  if ( result < 0 )
    return 0;
  return result;
}
