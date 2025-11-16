// Function: sub_18BD0B0
// Address: 0x18bd0b0
//
_QWORD *__fastcall sub_18BD0B0(
        __int64 **a1,
        __int64 *a2,
        unsigned __int64 a3,
        __int64 a4,
        _DWORD *a5,
        __m128 a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        __m128 a13,
        __int64 a14,
        _BYTE *a15,
        __int64 a16)
{
  __int64 *v20; // rax
  __int64 v21; // rax
  __int64 v22; // r9
  __int64 v23; // rax
  _BYTE *v24; // rsi
  __int64 *v25; // r12
  _BYTE *v26; // rsi
  __int64 v27; // rax
  _BYTE *v28; // rsi
  _BYTE *v29; // rdx
  __int64 v30; // rax
  __int64 *v31; // rax
  __int64 v32; // r14
  _QWORD *v33; // rax
  __int64 v34; // r12
  __int64 v35; // r14
  __int64 v36; // rbx
  _QWORD *v37; // rax
  __int64 v38; // r14
  _QWORD *v39; // rdi
  double v40; // xmm4_8
  double v41; // xmm5_8
  _QWORD *result; // rax
  __int64 v43; // rdi
  __int64 v44; // r14
  __int64 *v45; // r14
  __int64 v46; // rdx
  _BYTE *v47; // rax
  __int64 v48; // [rsp+0h] [rbp-E0h]
  __int64 *v49; // [rsp+8h] [rbp-D8h]
  __int64 v50; // [rsp+8h] [rbp-D8h]
  char v51; // [rsp+10h] [rbp-D0h]
  __int64 *v52; // [rsp+10h] [rbp-D0h]
  __int64 v53; // [rsp+10h] [rbp-D0h]
  __int64 v55; // [rsp+20h] [rbp-C0h]
  __int64 *v57; // [rsp+30h] [rbp-B0h] BYREF
  _BYTE *v58; // [rsp+38h] [rbp-A8h]
  _BYTE *v59; // [rsp+40h] [rbp-A0h]
  __int64 v60[2]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v61; // [rsp+60h] [rbp-80h]
  _QWORD *v62; // [rsp+70h] [rbp-70h] BYREF
  _QWORD v63[2]; // [rsp+80h] [rbp-60h] BYREF
  int v64; // [rsp+90h] [rbp-50h]

  v61 = 260;
  v60[0] = (__int64)(*a1 + 30);
  sub_16E1010((__int64)&v62, (__int64)v60);
  if ( v64 != 32 || (unsigned int)dword_4FAD7A0 < a3 )
    goto LABEL_31;
  if ( *(_BYTE *)(a4 + 24) )
  {
    v43 = *(_QWORD *)(a4 + 80);
    v44 = a4 + 64;
    if ( v43 == v44 )
      goto LABEL_31;
    while ( *(_BYTE *)(v43 + 80) )
    {
      v43 = sub_220EEE0(v43);
      if ( v44 == v43 )
        goto LABEL_31;
    }
  }
  v60[0] = (__int64)a1[6];
  v20 = (__int64 *)sub_1643270((_QWORD *)**a1);
  v21 = sub_1644EA0(v20, v60, 1, 1u);
  if ( *a15 )
  {
    v53 = v21;
    v45 = *a1;
    v60[0] = (__int64)"branch_funnel";
    v61 = 259;
    v55 = sub_1648B60(120);
    if ( v55 )
      sub_15E2490(v55, v53, 7, (__int64)v60, (__int64)v45);
  }
  else
  {
    v48 = v21;
    v49 = *a1;
    sub_18B61E0(v60, (__int64)a15, a16, 0, 0, v22, "branch_funnel", 0xDu);
    v57 = v60;
    LOWORD(v59) = 260;
    v23 = sub_1648B60(120);
    v55 = v23;
    if ( v23 )
      sub_15E2490(v23, v48, 0, (__int64)&v57, (__int64)v49);
    sub_2240A30(v60);
    v51 = *(_BYTE *)(v55 + 32);
    *(_BYTE *)(v55 + 32) = v51 & 0xCF | 0x10;
    if ( (v51 & 0xF) != 9 )
      *(_BYTE *)(v55 + 33) |= 0x40u;
  }
  sub_15E0D50(v55, 1, 19);
  v57 = 0;
  v58 = 0;
  v59 = 0;
  if ( (*(_BYTE *)(v55 + 18) & 1) == 0 )
  {
    v24 = 0;
    v60[0] = *(_QWORD *)(v55 + 88);
LABEL_11:
    sub_12879C0((__int64)&v57, v24, v60);
    goto LABEL_12;
  }
  sub_15E08E0(v55, 1);
  v46 = *(_QWORD *)(v55 + 88);
  v47 = v58;
  v24 = v59;
  v60[0] = v46;
  if ( v58 == v59 )
    goto LABEL_11;
  if ( v58 )
  {
    *(_QWORD *)v58 = v46;
    v47 = v58;
  }
  v58 = v47 + 8;
LABEL_12:
  v25 = &a2[4 * a3];
  while ( v25 != a2 )
  {
    while ( 1 )
    {
      v27 = sub_18B4CA0((__int64)a1, a2[1]);
      v28 = v58;
      v29 = v59;
      v60[0] = v27;
      if ( v58 == v59 )
        break;
      if ( v58 )
      {
        *(_QWORD *)v58 = v27;
        v28 = v58;
        v29 = v59;
      }
      v30 = *a2;
      v26 = v28 + 8;
      v58 = v26;
      v60[0] = v30;
      if ( v29 == v26 )
        goto LABEL_20;
LABEL_14:
      *(_QWORD *)v26 = v30;
      v26 = v58;
LABEL_15:
      a2 += 4;
      v58 = v26 + 8;
      if ( v25 == a2 )
        goto LABEL_21;
    }
    sub_12879C0((__int64)&v57, v58, v60);
    v30 = *a2;
    v26 = v58;
    v60[0] = *a2;
    if ( v59 != v58 )
    {
      if ( !v58 )
        goto LABEL_15;
      goto LABEL_14;
    }
LABEL_20:
    a2 += 4;
    sub_12879C0((__int64)&v57, v26, v60);
  }
LABEL_21:
  v31 = *a1;
  v61 = 257;
  v32 = *v31;
  v33 = (_QWORD *)sub_22077B0(64);
  v34 = (__int64)v33;
  if ( v33 )
    sub_157FB60(v33, v32, (__int64)v60, v55, 0);
  v35 = sub_15E26F0(*a1, 108, 0, 0);
  v61 = 257;
  v52 = v57;
  v36 = (v58 - (_BYTE *)v57) >> 3;
  v37 = sub_1648A60(72, (int)v36 + 1);
  if ( v37 )
  {
    v50 = (__int64)v37;
    sub_15F1F50(
      (__int64)v37,
      **(_QWORD **)(*(_QWORD *)(*(_QWORD *)v35 + 24LL) + 16LL),
      54,
      (__int64)&v37[-3 * v36 - 3],
      v36 + 1,
      v34);
    *(_QWORD *)(v50 + 56) = 0;
    sub_15F5B40(v50, *(_QWORD *)(*(_QWORD *)v35 + 24LL), v35, v52, v36, (__int64)v60, 0, 0);
    v37 = (_QWORD *)v50;
  }
  *((_WORD *)v37 + 9) = *((_WORD *)v37 + 9) & 0xFFFC | 2;
  v38 = **a1;
  v39 = sub_1648A60(56, 0);
  if ( v39 )
    sub_15F7090((__int64)v39, v38, 0, v34);
  LOBYTE(v60[0]) = 0;
  sub_18BD000((__int64)a1, a4, v55, v60, a6, a7, a8, a9, v40, v41, a12, a13);
  if ( LOBYTE(v60[0]) )
    *a5 = 2;
  if ( v57 )
    j_j___libc_free_0(v57, v59 - (_BYTE *)v57);
LABEL_31:
  result = v63;
  if ( v62 != v63 )
    return (_QWORD *)j_j___libc_free_0(v62, v63[0] + 1LL);
  return result;
}
