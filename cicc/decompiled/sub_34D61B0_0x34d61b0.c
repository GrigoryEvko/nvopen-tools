// Function: sub_34D61B0
// Address: 0x34d61b0
//
unsigned __int64 __fastcall sub_34D61B0(__int64 a1, unsigned int a2, _QWORD **a3, unsigned int a4)
{
  unsigned int v4; // r12d
  __int64 *v6; // r15
  unsigned int v8; // eax
  __int64 v9; // r12
  __int64 v10; // rcx
  unsigned __int16 v11; // bx
  __int64 v12; // rdx
  __int64 v13; // r13
  unsigned __int16 v14; // ax
  __int64 v15; // r13
  unsigned int v16; // r12d
  unsigned __int16 v17; // dx
  unsigned int v18; // ebx
  unsigned __int64 v19; // r12
  unsigned int v20; // eax
  __int64 v21; // r14
  __int64 v22; // r13
  unsigned int v23; // r15d
  signed __int64 v24; // rax
  int v25; // edi
  int v26; // edx
  bool v27; // of
  __int64 v28; // rax
  int v29; // ecx
  int v30; // edx
  unsigned int v31; // eax
  __int64 v32; // rax
  int v33; // edx
  __int64 v34; // rax
  unsigned __int64 v35; // r12
  signed __int64 v36; // rax
  int v37; // edx
  __int64 v38; // rax
  __int64 v39; // r14
  __int64 v40; // rdx
  __int64 v41; // r12
  unsigned __int64 result; // rax
  __int64 *v43; // r12
  int v44; // edx
  int v45; // eax
  _QWORD *v46; // rdi
  __int64 *v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rax
  unsigned __int16 v50; // r9
  bool v51; // cc
  unsigned __int64 v52; // rax
  unsigned __int64 v53; // rax
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // rax
  __int64 *v56; // [rsp+8h] [rbp-88h]
  unsigned int v57; // [rsp+10h] [rbp-80h]
  int v58; // [rsp+14h] [rbp-7Ch]
  unsigned int v59; // [rsp+18h] [rbp-78h]
  __int64 v60; // [rsp+20h] [rbp-70h]
  int v61; // [rsp+28h] [rbp-68h]
  unsigned __int64 v62; // [rsp+30h] [rbp-60h]
  unsigned int v63; // [rsp+38h] [rbp-58h]
  int v64; // [rsp+38h] [rbp-58h]
  __int64 v65; // [rsp+38h] [rbp-58h]
  unsigned __int16 v66; // [rsp+38h] [rbp-58h]
  __int64 v67; // [rsp+40h] [rbp-50h] BYREF
  __int64 v68; // [rsp+48h] [rbp-48h]
  __int64 v69; // [rsp+50h] [rbp-40h]

  if ( *((_BYTE *)a3 + 8) == 18 )
    return 0;
  v4 = *((_DWORD *)a3 + 8);
  v6 = (__int64 *)a3;
  v56 = a3[3];
  if ( a2 - 28 <= 1 && v56 == (__int64 *)sub_BCB2A0(*a3) && v4 > 1 )
  {
    v43 = (__int64 *)sub_BCCE00((_QWORD *)*v6, v4);
    v44 = *((unsigned __int8 *)v43 + 8);
    if ( (unsigned int)(v44 - 17) > 1 )
    {
      v48 = sub_BCB2A0((_QWORD *)*v43);
    }
    else
    {
      v45 = *((_DWORD *)v43 + 8);
      v46 = (_QWORD *)*v43;
      BYTE4(v67) = (_BYTE)v44 == 18;
      LODWORD(v67) = v45;
      v47 = (__int64 *)sub_BCB2A0(v46);
      v48 = sub_BCE1B0(v47, v67);
    }
    v65 = sub_34D1290(a1, 53, v43, v48, 42, a4, 0, 0, 0);
    v49 = sub_34D3270(a1, 0x31u, (__int64)v43, (__int64)v6, 0, a4, 0);
    v27 = __OFADD__(v65, v49);
    result = v65 + v49;
    if ( v27 )
    {
      result = 0x7FFFFFFFFFFFFFFFLL;
      if ( v65 <= 0 )
        return 0x8000000000000000LL;
    }
    return result;
  }
  v57 = -1;
  if ( v4 )
  {
    _BitScanReverse(&v8, v4);
    v57 = 31 - (v8 ^ 0x1F);
  }
  v63 = v4;
  v9 = *v6;
  v10 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v6, 0);
  v11 = v10;
  v13 = v12;
  while ( 1 )
  {
    LOWORD(v10) = v11;
    sub_2FE6CC0((__int64)&v67, *(_QWORD *)(a1 + 24), v9, v10, v13);
    v14 = v68;
    if ( (_BYTE)v67 == 10 )
      break;
    if ( !(_BYTE)v67 )
    {
      v15 = a1;
      v16 = v63;
      v50 = v11;
      goto LABEL_56;
    }
    if ( v11 == (_WORD)v68 )
    {
      if ( (_WORD)v68 )
      {
        v15 = a1;
        v16 = v63;
        v17 = v68 - 17;
        goto LABEL_11;
      }
      if ( v13 == v69 )
      {
        v15 = a1;
        v16 = v63;
        goto LABEL_12;
      }
    }
    v10 = v68;
    v13 = v69;
    v11 = v68;
  }
  v15 = a1;
  v16 = v63;
  v50 = v11;
  if ( !v11 )
    goto LABEL_12;
LABEL_56:
  v17 = v50 - 17;
  v14 = v50;
LABEL_11:
  if ( v17 > 0xD3u )
  {
LABEL_12:
    v59 = 1;
    goto LABEL_13;
  }
  if ( (unsigned __int16)(v14 - 176) <= 0x34u )
  {
    v66 = v14;
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
    v14 = v66;
  }
  v59 = word_4456340[v14 - 1];
LABEL_13:
  if ( v59 >= v16 )
  {
    v62 = 0;
    v19 = 0;
  }
  else
  {
    v58 = 0;
    v62 = 0;
    v64 = 0;
    v61 = 0;
    v18 = v16;
    v19 = 0;
    v20 = a4;
    v21 = v15;
    v22 = (__int64)v6;
    v23 = v20;
    do
    {
      v18 >>= 1;
      v60 = v22;
      v22 = sub_BCDA70(v56, v18);
      v24 = sub_34D5BE0(v21, 5, v60, 0, 0, v23, v18, v22);
      v25 = 1;
      if ( v26 != 1 )
        v25 = v61;
      v27 = __OFADD__(v24, v19);
      v19 += v24;
      v61 = v25;
      if ( v27 )
      {
        v19 = 0x8000000000000000LL;
        if ( v24 > 0 )
          v19 = 0x7FFFFFFFFFFFFFFFLL;
      }
      v28 = sub_34D2250(v21, a2, v22, v23, 0, 0, 0, 0, 0);
      v29 = 1;
      if ( v30 != 1 )
        v29 = v64;
      v64 = v29;
      if ( __OFADD__(v28, v62) )
      {
        v51 = v28 <= 0;
        v52 = 0x8000000000000000LL;
        if ( !v51 )
          v52 = 0x7FFFFFFFFFFFFFFFLL;
        v62 = v52;
      }
      else
      {
        v62 += v28;
      }
      ++v58;
    }
    while ( v59 < v18 );
    v57 -= v58;
    v31 = v23;
    v6 = (__int64 *)v22;
    v15 = v21;
    a4 = v31;
  }
  v32 = sub_34D5BE0(v15, 7, (__int64)v6, 0, 0, a4, 0, (__int64)v6);
  if ( v33 == 1 )
  {
    if ( is_mul_ok(v32, v57) )
    {
      v34 = v32 * v57;
      goto LABEL_27;
    }
    if ( v57 && v32 > 0 )
    {
      v53 = 0x7FFFFFFFFFFFFFFFLL;
      v27 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v19);
      v35 = v19 + 0x7FFFFFFFFFFFFFFFLL;
      if ( !v27 )
        goto LABEL_28;
    }
    else
    {
      v53 = 0x8000000000000000LL;
      v27 = __OFADD__(0x8000000000000000LL, v19);
      v35 = v19 + 0x8000000000000000LL;
      if ( !v27 )
        goto LABEL_28;
    }
LABEL_71:
    v35 = v53;
    goto LABEL_28;
  }
  if ( !is_mul_ok(v32, v57) )
  {
    if ( v32 > 0 && v57 )
    {
      v53 = 0x7FFFFFFFFFFFFFFFLL;
      v27 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v19);
      v35 = v19 + 0x7FFFFFFFFFFFFFFFLL;
      if ( !v27 )
        goto LABEL_28;
    }
    else
    {
      v53 = 0x8000000000000000LL;
      v27 = __OFADD__(0x8000000000000000LL, v19);
      v35 = v19 + 0x8000000000000000LL;
      if ( !v27 )
        goto LABEL_28;
    }
    goto LABEL_71;
  }
  v34 = v32 * v57;
LABEL_27:
  v27 = __OFADD__(v34, v19);
  v35 = v34 + v19;
  if ( v27 )
  {
    v35 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v34 <= 0 )
      v35 = 0x8000000000000000LL;
  }
LABEL_28:
  v36 = sub_34D2250(v15, a2, (__int64)v6, a4, 0, 0, 0, 0, 0);
  if ( v37 == 1 )
  {
    if ( is_mul_ok(v36, v57) )
    {
      v38 = v36 * v57;
      goto LABEL_31;
    }
    if ( v57 && v36 > 0 )
    {
      v55 = 0x7FFFFFFFFFFFFFFFLL;
      v39 = v62 + 0x7FFFFFFFFFFFFFFFLL;
      if ( !__OFADD__(0x7FFFFFFFFFFFFFFFLL, v62) )
        goto LABEL_101;
    }
    else
    {
      v55 = 0x8000000000000000LL;
      v39 = v62 + 0x8000000000000000LL;
      if ( !__OFADD__(0x8000000000000000LL, v62) )
        goto LABEL_101;
    }
    v39 = v55;
LABEL_101:
    if ( (unsigned int)*((unsigned __int8 *)v6 + 8) - 17 > 1 )
      goto LABEL_34;
    goto LABEL_33;
  }
  if ( !is_mul_ok(v36, v57) )
  {
    if ( v57 && v36 > 0 )
    {
      v54 = 0x7FFFFFFFFFFFFFFFLL;
      v39 = v62 + 0x7FFFFFFFFFFFFFFFLL;
      if ( !__OFADD__(0x7FFFFFFFFFFFFFFFLL, v62) )
        goto LABEL_32;
    }
    else
    {
      v54 = 0x8000000000000000LL;
      v39 = v62 + 0x8000000000000000LL;
      if ( !__OFADD__(0x8000000000000000LL, v62) )
        goto LABEL_32;
    }
    v39 = v54;
    goto LABEL_32;
  }
  v38 = v36 * v57;
LABEL_31:
  v39 = v38 + v62;
  if ( __OFADD__(v38, v62) )
  {
    v39 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v38 <= 0 )
      v39 = 0x8000000000000000LL;
  }
LABEL_32:
  if ( (unsigned int)*((unsigned __int8 *)v6 + 8) - 17 <= 1 )
LABEL_33:
    v6 = *(__int64 **)v6[2];
LABEL_34:
  v40 = (unsigned int)sub_34D06B0(v15, v6);
  v27 = __OFADD__(v39, v35);
  v41 = v39 + v35;
  if ( v27 )
  {
    if ( v39 <= 0 )
      return v40 + 0x8000000000000000LL;
    v41 = 0x7FFFFFFFFFFFFFFFLL;
  }
  result = v40 + v41;
  if ( __OFADD__(v40, v41) )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( !v40 )
      return 0x8000000000000000LL;
  }
  return result;
}
