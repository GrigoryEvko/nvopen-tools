// Function: sub_3069790
// Address: 0x3069790
//
unsigned __int64 __fastcall sub_3069790(__int64 a1, int a2, _QWORD **a3, unsigned int a4)
{
  unsigned int v4; // ebx
  __int64 *v6; // r15
  unsigned int v8; // eax
  __int64 v9; // rbx
  __int64 v10; // rcx
  unsigned __int16 v11; // r13
  __int64 v12; // rdx
  __int64 v13; // r12
  unsigned __int16 v14; // ax
  __int64 v15; // r12
  unsigned int v16; // ebx
  unsigned __int16 v17; // dx
  __int64 v18; // r13
  unsigned int v19; // r15d
  __int64 v20; // r14
  unsigned int v21; // r12d
  unsigned __int64 v22; // rbx
  signed __int64 v23; // rax
  int v24; // ecx
  int v25; // edx
  bool v26; // of
  __int64 v27; // rax
  int v28; // edi
  int v29; // edx
  __int64 v30; // rax
  int v31; // edx
  __int64 v32; // rax
  unsigned __int64 v33; // rbx
  signed __int64 v34; // rax
  int v35; // edx
  __int64 v36; // rax
  __int64 v37; // r14
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rsi
  __int64 v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // rbx
  unsigned __int64 result; // rax
  __int64 *v45; // r13
  int v46; // edx
  int v47; // eax
  _QWORD *v48; // rdi
  __int64 *v49; // rax
  __int64 v50; // rcx
  signed __int64 v51; // rax
  unsigned __int16 v52; // r9
  bool v53; // cc
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // rax
  unsigned __int64 v56; // rax
  __int64 *v57; // [rsp+8h] [rbp-88h]
  unsigned int v58; // [rsp+10h] [rbp-80h]
  int v59; // [rsp+14h] [rbp-7Ch]
  unsigned int v60; // [rsp+18h] [rbp-78h]
  __int64 v61; // [rsp+20h] [rbp-70h]
  unsigned __int64 v62; // [rsp+20h] [rbp-70h]
  int v63; // [rsp+28h] [rbp-68h]
  unsigned __int64 v64; // [rsp+30h] [rbp-60h]
  unsigned int v65; // [rsp+38h] [rbp-58h]
  int v66; // [rsp+38h] [rbp-58h]
  __int64 v67; // [rsp+38h] [rbp-58h]
  unsigned __int16 v68; // [rsp+38h] [rbp-58h]
  __int64 v69; // [rsp+40h] [rbp-50h] BYREF
  __int64 v70; // [rsp+48h] [rbp-48h]
  __int64 v71; // [rsp+50h] [rbp-40h]

  if ( *((_BYTE *)a3 + 8) == 18 )
    return 0;
  v4 = *((_DWORD *)a3 + 8);
  v6 = (__int64 *)a3;
  v57 = a3[3];
  if ( (unsigned int)(a2 - 28) <= 1 && v57 == (__int64 *)sub_BCB2A0(*a3) && v4 > 1 )
  {
    v45 = (__int64 *)sub_BCCE00((_QWORD *)*v6, v4);
    v46 = *((unsigned __int8 *)v45 + 8);
    if ( (unsigned int)(v46 - 17) > 1 )
    {
      v50 = sub_BCB2A0((_QWORD *)*v45);
    }
    else
    {
      v47 = *((_DWORD *)v45 + 8);
      v48 = (_QWORD *)*v45;
      BYTE4(v69) = (_BYTE)v46 == 18;
      LODWORD(v69) = v47;
      v49 = (__int64 *)sub_BCB2A0(v48);
      v50 = sub_BCE1B0(v49, v69);
    }
    v67 = sub_3066CD0(a1, 53, v45, v50, 42, a4, 0, 0, 0);
    v51 = sub_3065900(a1, 0x31u, (__int64)v45, (__int64)v6, 0, a4, 0);
    v26 = __OFADD__(v67, v51);
    result = v67 + v51;
    if ( v26 )
    {
      result = 0x7FFFFFFFFFFFFFFFLL;
      if ( v67 <= 0 )
        return 0x8000000000000000LL;
    }
    return result;
  }
  v58 = -1;
  if ( v4 )
  {
    _BitScanReverse(&v8, v4);
    v58 = 31 - (v8 ^ 0x1F);
  }
  v65 = v4;
  v9 = *v6;
  v10 = sub_2D5BAE0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v6, 0);
  v11 = v10;
  v13 = v12;
  while ( 1 )
  {
    LOWORD(v10) = v11;
    sub_2FE6CC0((__int64)&v69, *(_QWORD *)(a1 + 24), v9, v10, v13);
    v14 = v70;
    if ( (_BYTE)v69 == 10 )
      break;
    if ( !(_BYTE)v69 )
    {
      v15 = a1;
      v16 = v65;
      v52 = v11;
      goto LABEL_56;
    }
    if ( v11 == (_WORD)v70 )
    {
      if ( (_WORD)v70 )
      {
        v15 = a1;
        v16 = v65;
        v17 = v70 - 17;
        goto LABEL_11;
      }
      if ( v13 == v71 )
      {
        v15 = a1;
        v16 = v65;
        goto LABEL_12;
      }
    }
    v10 = v70;
    v13 = v71;
    v11 = v70;
  }
  v15 = a1;
  v16 = v65;
  v52 = v11;
  if ( !v11 )
    goto LABEL_12;
LABEL_56:
  v17 = v52 - 17;
  v14 = v52;
LABEL_11:
  if ( v17 > 0xD3u )
  {
LABEL_12:
    v60 = 1;
    goto LABEL_13;
  }
  if ( (unsigned __int16)(v14 - 176) <= 0x34u )
  {
    v68 = v14;
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
    v14 = v68;
  }
  v60 = word_4456340[v14 - 1];
LABEL_13:
  if ( v60 >= v16 )
  {
    v62 = 0;
    v64 = 0;
  }
  else
  {
    v59 = 0;
    v64 = 0;
    v66 = 0;
    v63 = 0;
    v18 = (__int64)v6;
    v19 = a4;
    v20 = v15;
    v21 = v16;
    v22 = 0;
    do
    {
      v21 >>= 1;
      v61 = v18;
      v18 = sub_BCDA70(v57, v21);
      v23 = sub_30690B0(v20, 5, v61, 0, 0, v19, v21, v18);
      v24 = 1;
      if ( v25 != 1 )
        v24 = v63;
      v26 = __OFADD__(v23, v22);
      v22 += v23;
      v63 = v24;
      if ( v26 )
      {
        v22 = 0x8000000000000000LL;
        if ( v23 > 0 )
          v22 = 0x7FFFFFFFFFFFFFFFLL;
      }
      v27 = sub_3075ED0(v20, a2, v18, v19, 0, 0, 0, 0, 0);
      v28 = 1;
      if ( v29 != 1 )
        v28 = v66;
      v66 = v28;
      if ( __OFADD__(v27, v64) )
      {
        v53 = v27 <= 0;
        v54 = 0x8000000000000000LL;
        if ( !v53 )
          v54 = 0x7FFFFFFFFFFFFFFFLL;
        v64 = v54;
      }
      else
      {
        v64 += v27;
      }
      ++v59;
    }
    while ( v60 < v21 );
    v62 = v22;
    v15 = v20;
    a4 = v19;
    v58 -= v59;
    v6 = (__int64 *)v18;
  }
  v30 = sub_30690B0(v15, 7, (__int64)v6, 0, 0, a4, 0, (__int64)v6);
  if ( v31 == 1 )
  {
    if ( is_mul_ok(v30, v58) )
    {
      v32 = v30 * v58;
      goto LABEL_27;
    }
    if ( v58 && v30 > 0 )
    {
      v55 = 0x7FFFFFFFFFFFFFFFLL;
      v33 = v62 + 0x7FFFFFFFFFFFFFFFLL;
      if ( !__OFADD__(0x7FFFFFFFFFFFFFFFLL, v62) )
        goto LABEL_28;
    }
    else
    {
      v55 = 0x8000000000000000LL;
      v33 = v62 + 0x8000000000000000LL;
      if ( !__OFADD__(0x8000000000000000LL, v62) )
        goto LABEL_28;
    }
LABEL_71:
    v33 = v55;
    goto LABEL_28;
  }
  if ( !is_mul_ok(v30, v58) )
  {
    if ( v58 && v30 > 0 )
    {
      v55 = 0x7FFFFFFFFFFFFFFFLL;
      v33 = v62 + 0x7FFFFFFFFFFFFFFFLL;
      if ( !__OFADD__(0x7FFFFFFFFFFFFFFFLL, v62) )
        goto LABEL_28;
    }
    else
    {
      v55 = 0x8000000000000000LL;
      v33 = v62 + 0x8000000000000000LL;
      if ( !__OFADD__(0x8000000000000000LL, v62) )
        goto LABEL_28;
    }
    goto LABEL_71;
  }
  v32 = v30 * v58;
LABEL_27:
  v33 = v32 + v62;
  if ( __OFADD__(v32, v62) )
  {
    v33 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v32 <= 0 )
      v33 = 0x8000000000000000LL;
  }
LABEL_28:
  v34 = sub_3075ED0(v15, a2, (_DWORD)v6, a4, 0, 0, 0, 0, 0);
  if ( v35 == 1 )
  {
    if ( is_mul_ok(v34, v58) )
    {
      v36 = v34 * v58;
      goto LABEL_31;
    }
    if ( v58 && v34 > 0 )
    {
      v56 = 0x7FFFFFFFFFFFFFFFLL;
      v37 = v64 + 0x7FFFFFFFFFFFFFFFLL;
      if ( !__OFADD__(0x7FFFFFFFFFFFFFFFLL, v64) )
        goto LABEL_32;
    }
    else
    {
      v56 = 0x8000000000000000LL;
      v37 = v64 + 0x8000000000000000LL;
      if ( !__OFADD__(0x8000000000000000LL, v64) )
        goto LABEL_32;
    }
    goto LABEL_92;
  }
  if ( is_mul_ok(v34, v58) )
  {
    v36 = v34 * v58;
LABEL_31:
    v37 = v36 + v64;
    if ( __OFADD__(v36, v64) )
    {
      v37 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v36 <= 0 )
        v37 = 0x8000000000000000LL;
    }
    goto LABEL_32;
  }
  if ( !v58 || v34 <= 0 )
  {
    v56 = 0x8000000000000000LL;
    v37 = v64 + 0x8000000000000000LL;
    if ( !__OFADD__(0x8000000000000000LL, v64) )
      goto LABEL_32;
    goto LABEL_92;
  }
  v56 = 0x7FFFFFFFFFFFFFFFLL;
  v37 = v64 + 0x7FFFFFFFFFFFFFFFLL;
  if ( __OFADD__(0x7FFFFFFFFFFFFFFFLL, v64) )
LABEL_92:
    v37 = v56;
LABEL_32:
  if ( (unsigned int)*((unsigned __int8 *)v6 + 8) - 17 <= 1 )
    v6 = *(__int64 **)v6[2];
  v38 = sub_2D5BAE0(*(_QWORD *)(v15 + 24), *(_QWORD *)(v15 + 8), v6, 0);
  v39 = *(_QWORD *)(v15 + 24);
  v40 = *v6;
  BYTE2(v69) = 0;
  v42 = (*(unsigned int (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v39 + 736LL))(
          v39,
          v40,
          v38,
          v41,
          v69);
  v26 = __OFADD__(v37, v33);
  v43 = v37 + v33;
  if ( v26 )
  {
    if ( v37 <= 0 )
      return v42 + 0x8000000000000000LL;
    v43 = 0x7FFFFFFFFFFFFFFFLL;
  }
  result = v42 + v43;
  if ( __OFADD__(v42, v43) )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( !v42 )
      return 0x8000000000000000LL;
  }
  return result;
}
