// Function: sub_2B16B50
// Address: 0x2b16b50
//
unsigned __int64 *__fastcall sub_2B16B50(
        unsigned __int64 *a1,
        __int64 **a2,
        __int64 *a3,
        __int64 a4,
        __int64 *a5,
        int a6,
        int a7,
        __int64 a8,
        __int64 a9)
{
  __int64 *v9; // r13
  __int64 v11; // r12
  __int64 *v12; // r15
  __int64 v13; // rax
  __int64 *v14; // r12
  __int64 *v15; // r14
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned __int64 v18; // r14
  char v19; // al
  int v20; // edx
  __int64 *v21; // rbx
  _BYTE *v22; // rdx
  __int64 *v23; // rcx
  char v24; // al
  _BYTE *v25; // rdx
  char v26; // al
  _BYTE *v27; // rdx
  char v28; // al
  unsigned __int8 v29; // al
  int v30; // r15d
  unsigned __int64 v31; // rbx
  _BYTE *v32; // rdi
  __int64 *v33; // rcx
  char v34; // al
  _BYTE *v35; // rdi
  char v36; // al
  _BYTE *v37; // rdi
  char v38; // al
  __int64 v39; // rax
  int v40; // edx
  __int64 v41; // rdx
  __int64 *v42; // rbx
  __int64 v43; // r15
  __int64 *v44; // rax
  int v45; // r8d
  __int64 *v46; // rcx
  __int64 v47; // rbx
  __int64 v48; // r15
  __int64 v49; // rax
  unsigned __int64 v50; // rax
  int v51; // edx
  unsigned __int64 v53; // rax
  __int64 v54; // r9
  __int64 v55; // r8
  int v56; // edx
  __int64 v57; // rdx
  __int64 *v58; // r15
  unsigned __int64 v59; // rdx
  _BYTE *v60; // rbx
  __int64 v61; // rax
  __int64 v62; // rcx
  __int64 *v63; // rsi
  unsigned __int8 v64; // al
  unsigned __int8 v65; // al
  unsigned __int8 v66; // al
  __int64 v67; // [rsp-8h] [rbp-C8h]
  __int64 v68; // [rsp+8h] [rbp-B8h]
  __int64 v69; // [rsp+10h] [rbp-B0h]
  int v70; // [rsp+18h] [rbp-A8h]
  __int64 *v73; // [rsp+30h] [rbp-90h]
  __int64 v74; // [rsp+30h] [rbp-90h]
  int v76; // [rsp+4Ch] [rbp-74h] BYREF
  __int64 *v77; // [rsp+50h] [rbp-70h] BYREF
  __int64 v78; // [rsp+58h] [rbp-68h]
  _BYTE v79[96]; // [rsp+60h] [rbp-60h] BYREF

  v9 = a5;
  v73 = &a3[a4];
  if ( (unsigned int)(a6 - 32) <= 1 )
  {
    LODWORD(v77) = 7;
    v53 = sub_DF9530(a2, a3, a4, a5, &v77, a8, a7);
    v55 = v67;
    v77 = (__int64 *)v79;
    v18 = v53;
    v70 = v56;
    v78 = 0x600000000LL;
    if ( a3 == v73 )
    {
      if ( !a4 )
      {
        *a1 = 0;
        a1[1] = 0;
        a1[2] = 0;
        a1[3] = 0;
        return a1;
      }
      v57 = 0;
      v63 = (__int64 *)v79;
      goto LABEL_82;
    }
    v57 = 0;
    v58 = a3;
    while ( 1 )
    {
      v60 = (_BYTE *)*v58;
      v61 = v57;
      if ( v9 == (__int64 *)*v58 )
        break;
      if ( *v60 == 63 && (v62 = *((_QWORD *)v60 + 2)) != 0 )
      {
        if ( !*(_QWORD *)(v62 + 8) )
          goto LABEL_70;
        v59 = v57 + 1;
        if ( v59 > HIDWORD(v78) )
          goto LABEL_76;
      }
      else
      {
        v59 = v57 + 1;
        if ( v59 > HIDWORD(v78) )
          goto LABEL_76;
      }
LABEL_69:
      v77[v61] = (__int64)v60;
      v57 = (unsigned int)(v78 + 1);
      LODWORD(v78) = v78 + 1;
LABEL_70:
      if ( v73 == ++v58 )
      {
        v63 = v77;
        if ( a4 == v57 )
        {
          *a1 = 0;
          a1[1] = 0;
          a1[2] = 0;
          a1[3] = 0;
          if ( v63 != (__int64 *)v79 )
            _libc_free((unsigned __int64)v63);
          return a1;
        }
LABEL_82:
        v76 = 5;
        v50 = sub_DF9530(a2, v63, v57, v9, &v76, a9, a7);
LABEL_51:
        v31 = v50;
        v30 = v51;
        if ( v77 != (__int64 *)v79 )
          _libc_free((unsigned __int64)v77);
        goto LABEL_53;
      }
    }
    v59 = v57 + 1;
    if ( v59 <= HIDWORD(v78) )
      goto LABEL_69;
LABEL_76:
    sub_C8D5F0((__int64)&v77, v79, v59, 8u, v55, v54);
    v61 = (unsigned int)v78;
    goto LABEL_69;
  }
  v11 = (8 * a4) >> 5;
  v12 = a3;
  v69 = v11;
  v68 = (8 * a4) >> 3;
  if ( v11 <= 0 )
  {
    v39 = (8 * a4) >> 3;
    v14 = a3;
LABEL_35:
    if ( v39 != 2 )
    {
      if ( v39 != 3 )
      {
        if ( v39 != 1 )
          goto LABEL_27;
        goto LABEL_38;
      }
      if ( *(_BYTE *)*v14 != 63 || (unsigned __int8)sub_B4DD90(*v14) )
        goto LABEL_5;
      ++v14;
    }
    if ( *(_BYTE *)*v14 != 63 || (unsigned __int8)sub_B4DD90(*v14) )
      goto LABEL_5;
    ++v14;
LABEL_38:
    if ( *(_BYTE *)*v14 == 63 && !(unsigned __int8)sub_B4DD90(*v14) )
      goto LABEL_27;
    goto LABEL_5;
  }
  v13 = 4 * v11;
  v14 = a3;
  v15 = &a3[v13];
  while ( *(_BYTE *)*v14 == 63 && !(unsigned __int8)sub_B4DD90(*v14) )
  {
    v32 = (_BYTE *)v14[1];
    v33 = v14 + 1;
    if ( *v32 != 63 )
      goto LABEL_26;
    v34 = sub_B4DD90((__int64)v32);
    v33 = v14 + 1;
    if ( v34 )
      goto LABEL_26;
    v35 = (_BYTE *)v14[2];
    v33 = v14 + 2;
    if ( *v35 != 63
      || (v36 = sub_B4DD90((__int64)v35), v33 = v14 + 2, v36)
      || (v37 = (_BYTE *)v14[3], v33 = v14 + 3, *v37 != 63)
      || (v38 = sub_B4DD90((__int64)v37), v33 = v14 + 3, v38) )
    {
LABEL_26:
      if ( v33 != v73 )
        goto LABEL_6;
      goto LABEL_27;
    }
    v14 += 4;
    if ( v15 == v14 )
    {
      v39 = v73 - v14;
      goto LABEL_35;
    }
  }
LABEL_5:
  if ( v14 != v73 )
  {
LABEL_6:
    v76 = 5;
    goto LABEL_7;
  }
LABEL_27:
  v76 = 1;
LABEL_7:
  v18 = sub_DF9530(a2, a3, a4, v9, &v76, a8, a7);
  v19 = *(_BYTE *)v9;
  v70 = v20;
  if ( *(_BYTE *)v9 <= 0x1Cu )
  {
    if ( v19 == 5 && *((_WORD *)v9 + 1) == 34 )
      goto LABEL_43;
  }
  else if ( v19 == 63 )
  {
LABEL_43:
    v40 = *((_DWORD *)v9 + 1);
    v77 = (__int64 *)v79;
    v41 = v40 & 0x7FFFFFF;
    v78 = 0x600000000LL;
    v42 = &v9[4 * (1 - v41)];
    v43 = (-32 * (1 - v41)) >> 5;
    if ( (unsigned __int64)(-32 * (1 - v41)) > 0xC0 )
    {
      sub_C8D5F0((__int64)&v77, v79, (-32 * (1 - v41)) >> 5, 8u, v16, v17);
      v46 = v77;
      v45 = v78;
      v44 = &v77[(unsigned int)v78];
    }
    else
    {
      v44 = (__int64 *)v79;
      v45 = 0;
      v46 = (__int64 *)v79;
    }
    if ( v42 != v9 )
    {
      do
      {
        if ( v44 )
          *v44 = *v42;
        v42 += 4;
        ++v44;
      }
      while ( v42 != v9 );
      v46 = v77;
      v45 = v78;
    }
    v74 = (__int64)v46;
    LODWORD(v78) = v45 + v43;
    v47 = (unsigned int)(v45 + v43);
    v48 = v9[-4 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)];
    v49 = sub_BB5290((__int64)v9);
    v50 = sub_DF9500(a2, v49, v48, v74, v47);
    goto LABEL_51;
  }
  if ( v69 <= 0 )
  {
LABEL_94:
    if ( v68 != 2 )
    {
      if ( v68 != 3 )
      {
        if ( v68 != 1 )
          goto LABEL_23;
LABEL_102:
        v66 = *(_BYTE *)*v12;
        if ( v66 <= 0x1Cu )
        {
          if ( v66 != 5 || *(_WORD *)(*v12 + 2) != 34 )
            goto LABEL_23;
        }
        else if ( v66 != 63 )
        {
          goto LABEL_23;
        }
        goto LABEL_22;
      }
      v64 = *(_BYTE *)*v12;
      if ( v64 <= 0x1Cu )
      {
        if ( v64 == 5 && *(_WORD *)(*v12 + 2) == 34 )
          goto LABEL_22;
      }
      else if ( v64 == 63 )
      {
        goto LABEL_22;
      }
      ++v12;
    }
    v65 = *(_BYTE *)*v12;
    if ( v65 <= 0x1Cu )
    {
      if ( v65 != 5 || *(_WORD *)(*v12 + 2) != 34 )
        goto LABEL_101;
    }
    else if ( v65 != 63 )
    {
LABEL_101:
      ++v12;
      goto LABEL_102;
    }
LABEL_22:
    if ( v12 == v73 )
      goto LABEL_23;
    v9 = (__int64 *)*v12;
    goto LABEL_43;
  }
  v21 = &a3[4 * v69];
  while ( 1 )
  {
    v29 = *(_BYTE *)*v12;
    if ( v29 > 0x1Cu )
    {
      if ( v29 == 63 )
        goto LABEL_22;
    }
    else if ( v29 == 5 && *(_WORD *)(*v12 + 2) == 34 )
    {
      goto LABEL_22;
    }
    v22 = (_BYTE *)v12[1];
    v23 = v12 + 1;
    v24 = *v22;
    if ( *v22 <= 0x1Cu )
      break;
    if ( v24 == 63 )
      goto LABEL_57;
LABEL_14:
    v25 = (_BYTE *)v12[2];
    v23 = v12 + 2;
    v26 = *v25;
    if ( *v25 > 0x1Cu )
    {
      if ( v26 == 63 )
        goto LABEL_57;
LABEL_16:
      v27 = (_BYTE *)v12[3];
      v23 = v12 + 3;
      v28 = *v27;
      if ( *v27 > 0x1Cu )
        goto LABEL_17;
      goto LABEL_63;
    }
    if ( v26 != 5 )
      goto LABEL_16;
    if ( *((_WORD *)v25 + 1) == 34 )
      goto LABEL_57;
    v27 = (_BYTE *)v12[3];
    v23 = v12 + 3;
    v28 = *v27;
    if ( *v27 > 0x1Cu )
    {
LABEL_17:
      if ( v28 == 63 )
        goto LABEL_57;
      goto LABEL_18;
    }
LABEL_63:
    if ( v28 == 5 && *((_WORD *)v27 + 1) == 34 )
      goto LABEL_57;
LABEL_18:
    v12 += 4;
    if ( v21 == v12 )
    {
      v68 = v73 - v12;
      goto LABEL_94;
    }
  }
  if ( v24 != 5 || *((_WORD *)v22 + 1) != 34 )
    goto LABEL_14;
LABEL_57:
  if ( v23 != v73 )
  {
    v9 = (__int64 *)*v23;
    if ( *v23 )
      goto LABEL_43;
  }
LABEL_23:
  v30 = 0;
  v31 = 0;
LABEL_53:
  *a1 = v18;
  *((_DWORD *)a1 + 2) = v70;
  a1[2] = v31;
  *((_DWORD *)a1 + 6) = v30;
  return a1;
}
