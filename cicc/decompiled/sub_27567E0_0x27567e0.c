// Function: sub_27567E0
// Address: 0x27567e0
//
__int64 __fastcall sub_27567E0(
        __int64 *a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        __int64 *a7)
{
  __int64 v7; // r14
  __int64 v11; // rdx
  unsigned __int8 *v12; // r15
  unsigned __int8 *v13; // rax
  unsigned __int64 v14; // r11
  __int64 v15; // rax
  int v16; // eax
  unsigned __int64 v17; // r11
  int v18; // ebx
  char v20; // dl
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // rax
  char v24; // al
  unsigned __int8 v25; // al
  char v26; // al
  _QWORD *v27; // rdi
  __int64 v28; // rax
  char v29; // cl
  __int64 v30; // rax
  int v31; // eax
  __int64 v32; // r12
  __int64 v33; // r13
  __int64 v34; // r15
  int v35; // eax
  unsigned __int8 *v36; // r12
  unsigned __int8 *v37; // rax
  __int64 v38; // rdi
  unsigned __int64 v39; // rbx
  unsigned __int64 v40; // rax
  unsigned __int8 *v41; // rax
  __int64 v42; // r12
  unsigned __int8 *v43; // rbx
  unsigned __int64 *v44; // r10
  unsigned __int8 *v45; // rax
  unsigned __int64 v46; // rdi
  unsigned __int8 *v47; // r12
  __int64 v48; // rdx
  __int64 v49; // rbx
  __int64 v50; // r12
  unsigned __int64 v51; // rbx
  unsigned __int64 v52; // rbx
  unsigned __int64 v53; // rbx
  unsigned __int64 v54; // rax
  unsigned __int64 v55; // [rsp+0h] [rbp-E0h]
  unsigned __int8 *v56; // [rsp+8h] [rbp-D8h]
  __int64 v57; // [rsp+8h] [rbp-D8h]
  __int64 v60; // [rsp+20h] [rbp-C0h]
  __int64 v61; // [rsp+20h] [rbp-C0h]
  unsigned __int8 *v62; // [rsp+28h] [rbp-B8h]
  __int64 *v63; // [rsp+30h] [rbp-B0h]
  unsigned __int8 *v64; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v65; // [rsp+38h] [rbp-A8h] BYREF
  signed __int64 v66; // [rsp+40h] [rbp-A0h] BYREF
  bool v67; // [rsp+48h] [rbp-98h]
  __int64 v68; // [rsp+50h] [rbp-90h] BYREF
  __int64 v69; // [rsp+58h] [rbp-88h]
  __int64 v70; // [rsp+60h] [rbp-80h]
  __int64 v71; // [rsp+68h] [rbp-78h]
  __int64 v72; // [rsp+70h] [rbp-70h]
  __int64 v73; // [rsp+78h] [rbp-68h]
  unsigned __int64 v74; // [rsp+80h] [rbp-60h] BYREF
  __int64 v75; // [rsp+88h] [rbp-58h]
  __int64 v76; // [rsp+90h] [rbp-50h]
  __int64 v77; // [rsp+98h] [rbp-48h]
  __int64 v78; // [rsp+A0h] [rbp-40h]
  __int64 v79; // [rsp+A8h] [rbp-38h]

  v7 = a2;
  v65 = *(_QWORD *)(a4 + 8);
  if ( (unsigned __int8)(*(_BYTE *)a2 - 34) <= 0x33u
    && (v11 = 0x8000000000041LL, _bittest64(&v11, (unsigned int)*(unsigned __int8 *)a2 - 34))
    && ((v63 = (__int64 *)a1[101], !(unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 23))
     && !(unsigned __int8)sub_B49560(a2, 23)
     || (unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 4)
     || (a2 = 4, (unsigned __int8)sub_B49560(v7, 4)))
    && (a2 = *(_QWORD *)(v7 - 32)) != 0
    && !*(_BYTE *)a2
    && *(_QWORD *)(a2 + 24) == *(_QWORD *)(v7 + 80)
    && sub_981210(*v63, a2, (unsigned int *)&v74)
    && (v27 = (_QWORD *)a1[101], a2 = (unsigned __int64)(unsigned int)v74 >> 6, (v27[a2 + 1] & (1LL << v74)) == 0)
    && (((int)*(unsigned __int8 *)(*v27 + ((unsigned int)v74 >> 2)) >> (2 * (v74 & 3))) & 3) != 0
    && ((_DWORD)v74 == 124 || (_DWORD)v74 == 121)
    && (v28 = *(_QWORD *)(v7 + 32 * (2LL - (*(_DWORD *)(v7 + 4) & 0x7FFFFFF))), *(_BYTE *)v28 == 17) )
  {
    if ( *(_DWORD *)(v28 + 32) <= 0x40u )
      v65 = *(_QWORD *)(v28 + 24);
    else
      v65 = **(_QWORD **)(v28 + 24);
    if ( v65 > 0x3FFFFFFFFFFFFFFBLL )
    {
      v65 = 0xBFFFFFFFFFFFFFFELL;
      v60 = 0xBFFFFFFFFFFFFFFELL;
    }
    else
    {
      v60 = v65;
    }
  }
  else
  {
    v60 = v65;
  }
  v62 = sub_BD3990(*(unsigned __int8 **)a5, a2);
  v12 = sub_BD3990(*(unsigned __int8 **)a4, a2);
  v64 = sub_98ACB0(v62, 6u);
  v13 = sub_98ACB0(v12, 6u);
  v14 = (unsigned __int64)v13;
  if ( v64 == v13 )
  {
    v56 = v13;
    if ( v60 < 0 )
    {
LABEL_20:
      v20 = *(_BYTE *)v7;
      if ( *(_BYTE *)v7 != 85 )
      {
        if ( *(_BYTE *)a3 != 85 )
          return 6;
        v21 = *(_QWORD *)(a3 - 32);
        if ( !v21 )
          return 6;
        v22 = 0;
LABEL_24:
        if ( !*(_BYTE *)v21 && *(_QWORD *)(v21 + 24) == *(_QWORD *)(a3 + 80) && (*(_BYTE *)(v21 + 33) & 0x20) != 0 )
        {
          if ( (unsigned int)(*(_DWORD *)(v21 + 36) - 238) > 7 || ((1LL << (*(_BYTE *)(v21 + 36) + 18)) & 0xAD) == 0 )
          {
            if ( v20 == 85 )
            {
              v23 = *(_QWORD *)(v7 - 32);
              if ( v23 )
                goto LABEL_101;
            }
            goto LABEL_121;
          }
          if ( v22
            && *(_QWORD *)(v22 + 32 * (2LL - (*(_DWORD *)(v22 + 4) & 0x7FFFFFF))) == *(_QWORD *)(a3
                                                                                               + 32
                                                                                               * (2LL
                                                                                                - (*(_DWORD *)(a3 + 4)
                                                                                                 & 0x7FFFFFF))) )
          {
            v22 = a5;
            if ( (unsigned __int8)sub_CF4D50(a1[13], a5, a4, (__int64)(a1 + 14), 0) == 3 )
              return 1;
            v20 = *(_BYTE *)a3;
            if ( *(_BYTE *)v7 != 85 )
              goto LABEL_32;
            v23 = *(_QWORD *)(v7 - 32);
            if ( !v23 )
              goto LABEL_32;
LABEL_101:
            if ( *(_BYTE *)v23 )
              goto LABEL_32;
            goto LABEL_31;
          }
        }
        if ( v20 != 85 )
        {
          v7 = 0;
LABEL_62:
          if ( !*(_BYTE *)v21
            && *(_QWORD *)(v21 + 24) == *(_QWORD *)(a3 + 80)
            && (*(_BYTE *)(v21 + 33) & 0x20) != 0
            && v7 )
          {
            v30 = *(_QWORD *)(v7 - 32);
            if ( !v30 || *(_BYTE *)v30 || *(_QWORD *)(v30 + 24) != *(_QWORD *)(v7 + 80) )
              BUG();
            v31 = *(_DWORD *)(v30 + 36);
            if ( *(_DWORD *)(v21 + 36) == v31 && v31 == 230 )
            {
              v32 = *(_DWORD *)(v7 + 4) & 0x7FFFFFF;
              v33 = *(_QWORD *)(*(_QWORD *)(v7 - 32 * v32) + 8LL);
              v34 = *(_QWORD *)(*(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)) + 8LL);
              LODWORD(v65) = sub_BCB060(v33);
              v35 = sub_BCB060(v34);
              if ( (_DWORD)v65 == v35
                && (*(_BYTE *)(v34 + 8) == 18) == (*(_BYTE *)(v33 + 8) == 18)
                && *(_DWORD *)(v34 + 32) == *(_DWORD *)(v33 + 32) )
              {
                v36 = sub_BD3990(*(unsigned __int8 **)(v7 + 32 * (1 - v32)), v22);
                v37 = sub_BD3990(*(unsigned __int8 **)(a3 + 32 * (1LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))), v22);
                if ( v36 == v37 )
                  goto LABEL_76;
                v74 = (unsigned __int64)v37;
                v75 = 1;
                v38 = a1[13];
                v76 = 0;
                v77 = 0;
                v78 = 0;
                v79 = 0;
                v68 = (__int64)v36;
                v69 = 1;
                v70 = 0;
                v71 = 0;
                v72 = 0;
                v73 = 0;
                if ( (unsigned __int8)sub_CF4D50(v38, (__int64)&v68, (__int64)&v74, (__int64)(a1 + 14), 0) == 3 )
                {
LABEL_76:
                  if ( *(_QWORD *)(v7 + 32 * (3LL - (*(_DWORD *)(v7 + 4) & 0x7FFFFFF))) == *(_QWORD *)(a3 + 32 * (3LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))) )
                    return 1;
                }
              }
            }
          }
          return 6;
        }
        v23 = *(_QWORD *)(v7 - 32);
LABEL_28:
        if ( v23 && !*(_BYTE *)v23 )
        {
          v20 = 85;
          goto LABEL_31;
        }
LABEL_121:
        v7 = 0;
LABEL_122:
        if ( !v21 )
          return 6;
        goto LABEL_62;
      }
      v23 = *(_QWORD *)(v7 - 32);
      v22 = v23;
      if ( v23
        && (v22 = 0, !*(_BYTE *)v23)
        && *(_QWORD *)(v23 + 24) == *(_QWORD *)(v7 + 80)
        && (*(_BYTE *)(v23 + 33) & 0x20) != 0
        && (unsigned int)(*(_DWORD *)(v23 + 36) - 238) <= 7 )
      {
        v29 = *(_BYTE *)a3;
        if ( ((1LL << (*(_BYTE *)(v23 + 36) + 18)) & 0xAD) != 0 )
          v22 = v7;
        if ( v29 != 85 )
          goto LABEL_58;
      }
      else
      {
        v29 = *(_BYTE *)a3;
        if ( *(_BYTE *)a3 != 85 )
        {
LABEL_58:
          if ( !v23 || *(_BYTE *)v23 )
            return 6;
          v20 = v29;
LABEL_31:
          v22 = *(_QWORD *)(v7 + 80);
          if ( *(_QWORD *)(v23 + 24) == v22 )
          {
            if ( (*(_BYTE *)(v23 + 33) & 0x20) == 0 )
              v7 = 0;
            goto LABEL_33;
          }
LABEL_32:
          v7 = 0;
LABEL_33:
          if ( v20 != 85 )
            return 6;
          v21 = *(_QWORD *)(a3 - 32);
          goto LABEL_122;
        }
      }
      v21 = *(_QWORD *)(a3 - 32);
      if ( !v21 )
        goto LABEL_28;
      goto LABEL_24;
    }
    v24 = sub_CF7060(v64);
    v14 = (unsigned __int64)v56;
    if ( v24 )
    {
      v55 = (unsigned __int64)v56;
      v57 = a1[101];
      v61 = a1[102];
      v25 = sub_B2F070(*a1, 0);
      v26 = sub_D62CA0((__int64)v64, &v74, v61, v57, v25 << 16, 0);
      v14 = v55;
      if ( v26 )
      {
        if ( (v65 & 0x3FFFFFFFFFFFFFFFLL) == v74 && !_bittest64((const signed __int64 *)&v65, 0x3Eu) )
          return 1;
      }
    }
  }
  else if ( v60 < 0 )
  {
    goto LABEL_20;
  }
  v15 = *(_QWORD *)(a5 + 8);
  if ( v15 < 0 )
    goto LABEL_20;
  v66 = v65 & 0x3FFFFFFFFFFFFFFFLL;
  v68 = v15 & 0x3FFFFFFFFFFFFFFFLL;
  v67 = (v65 & 0x4000000000000000LL) != 0;
  LOBYTE(v69) = (v15 & 0x4000000000000000LL) != 0;
  if ( (v15 & 0x4000000000000000LL) != 0 || (v65 & 0x4000000000000000LL) != 0 )
    return 6;
  v65 = v14;
  v16 = sub_CF4D50(a1[13], a4, a5, (__int64)(a1 + 14), 0);
  v17 = v65;
  v18 = v16 >> 9;
  if ( (_BYTE)v16 == 3 )
  {
    v53 = sub_CA1930(&v66);
    v54 = sub_CA1930(&v68);
    v17 = v65;
    if ( v53 >= v54 )
      return 1;
LABEL_81:
    if ( v64 == (unsigned __int8 *)v17 )
      goto LABEL_82;
    return 6;
  }
  if ( (_BYTE)v16 == 2 )
  {
    if ( (v16 & 0x100) != 0 && v18 >= 0 )
    {
      v39 = sub_CA1930(&v68) + v18;
      v40 = sub_CA1930(&v66);
      v17 = v65;
      if ( v39 <= v40 )
        return 1;
    }
    goto LABEL_81;
  }
  if ( v64 != (unsigned __int8 *)v65 )
  {
    if ( !(_BYTE)v16 )
      return 5;
    return 6;
  }
LABEL_82:
  *a7 = 0;
  *a6 = 0;
  v41 = sub_25536C0((__int64)v62, a7, a1[102], 1);
  v42 = a1[102];
  v43 = v41;
  LODWORD(v75) = sub_AE43F0(v42, *((_QWORD *)v12 + 1));
  if ( (unsigned int)v75 > 0x40 )
  {
    v65 = (unsigned __int64)&v74;
    sub_C43690((__int64)&v74, 0, 0);
    v44 = (unsigned __int64 *)v65;
  }
  else
  {
    v74 = 0;
    v44 = &v74;
  }
  v45 = sub_BD45C0(v12, v42, (__int64)v44, 1, 0, 0, 0, 0);
  v46 = v74;
  v47 = v45;
  if ( (unsigned int)v75 > 0x40 )
  {
    *a6 = *(_QWORD *)v74;
    j_j___libc_free_0_0(v46);
  }
  else
  {
    v48 = 0;
    if ( (_DWORD)v75 )
      v48 = (__int64)(v74 << (64 - (unsigned __int8)v75)) >> (64 - (unsigned __int8)v75);
    *a6 = v48;
  }
  if ( v43 != v47 )
    return 6;
  v49 = *a7;
  v50 = *a6;
  if ( *a7 >= *a6 )
  {
    v51 = sub_CA1930(&v68) + v49 - v50;
    if ( v51 > sub_CA1930(&v66) )
    {
      v52 = *a7 - *a6;
      if ( v52 >= sub_CA1930(&v66) )
        return 5;
      return 4;
    }
    return 1;
  }
  if ( v50 - v49 >= (unsigned __int64)sub_CA1930(&v68) )
    return 5;
  return 4;
}
