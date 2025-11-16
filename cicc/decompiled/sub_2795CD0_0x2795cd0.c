// Function: sub_2795CD0
// Address: 0x2795cd0
//
__int64 __fastcall sub_2795CD0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, char a5)
{
  __int64 v6; // rdi
  __int64 v7; // rax
  _QWORD *v8; // rdi
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rdx
  unsigned __int8 *v12; // r14
  __int64 v13; // r13
  __int64 v14; // r12
  int v15; // eax
  unsigned __int8 v16; // dl
  int v17; // r10d
  unsigned int v18; // eax
  __int64 v19; // r9
  unsigned __int8 *v20; // rcx
  unsigned __int8 v21; // dl
  __int64 v22; // rax
  int v23; // ecx
  __int64 v24; // rsi
  __int64 v25; // rdi
  __int64 v26; // rcx
  unsigned int v27; // edx
  __int64 *v28; // rax
  __int64 v29; // r8
  __int64 v30; // rdi
  char v31; // al
  __int64 v32; // rax
  __int64 v33; // rdx
  int v34; // eax
  __int64 v35; // rdi
  unsigned int v36; // eax
  __int64 v37; // r8
  __int64 v38; // rdx
  unsigned int v39; // r12d
  int v40; // eax
  bool v41; // si
  __int64 v42; // rcx
  unsigned __int8 v43; // r12
  unsigned __int8 v44; // al
  __int64 v45; // r13
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // r12
  unsigned int v49; // r13d
  unsigned __int8 v51; // dl
  unsigned __int8 *v52; // rax
  unsigned __int8 *v53; // rax
  unsigned __int8 *v54; // rax
  __int64 v55; // rdi
  bool v56; // al
  bool v57; // zf
  unsigned __int8 *v58; // rax
  __int64 v59; // rdi
  bool v60; // al
  _BYTE *v61; // rdi
  unsigned int v62; // eax
  __int64 v63; // r9
  _BYTE *v64; // rax
  __int64 v65; // r14
  __int64 v66; // rdx
  int v67; // eax
  __int64 v68; // rdi
  __int64 *v69; // r10
  __int64 v70; // rax
  unsigned __int64 v71; // rdx
  __int64 *v72; // rax
  __int64 v73; // rax
  __int64 *v74; // rax
  __int64 v75; // rax
  unsigned __int64 v76; // rdx
  __int64 *v77; // rax
  int v78; // eax
  __int64 *v79; // rcx
  _BYTE *v80; // rdi
  bool v81; // al
  unsigned int v82; // [rsp+0h] [rbp-C0h]
  __int64 v83; // [rsp+0h] [rbp-C0h]
  __int64 v84; // [rsp+0h] [rbp-C0h]
  __int64 v85; // [rsp+0h] [rbp-C0h]
  unsigned int v86; // [rsp+10h] [rbp-B0h]
  int v87; // [rsp+10h] [rbp-B0h]
  unsigned __int8 v88; // [rsp+10h] [rbp-B0h]
  __int64 v89; // [rsp+10h] [rbp-B0h]
  int v90; // [rsp+10h] [rbp-B0h]
  unsigned __int8 v91; // [rsp+10h] [rbp-B0h]
  __int64 v92; // [rsp+10h] [rbp-B0h]
  unsigned __int8 v93; // [rsp+10h] [rbp-B0h]
  __int64 v94; // [rsp+10h] [rbp-B0h]
  __int64 v95; // [rsp+10h] [rbp-B0h]
  __int64 v96; // [rsp+10h] [rbp-B0h]
  __int64 v97; // [rsp+18h] [rbp-A8h]
  unsigned int v98; // [rsp+20h] [rbp-A0h]
  unsigned __int8 v99; // [rsp+20h] [rbp-A0h]
  unsigned __int8 v100; // [rsp+26h] [rbp-9Ah]
  __int64 v103; // [rsp+38h] [rbp-88h] BYREF
  _QWORD *v104; // [rsp+40h] [rbp-80h] BYREF
  __int64 v105; // [rsp+48h] [rbp-78h]
  _QWORD v106[14]; // [rsp+50h] [rbp-70h] BYREF

  v6 = a4[1];
  v104 = v106;
  v106[0] = a2;
  v106[1] = a3;
  v105 = 0x400000001LL;
  v7 = sub_AA54C0(v6);
  v100 = 0;
  v8 = v106;
  v97 = v7;
  v9 = 1;
  do
  {
    while ( 1 )
    {
      v10 = v9--;
      v11 = (__int64)&v8[2 * v10 - 2];
      v12 = *(unsigned __int8 **)v11;
      v13 = *(_QWORD *)(v11 + 8);
      LODWORD(v105) = v9;
      if ( v12 != (unsigned __int8 *)v13 )
        break;
LABEL_40:
      if ( !v9 )
        goto LABEL_41;
    }
    if ( *v12 <= 0x15u )
    {
      v51 = *(_BYTE *)v13;
      if ( *(_BYTE *)v13 <= 0x15u )
        goto LABEL_40;
    }
    else
    {
      if ( *v12 != 22 )
        goto LABEL_5;
      v51 = *(_BYTE *)v13;
      if ( *(_BYTE *)v13 <= 0x15u )
      {
        v53 = (unsigned __int8 *)v13;
        v13 = (__int64)v12;
        v12 = v53;
        goto LABEL_57;
      }
    }
    if ( v51 != 22 )
    {
      v52 = v12;
      v12 = (unsigned __int8 *)v13;
      v13 = (__int64)v52;
LABEL_5:
      v14 = sub_B43CC0((__int64)v12);
      goto LABEL_6;
    }
LABEL_57:
    v14 = sub_B2BEC0(*(_QWORD *)(v13 + 24));
    v54 = v12;
    v12 = (unsigned __int8 *)v13;
    v13 = (__int64)v54;
LABEL_6:
    v15 = sub_2792F80(a1 + 136, (__int64)v12);
    v16 = *(_BYTE *)v13;
    v17 = v15;
    if ( *v12 == 22 )
    {
      if ( v16 != 22 )
      {
LABEL_48:
        if ( v16 > 0x1Cu )
          goto LABEL_21;
LABEL_49:
        if ( !v97 )
          goto LABEL_21;
        goto LABEL_50;
      }
    }
    else
    {
      if ( *v12 <= 0x1Cu )
        goto LABEL_48;
      if ( v16 <= 0x1Cu )
        goto LABEL_49;
    }
    v86 = v15;
    v18 = sub_2792F80(a1 + 136, v13);
    v17 = v86;
    if ( v86 >= v18 )
    {
      v21 = *(_BYTE *)v13;
    }
    else
    {
      v20 = v12;
      v21 = *v12;
      v17 = v18;
      v12 = (unsigned __int8 *)v13;
      v13 = (__int64)v20;
    }
    if ( v21 <= 0x1Cu )
      goto LABEL_49;
    if ( *v12 <= 0x1Cu )
      goto LABEL_21;
    v22 = *(_QWORD *)(a1 + 112);
    if ( !v22 )
      goto LABEL_21;
    v23 = *(_DWORD *)(v22 + 24);
    v24 = *((_QWORD *)v12 + 5);
    v25 = *(_QWORD *)(v22 + 8);
    if ( !v23 )
      goto LABEL_21;
    v26 = (unsigned int)(v23 - 1);
    v27 = v26 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
    v28 = (__int64 *)(v25 + 16LL * v27);
    v29 = *v28;
    if ( v24 != *v28 )
    {
      v78 = 1;
      while ( v29 != -4096 )
      {
        v19 = (unsigned int)(v78 + 1);
        v27 = v26 & (v78 + v27);
        v28 = (__int64 *)(v25 + 16LL * v27);
        v29 = *v28;
        if ( v24 == *v28 )
          goto LABEL_16;
        v78 = v19;
      }
LABEL_21:
      v32 = *((_QWORD *)v12 + 2);
      if ( v32 )
        goto LABEL_52;
      goto LABEL_22;
    }
LABEL_16:
    v30 = v28[1];
    if ( !v30 )
      goto LABEL_21;
    v87 = v17;
    v31 = sub_D4A290(v30, "llvm.loop.unroll.full", 0x15u, v26, v29, v19);
    v17 = v87;
    if ( v31 )
    {
      v58 = v12;
      v12 = (unsigned __int8 *)v13;
      v13 = (__int64)v58;
    }
    if ( !v97 || *(_BYTE *)v13 > 0x1Cu )
      goto LABEL_21;
LABEL_50:
    v90 = v17;
    if ( !sub_D31FB0(v12, (unsigned __int8 *)v13, v14) )
      goto LABEL_21;
    sub_27915B0(a1 + 352, v90, v13, a4[1]);
    v32 = *((_QWORD *)v12 + 2);
    if ( v32 )
    {
LABEL_52:
      if ( !*(_QWORD *)(v32 + 8) )
        goto LABEL_27;
    }
LABEL_22:
    v103 = v14;
    v33 = *(_QWORD *)(a1 + 24);
    if ( a5 )
      v34 = sub_F57430(
              (__int64)v12,
              v13,
              v33,
              a4,
              (unsigned __int8 (__fastcall *)(__int64, __int64, __int64))sub_2789520,
              (__int64)&v103);
    else
      v34 = sub_F57550(
              (__int64)v12,
              v13,
              v33,
              *a4,
              (unsigned __int8 (__fastcall *)(__int64, __int64, __int64))sub_2789520,
              (__int64)&v103);
    if ( v34 )
    {
      v35 = *(_QWORD *)(a1 + 16);
      v100 = 1;
      if ( v35 )
        sub_102B9D0(v35, (__int64)v12);
    }
LABEL_27:
    LOBYTE(v36) = sub_BCAC40(*(_QWORD *)(v13 + 8), 1);
    v38 = v36;
    if ( !(_BYTE)v36 || *(_BYTE *)v13 != 17 )
      goto LABEL_39;
    v39 = *(_DWORD *)(v13 + 32);
    if ( v39 )
    {
      if ( v39 <= 0x40 )
      {
        v41 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v39) == *(_QWORD *)(v13 + 24);
      }
      else
      {
        v88 = v36;
        v40 = sub_C445E0(v13 + 24);
        v38 = v88;
        v41 = v39 == v40;
      }
      v42 = *v12;
      v43 = !v41;
      v44 = *v12;
      if ( !v41 )
        goto LABEL_33;
    }
    else
    {
      v42 = *v12;
      v43 = 0;
    }
    v44 = v42;
    if ( (unsigned __int8)v42 <= 0x1Cu )
      goto LABEL_34;
    v55 = *((_QWORD *)v12 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v55 + 8) - 17 <= 1 )
      v55 = **(_QWORD **)(v55 + 16);
    v91 = v38;
    v56 = sub_BCAC40(v55, 1);
    v38 = v91;
    v57 = !v56;
    v44 = *v12;
    if ( !v57 )
    {
      if ( v44 == 57 )
      {
        if ( (v12[7] & 0x40) != 0 )
          v79 = (__int64 *)*((_QWORD *)v12 - 1);
        else
          v79 = (__int64 *)&v12[-32 * (*((_DWORD *)v12 + 1) & 0x7FFFFFF)];
        v63 = *v79;
        if ( *v79 )
        {
          v37 = v79[4];
          if ( v37 )
            goto LABEL_90;
        }
LABEL_67:
        if ( !v43 )
        {
          v9 = v105;
          v8 = v104;
          goto LABEL_40;
        }
        goto LABEL_70;
      }
      if ( v44 == 86 )
      {
        v92 = *((_QWORD *)v12 - 12);
        if ( *(_QWORD *)(v92 + 8) != *((_QWORD *)v12 + 1) )
          goto LABEL_67;
        v80 = (_BYTE *)*((_QWORD *)v12 - 4);
        if ( *v80 > 0x15u )
          goto LABEL_67;
        v99 = v38;
        v84 = *((_QWORD *)v12 - 8);
        v81 = sub_AC30F0((__int64)v80);
        v37 = v84;
        v38 = v99;
        v63 = v92;
        if ( v81 && v84 )
          goto LABEL_90;
        v44 = *v12;
      }
    }
LABEL_33:
    if ( !v43 || v44 <= 0x1Cu )
      goto LABEL_34;
LABEL_70:
    v59 = *((_QWORD *)v12 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v59 + 8) - 17 <= 1 )
      v59 = **(_QWORD **)(v59 + 16);
    v93 = v38;
    v60 = sub_BCAC40(v59, 1);
    v38 = v93;
    v43 = v60;
    v44 = *v12;
    if ( !v43 )
      goto LABEL_98;
    if ( v44 == 58 )
    {
      if ( (v12[7] & 0x40) != 0 )
        v69 = (__int64 *)*((_QWORD *)v12 - 1);
      else
        v69 = (__int64 *)&v12[-32 * (*((_DWORD *)v12 + 1) & 0x7FFFFFF)];
      v63 = *v69;
      if ( !*v69 )
        goto LABEL_39;
      v37 = v69[4];
      if ( !v37 )
        goto LABEL_39;
    }
    else
    {
      if ( v44 != 86 )
        goto LABEL_34;
      v94 = *((_QWORD *)v12 - 12);
      if ( *(_QWORD *)(v94 + 8) != *((_QWORD *)v12 + 1) )
        goto LABEL_39;
      v61 = (_BYTE *)*((_QWORD *)v12 - 8);
      if ( *v61 > 0x15u )
        goto LABEL_39;
      v83 = *((_QWORD *)v12 - 4);
      LOBYTE(v62) = sub_AD7A80(v61, 1, v38, v42, v83);
      v37 = v83;
      v63 = v94;
      v38 = v62;
      if ( !(_BYTE)v62 )
      {
        v44 = *v12;
LABEL_34:
        if ( (unsigned __int8)(v44 - 82) <= 1u )
        {
          v45 = *((_QWORD *)v12 - 8);
          v89 = *((_QWORD *)v12 - 4);
          if ( (unsigned __int8)sub_B52A20((__int64)v12, v43, v38, v42, v37) )
          {
            v75 = (unsigned int)v105;
            v76 = (unsigned int)v105 + 1LL;
            if ( v76 > HIDWORD(v105) )
            {
              sub_C8D5F0((__int64)&v104, v106, v76, 0x10u, v46, v47);
              v75 = (unsigned int)v105;
            }
            v77 = &v104[2 * v75];
            *v77 = v45;
            v77[1] = v89;
            LODWORD(v105) = v105 + 1;
          }
          v98 = sub_B52870(*((_WORD *)v12 + 1) & 0x3F);
          v48 = sub_AD64C0(*((_QWORD *)v12 + 1), v43, 0);
          v82 = *(_DWORD *)(a1 + 344);
          v49 = sub_2795C70(a1 + 136, (unsigned int)*v12 - 29, v98, v45, v89);
          if ( v49 < v82 )
          {
            v64 = sub_278BCD0(a1, a4[1], v49);
            v65 = (__int64)v64;
            if ( v64 )
            {
              if ( *v64 > 0x1Cu )
              {
                v66 = *(_QWORD *)(a1 + 24);
                v67 = a5 ? sub_F57230((__int64)v64, v48, v66, a4) : sub_F57330((__int64)v64, v48, v66, *a4);
                v68 = *(_QWORD *)(a1 + 16);
                v100 |= v67 != 0;
                if ( v68 )
                  sub_102B9D0(v68, v65);
              }
            }
          }
          if ( v97 )
            sub_27915B0(a1 + 352, v49, v48, a4[1]);
        }
LABEL_39:
        v9 = v105;
        v8 = v104;
        goto LABEL_40;
      }
      if ( !v83 )
      {
        v44 = *v12;
LABEL_98:
        v43 = v38;
        goto LABEL_34;
      }
    }
LABEL_90:
    v70 = (unsigned int)v105;
    v71 = (unsigned int)v105 + 1LL;
    if ( v71 > HIDWORD(v105) )
    {
      v85 = v63;
      v96 = v37;
      sub_C8D5F0((__int64)&v104, v106, v71, 0x10u, v37, v63);
      v70 = (unsigned int)v105;
      v63 = v85;
      v37 = v96;
    }
    v72 = &v104[2 * v70];
    *v72 = v63;
    v72[1] = v13;
    LODWORD(v105) = v105 + 1;
    v73 = (unsigned int)v105;
    if ( (unsigned __int64)(unsigned int)v105 + 1 > HIDWORD(v105) )
    {
      v95 = v37;
      sub_C8D5F0((__int64)&v104, v106, (unsigned int)v105 + 1LL, 0x10u, v37, v63);
      v73 = (unsigned int)v105;
      v37 = v95;
    }
    v74 = &v104[2 * v73];
    *v74 = v37;
    v8 = v104;
    v74[1] = v13;
    v9 = v105 + 1;
    LODWORD(v105) = v9;
  }
  while ( v9 );
LABEL_41:
  if ( v8 != v106 )
    _libc_free((unsigned __int64)v8);
  return v100;
}
