// Function: sub_17594C0
// Address: 0x17594c0
//
_QWORD *__fastcall sub_17594C0(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4, double a5, double a6, double a7)
{
  int v7; // r12d
  __int64 *v8; // r14
  unsigned __int8 v10; // al
  bool v11; // al
  unsigned int v12; // r13d
  unsigned int v13; // ecx
  unsigned int v14; // edx
  unsigned int v15; // ecx
  int v16; // eax
  _QWORD *v17; // rbx
  bool v19; // al
  unsigned int v20; // r13d
  int v21; // eax
  bool v22; // al
  __int64 v23; // rax
  __int64 v24; // r12
  _QWORD *v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r12
  _QWORD *v29; // rax
  __int64 v30; // rax
  __int64 v31; // r13
  _QWORD *v32; // rax
  __int64 v33; // rax
  __int64 v34; // r12
  _QWORD *v35; // rax
  __int64 v36; // rax
  __int64 v37; // r12
  _QWORD *v38; // rax
  unsigned int v39; // eax
  bool v40; // al
  unsigned int v41; // eax
  _QWORD *v42; // r12
  __int64 v43; // rbx
  unsigned int v44; // eax
  __int64 v45; // rax
  __int64 v46; // rcx
  __int64 v47; // rax
  __int64 v48; // r14
  _QWORD *v49; // rax
  unsigned int v50; // edx
  __int64 v51; // rdx
  __int64 v52; // rbx
  unsigned int v53; // edx
  unsigned __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rcx
  __int64 v57; // rax
  __int64 v58; // r14
  _QWORD *v59; // rax
  unsigned int v60; // ecx
  unsigned int v61; // [rsp+8h] [rbp-E8h]
  unsigned int v62; // [rsp+Ch] [rbp-E4h]
  unsigned int v63; // [rsp+Ch] [rbp-E4h]
  __int64 v64; // [rsp+10h] [rbp-E0h]
  __int64 *v65; // [rsp+20h] [rbp-D0h]
  __int64 *v66; // [rsp+28h] [rbp-C8h]
  unsigned int v68; // [rsp+38h] [rbp-B8h]
  unsigned __int8 *v69; // [rsp+38h] [rbp-B8h]
  unsigned __int8 *v70; // [rsp+38h] [rbp-B8h]
  __int64 *v71; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v72; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v73; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v74; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v75; // [rsp+58h] [rbp-98h]
  __int64 v76; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v77; // [rsp+68h] [rbp-88h]
  __int16 v78; // [rsp+70h] [rbp-80h]
  __int64 v79; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v80; // [rsp+88h] [rbp-68h]
  __int64 v81; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v82; // [rsp+98h] [rbp-58h]
  _QWORD *v83; // [rsp+A0h] [rbp-50h] BYREF
  unsigned int v84; // [rsp+A8h] [rbp-48h]
  __int64 v85; // [rsp+B0h] [rbp-40h]
  unsigned int v86; // [rsp+B8h] [rbp-38h]

  v7 = *(_WORD *)(a2 + 18) & 0x7FFF;
  if ( (unsigned int)(v7 - 32) <= 1 )
    return 0;
  v8 = (__int64 *)*(a3 - 3);
  v10 = *((_BYTE *)v8 + 16);
  if ( v10 != 13 )
  {
    if ( *(_BYTE *)(*v8 + 8) == 16 && v10 <= 0x10u )
    {
      v26 = sub_15A1020(v8, a2, *v8, (__int64)a4);
      if ( v26 )
      {
        if ( *(_BYTE *)(v26 + 16) == 13 )
        {
          v66 = (__int64 *)(v26 + 24);
          v7 = *(_WORD *)(a2 + 18) & 0x7FFF;
          goto LABEL_4;
        }
      }
    }
    return 0;
  }
  v66 = v8 + 3;
LABEL_4:
  v65 = (__int64 *)*(a3 - 6);
  v64 = *a3;
  if ( !sub_15F2380((__int64)a3) || ((v7 - 38) & 0xFFFFFFFD) != 0 )
  {
LABEL_6:
    sub_158B890((__int64)&v83, v7, (__int64)a4);
    sub_158BC30((__int64)&v79, (__int64 *)&v83, (__int64)v66);
    if ( v86 > 0x40 && v85 )
      j_j___libc_free_0_0(v85);
    if ( v84 > 0x40 && v83 )
      j_j___libc_free_0_0(v83);
    v11 = sub_15FF7F0(*(_WORD *)(a2 + 18) & 0x7FFF);
    v12 = v80;
    if ( !v11 )
    {
      if ( v80 <= 0x40 )
        v19 = v79 == 0;
      else
        v19 = v12 == (unsigned int)sub_16A57B0((__int64)&v79);
      if ( v19 )
      {
        v33 = sub_15A1070(v64, (__int64)&v81);
        LOWORD(v85) = 257;
        v34 = v33;
        v35 = sub_1648A60(56, 2u);
        v17 = v35;
        if ( v35 )
          sub_17582E0((__int64)v35, 36, (__int64)v65, v34, (__int64)&v83);
        goto LABEL_41;
      }
      v14 = v82;
      v20 = v82;
      if ( v82 <= 0x40 )
      {
        v22 = v81 == 0;
      }
      else
      {
        v63 = v82;
        v21 = sub_16A57B0((__int64)&v81);
        v14 = v63;
        v22 = v63 == v21;
      }
      if ( v22 )
      {
        v36 = sub_15A1070(v64, (__int64)&v79);
        LOWORD(v85) = 257;
        v37 = v36;
        v38 = sub_1648A60(56, 2u);
        v17 = v38;
        if ( v38 )
          sub_17582E0((__int64)v38, 35, (__int64)v65, v37, (__int64)&v83);
        goto LABEL_41;
      }
      v17 = (_QWORD *)a3[1];
      if ( !v17 )
        goto LABEL_42;
LABEL_32:
      v17 = (_QWORD *)v17[1];
      if ( v17 )
      {
        v20 = v14;
        v17 = 0;
LABEL_42:
        if ( v20 <= 0x40 )
        {
LABEL_45:
          v12 = v80;
LABEL_46:
          if ( v12 > 0x40 && v79 )
            j_j___libc_free_0_0(v79);
          return v17;
        }
LABEL_43:
        if ( v81 )
          j_j___libc_free_0_0(v81);
        goto LABEL_45;
      }
      if ( v7 != 36 )
      {
        if ( v7 != 34 )
        {
LABEL_35:
          v20 = v14;
          goto LABEL_42;
        }
        sub_13A38D0((__int64)&v72, (__int64)a4);
        sub_16A7490((__int64)&v72, 1);
        v39 = v73;
        v73 = 0;
        v75 = v39;
        v74 = v72;
        if ( !sub_14A9C60((__int64)&v74) )
          goto LABEL_69;
        sub_13A38D0((__int64)&v76, (__int64)v66);
        if ( v77 > 0x40 )
        {
          sub_16A8890(&v76, a4);
          v60 = v77;
          v51 = v76;
          v77 = 0;
          v84 = v60;
          v83 = (_QWORD *)v76;
          if ( v60 > 0x40 )
          {
            v71 = (__int64 *)v76;
            if ( v60 - (unsigned int)sub_16A57B0((__int64)&v83) > 0x40 )
              goto LABEL_86;
            v51 = *v71;
          }
        }
        else
        {
          v84 = v77;
          v77 = 0;
          v51 = *a4 & v76;
          v76 = v51;
          v83 = (_QWORD *)v51;
        }
        if ( v51 )
        {
LABEL_86:
          sub_135E100((__int64 *)&v83);
          sub_135E100(&v76);
          sub_135E100((__int64 *)&v74);
          sub_135E100((__int64 *)&v72);
          goto LABEL_41;
        }
        sub_135E100((__int64 *)&v83);
        sub_135E100(&v76);
        sub_135E100((__int64 *)&v74);
        sub_135E100((__int64 *)&v72);
        v52 = *(_QWORD *)(a1 + 8);
        v78 = 257;
        sub_13A38D0((__int64)&v72, (__int64)a4);
        v53 = v73;
        if ( v73 > 0x40 )
        {
          sub_16A8F40((__int64 *)&v72);
          v53 = v73;
          v54 = v72;
        }
        else
        {
          v54 = ~v72 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v73);
          v72 = v54;
        }
        v74 = v54;
        v75 = v53;
        v73 = 0;
        v55 = sub_15A1070(*v65, (__int64)&v74);
        v70 = sub_1729500(v52, (unsigned __int8 *)v65, v55, &v76, a5, a6, a7);
        v57 = sub_15A2B90(v8, 0, 0, v56, a5, a6, a7);
        LOWORD(v85) = 257;
        v58 = v57;
        v59 = sub_1648A60(56, 2u);
        v17 = v59;
        if ( v59 )
          sub_17582E0((__int64)v59, 33, (__int64)v70, v58, (__int64)&v83);
LABEL_69:
        sub_135E100((__int64 *)&v74);
        sub_135E100((__int64 *)&v72);
LABEL_41:
        v20 = v82;
        goto LABEL_42;
      }
      v68 = v14;
      v40 = sub_14A9C60((__int64)a4);
      v14 = v68;
      if ( !v40 )
        goto LABEL_35;
      sub_13A38D0((__int64)&v74, (__int64)a4);
      sub_16A7800((__int64)&v74, 1u);
      v41 = v75;
      v75 = 0;
      v77 = v41;
      v76 = v74;
      if ( v41 > 0x40 )
      {
        sub_16A8890(&v76, v66);
        v50 = v77;
        v42 = (_QWORD *)v76;
        v77 = 0;
        v84 = v50;
        v83 = (_QWORD *)v76;
        if ( v50 > 0x40 )
        {
          if ( v50 - (unsigned int)sub_16A57B0((__int64)&v83) > 0x40 )
            goto LABEL_74;
          v42 = (_QWORD *)*v42;
        }
      }
      else
      {
        v42 = (_QWORD *)(*v66 & v74);
        v84 = v41;
        v76 = (__int64)v42;
        v83 = v42;
        v77 = 0;
      }
      if ( !v42 )
      {
        sub_135E100((__int64 *)&v83);
        sub_135E100(&v76);
        sub_135E100((__int64 *)&v74);
        v78 = 257;
        v43 = *(_QWORD *)(a1 + 8);
        sub_13A38D0((__int64)&v72, (__int64)a4);
        if ( v73 > 0x40 )
          sub_16A8F40((__int64 *)&v72);
        else
          v72 = ~v72 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v73);
        sub_16A7400((__int64)&v72);
        v44 = v73;
        v73 = 0;
        v75 = v44;
        v74 = v72;
        v45 = sub_15A1070(*v65, (__int64)&v74);
        v69 = sub_1729500(v43, (unsigned __int8 *)v65, v45, &v76, a5, a6, a7);
        v47 = sub_15A2B90(v8, 0, 0, v46, a5, a6, a7);
        LOWORD(v85) = 257;
        v48 = v47;
        v49 = sub_1648A60(56, 2u);
        v17 = v49;
        if ( v49 )
          sub_17582E0((__int64)v49, 32, (__int64)v69, v48, (__int64)&v83);
        goto LABEL_69;
      }
LABEL_74:
      sub_135E100((__int64 *)&v83);
      sub_135E100(&v76);
      sub_135E100((__int64 *)&v74);
      v14 = v82;
      goto LABEL_35;
    }
    v13 = v80 - 1;
    if ( v80 <= 0x40 )
    {
      if ( 1LL << v13 != v79 )
        goto LABEL_16;
    }
    else if ( (*(_QWORD *)(v79 + 8LL * (v13 >> 6)) & (1LL << v13)) == 0
           || (unsigned int)sub_16A58A0((__int64)&v79) != v13 )
    {
LABEL_16:
      v14 = v82;
      v15 = v82 - 1;
      if ( v82 <= 0x40 )
      {
        if ( 1LL << v15 != v81 )
        {
          v17 = (_QWORD *)a3[1];
          if ( !v17 )
            goto LABEL_46;
          goto LABEL_32;
        }
      }
      else
      {
        v62 = v82 - 1;
        if ( (*(_QWORD *)(v81 + 8LL * (v15 >> 6)) & (1LL << v15)) == 0
          || (v61 = v82, v16 = sub_16A58A0((__int64)&v81), v14 = v61, v16 != v62) )
        {
          v17 = (_QWORD *)a3[1];
          if ( !v17 )
            goto LABEL_43;
          goto LABEL_32;
        }
      }
      v27 = sub_15A1070(v64, (__int64)&v79);
      LOWORD(v85) = 257;
      v28 = v27;
      v29 = sub_1648A60(56, 2u);
      v17 = v29;
      if ( v29 )
        sub_17582E0((__int64)v29, 39, (__int64)v65, v28, (__int64)&v83);
      goto LABEL_41;
    }
    v23 = sub_15A1070(v64, (__int64)&v81);
    LOWORD(v85) = 257;
    v24 = v23;
    v25 = sub_1648A60(56, 2u);
    v17 = v25;
    if ( v25 )
      sub_17582E0((__int64)v25, 40, (__int64)v65, v24, (__int64)&v83);
    goto LABEL_41;
  }
  sub_16A7620((__int64)&v79, (__int64)a4, (__int64)v66, &v76);
  if ( (_BYTE)v76 )
  {
    sub_135E100(&v79);
    goto LABEL_6;
  }
  v30 = sub_15A1070(v64, (__int64)&v79);
  LOWORD(v85) = 257;
  v31 = v30;
  v32 = sub_1648A60(56, 2u);
  v17 = v32;
  if ( v32 )
    sub_17582E0((__int64)v32, v7, (__int64)v65, v31, (__int64)&v83);
  sub_135E100(&v79);
  return v17;
}
