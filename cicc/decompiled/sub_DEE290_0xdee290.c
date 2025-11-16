// Function: sub_DEE290
// Address: 0xdee290
//
__int64 __fastcall sub_DEE290(
        __int64 a1,
        __int64 *a2,
        char *a3,
        __int64 a4,
        __int64 **a5,
        __int64 **a6,
        unsigned __int8 a7,
        unsigned __int8 a8)
{
  __int64 **v11; // rax
  __int64 **v12; // r8
  unsigned __int8 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 **v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 **v22; // r10
  __int16 v23; // ax
  __int64 v24; // rax
  char v25; // al
  __int64 v26; // rcx
  int v27; // eax
  int v28; // edi
  __int64 v29; // rdx
  _QWORD *v30; // rax
  char **v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v42; // rax
  __int64 **v43; // r9
  __int64 v44; // rax
  _QWORD *v45; // rax
  __int64 **v46; // rax
  char v47; // al
  bool v48; // al
  unsigned __int8 v49; // al
  __int64 **v50; // rax
  __int64 v51; // rax
  _QWORD *v52; // rax
  __int64 **v53; // rax
  char v54; // al
  _QWORD *v55; // rax
  int v56; // eax
  __int64 v57; // rax
  char v58; // al
  __int64 v59; // rcx
  __int64 v60; // r9
  int v61; // edx
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  int v70; // eax
  __int64 v71; // [rsp-10h] [rbp-C0h]
  __int64 v72; // [rsp-8h] [rbp-B8h]
  __int64 v73; // [rsp+0h] [rbp-B0h]
  _QWORD *v74; // [rsp+0h] [rbp-B0h]
  unsigned __int8 v75; // [rsp+0h] [rbp-B0h]
  int v76; // [rsp+0h] [rbp-B0h]
  char v77; // [rsp+0h] [rbp-B0h]
  unsigned __int8 v78; // [rsp+8h] [rbp-A8h]
  unsigned int v79; // [rsp+8h] [rbp-A8h]
  char v80; // [rsp+8h] [rbp-A8h]
  unsigned int v81; // [rsp+8h] [rbp-A8h]
  unsigned __int8 v82; // [rsp+10h] [rbp-A0h]
  __int64 v83; // [rsp+10h] [rbp-A0h]
  __int64 v84; // [rsp+10h] [rbp-A0h]
  __int64 v85; // [rsp+10h] [rbp-A0h]
  unsigned __int8 v86; // [rsp+10h] [rbp-A0h]
  __int64 **v87; // [rsp+10h] [rbp-A0h]
  __int64 **v88; // [rsp+10h] [rbp-A0h]
  __int64 **v89; // [rsp+18h] [rbp-98h] BYREF
  __int64 **v90; // [rsp+20h] [rbp-90h] BYREF
  __int64 v91; // [rsp+28h] [rbp-88h] BYREF
  __int64 *v92; // [rsp+30h] [rbp-80h] BYREF
  __int64 v93; // [rsp+38h] [rbp-78h]
  __int64 v94; // [rsp+40h] [rbp-70h] BYREF
  char v95; // [rsp+48h] [rbp-68h]
  char *v96; // [rsp+50h] [rbp-60h] BYREF
  int v97; // [rsp+58h] [rbp-58h]
  _BYTE v98[80]; // [rsp+60h] [rbp-50h] BYREF

  v91 = a4;
  v89 = a6;
  v90 = (__int64 **)sub_DDF4E0((__int64)a2, a5, a3);
  v89 = (__int64 **)sub_DDF4E0((__int64)a2, v89, a3);
  if ( sub_DADE90((__int64)a2, (__int64)v90, (__int64)a3) && !sub_DADE90((__int64)a2, (__int64)v89, (__int64)a3) )
  {
    v50 = v90;
    v90 = v89;
    v89 = v50;
    LODWORD(v91) = sub_B52F50(v91);
  }
  if ( !a7
    || !(unsigned __int8)sub_DB5FD0((__int64)a2, (__int64)a3)
    || (v49 = sub_DB6630((__int64)a2, (__int64)a3)) == 0 )
  {
    sub_DC7F30(a2, (__int64)&v91, (__int64 *)&v90, (__int64 *)&v89, 0);
    v11 = v89;
    if ( *((_WORD *)v89 + 12) )
      goto LABEL_22;
    v12 = v90;
    if ( *((_WORD *)v90 + 12) != 8 )
      goto LABEL_22;
    v13 = 0;
    if ( v90[6] != (__int64 *)a3 )
      goto LABEL_22;
    goto LABEL_6;
  }
  v86 = v49;
  sub_DC7F30(a2, (__int64)&v91, (__int64 *)&v90, (__int64 *)&v89, 0);
  v17 = v89;
  v13 = v86;
  v11 = v89;
  if ( !*((_WORD *)v89 + 12) )
  {
    v12 = v90;
    if ( *((_WORD *)v90 + 12) == 8 && v90[6] == (__int64 *)a3 )
    {
LABEL_6:
      v82 = v13;
      v73 = (__int64)v12;
      sub_AB1A50((__int64)&v92, v91, (__int64)(v11[4] + 3));
      v74 = sub_DD1050(v73, (__int64)&v92, a2);
      if ( !sub_D96A50((__int64)v74) )
      {
        sub_D97F80(a1, (__int64)v74, v14, v15, v16, v82);
        sub_969240(&v94);
        sub_969240((__int64 *)&v92);
        return a1;
      }
      sub_969240(&v94);
      sub_969240((__int64 *)&v92);
      if ( v82 )
      {
        v17 = v89;
        goto LABEL_9;
      }
LABEL_22:
      v28 = v91;
      LOBYTE(v21) = 0;
      goto LABEL_23;
    }
  }
LABEL_9:
  LOBYTE(v21) = sub_DADE90((__int64)a2, (__int64)v17, (__int64)a3);
  if ( (_BYTE)v21 )
  {
    v22 = v90;
    v23 = *((_WORD *)v90 + 12);
    if ( v23 == 3 )
    {
      v22 = (__int64 **)v90[4];
      v23 = *((_WORD *)v22 + 12);
    }
    if ( v23 == 8 && (*((_BYTE *)v22 + 28) & 1) == 0 && a3 == (char *)v22[6] && v22[5] == (__int64 *)2 )
    {
      v78 = v21;
      v83 = (__int64)v22;
      v24 = sub_D33D80(v22, (__int64)a2, v18, v19, v20);
      v25 = sub_DBE140(a2, v24, 1u, 1);
      v21 = v78;
      if ( v25 )
      {
        v75 = v78;
        v79 = *(_WORD *)(v83 + 28) & 6 | 1;
        sub_D9CAF0(&v92, *(const void **)(v83 + 32), *(_QWORD *)(v83 + 40), v26, v79, v21);
        v27 = sub_DBF900((__int64)a2, 8, v92, (unsigned int)v93, v79);
        sub_D97270((__int64)a2, v83, v27);
        LOBYTE(v21) = v75;
        if ( v92 != &v94 )
        {
          _libc_free(v92, v83);
          LOBYTE(v21) = v75;
        }
      }
    }
    v28 = v91;
    if ( (_DWORD)v91 == 40 )
    {
      v29 = 4;
    }
    else
    {
      if ( (_DWORD)v91 != 36 )
        goto LABEL_23;
      v29 = 2;
    }
    if ( *((_WORD *)v90 + 12) != 8 )
      goto LABEL_39;
    if ( a3 == (char *)v90[6] && v90[5] == (__int64 *)2 )
    {
      v56 = *((unsigned __int16 *)v90 + 14);
      if ( (v56 & (unsigned int)v29) == 0 && (v56 & 1) != 0 )
      {
        v76 = v29;
        v88 = v90;
        v80 = v56 & 1;
        v57 = sub_D33D80(v90, (__int64)a2, v29, v56 & 1, v20);
        v58 = sub_DBEDC0((__int64)a2, v57);
        v61 = v76;
        if ( v58 )
        {
          v77 = v58;
          v81 = v61 | *((_WORD *)v88 + 14) & 7;
          sub_D9CAF0(&v92, v88[4], (__int64)v88[5], v59, v81, v60);
          v70 = sub_DBF900((__int64)a2, 8, v92, (unsigned int)v93, v81);
          sub_D97270((__int64)a2, (__int64)v88, v70);
          LOBYTE(v21) = v77;
          if ( v92 != &v94 )
          {
            _libc_free(v92, v88);
            LOBYTE(v21) = v77;
          }
          v28 = v91;
        }
        else
        {
          v28 = v91;
          LOBYTE(v21) = v80;
        }
      }
    }
  }
  else
  {
    v28 = v91;
    LOBYTE(v21) = 1;
  }
LABEL_23:
  switch ( v28 )
  {
    case ' ':
      if ( *(_BYTE *)(sub_D95540((__int64)v90) + 8) == 14 )
      {
        v90 = (__int64 **)sub_DD3750((__int64)a2, (__int64)v90);
        if ( sub_D96A50((__int64)v90) )
          goto LABEL_74;
      }
      if ( *(_BYTE *)(sub_D95540((__int64)v89) + 8) == 14 )
      {
        v89 = (__int64 **)sub_DD3750((__int64)a2, (__int64)v89);
        if ( sub_D96A50((__int64)v89) )
          goto LABEL_78;
      }
      v55 = sub_DCC810(a2, (__int64)v90, (__int64)v89, 0, 0);
      v31 = (char **)a2;
      sub_DA2C90((__int64)&v92, (__int64)a2, (__int64)v55);
      if ( sub_D96A50((__int64)v92) )
        goto LABEL_27;
      goto LABEL_41;
    case '!':
      if ( *(_BYTE *)(sub_D95540((__int64)v90) + 8) == 14 )
      {
        v90 = (__int64 **)sub_DD3750((__int64)a2, (__int64)v90);
        if ( sub_D96A50((__int64)v90) )
        {
LABEL_74:
          sub_D97F80(a1, (__int64)v90, v62, v63, v64, v65);
          return a1;
        }
      }
      if ( *(_BYTE *)(sub_D95540((__int64)v89) + 8) == 14 )
      {
        v89 = (__int64 **)sub_DD3750((__int64)a2, (__int64)v89);
        if ( sub_D96A50((__int64)v89) )
        {
LABEL_78:
          sub_D97F80(a1, (__int64)v89, v66, v67, v68, v69);
          return a1;
        }
      }
      v30 = sub_DCC810(a2, (__int64)v90, (__int64)v89, 0, 0);
      v31 = (char **)a2;
      sub_DEC310((__int64)&v92, a2, (__int64)v30, (char **)a3, a7, a8);
      if ( !sub_D96A50((__int64)v92) )
        goto LABEL_41;
      goto LABEL_27;
    case '"':
    case '&':
      goto LABEL_57;
    case '#':
    case '\'':
      if ( !(_BYTE)qword_4F88A48 || !(_BYTE)v21 || !sub_DADE90((__int64)a2, (__int64)v89, (__int64)a3) )
        goto LABEL_30;
      v87 = v89;
      v51 = sub_D95540((__int64)v89);
      v52 = sub_DA2C50((__int64)a2, v51, -1, 1u);
      v53 = (__int64 **)sub_DC7ED0(a2, (__int64)v52, (__int64)v87, 0, 0);
      v28 = v91;
      v89 = v53;
LABEL_57:
      v54 = sub_B532B0(v28);
      v31 = (char **)a2;
      sub_DEDA50((__int64)&v92, a2, (__int64)v90, (__int64)v89, (__int64)a3, v54, a7, a8);
      goto LABEL_40;
    case '$':
    case '(':
      break;
    case '%':
    case ')':
      if ( (_BYTE)qword_4F88A48 && (_BYTE)v21 && sub_DADE90((__int64)a2, (__int64)v89, (__int64)a3) )
      {
        v43 = v89;
      }
      else
      {
        v42 = sub_D95540((__int64)v90);
        if ( *(_BYTE *)(v42 + 8) != 12 )
          goto LABEL_30;
        v84 = sub_BCD140(*(_QWORD **)v42, 2 * (*(_DWORD *)(v42 + 8) >> 8));
        if ( sub_B532B0(v91) )
        {
          v90 = (__int64 **)sub_DC5000((__int64)a2, (__int64)v90, v84, 0);
          v89 = (__int64 **)sub_DC5000((__int64)a2, (__int64)v89, v84, 0);
        }
        else
        {
          v90 = (__int64 **)sub_DC2B70((__int64)a2, (__int64)v90, v84, 0);
          v89 = (__int64 **)sub_DC2B70((__int64)a2, (__int64)v89, v84, 0);
        }
        v43 = v89;
      }
      v85 = (__int64)v43;
      v44 = sub_D95540((__int64)v43);
      v45 = sub_DA2C50((__int64)a2, v44, 1, 0);
      v46 = (__int64 **)sub_DC7ED0(a2, (__int64)v45, v85, 0, 0);
      v28 = v91;
      v89 = v46;
      break;
    default:
      goto LABEL_30;
  }
LABEL_39:
  v47 = sub_B532B0(v28);
  v31 = (char **)a2;
  sub_DEACA0((__int64)&v92, (__int64)a2, (__int64)v90, (__int64)v89, (__int64)a3, v47, a7, a8);
LABEL_40:
  v48 = sub_D96A50((__int64)v92);
  v32 = v71;
  v33 = v72;
  if ( !v48 )
    goto LABEL_41;
LABEL_27:
  if ( !sub_D96A50(v93) )
  {
LABEL_41:
    *(_QWORD *)a1 = v92;
    *(_QWORD *)(a1 + 8) = v93;
    *(_QWORD *)(a1 + 16) = v94;
    *(_BYTE *)(a1 + 24) = v95;
    *(_QWORD *)(a1 + 32) = a1 + 48;
    *(_QWORD *)(a1 + 40) = 0x400000000LL;
    if ( v97 )
    {
      v31 = &v96;
      sub_D91460(a1 + 32, &v96, v32, v33, v34, v35);
    }
    if ( v96 != v98 )
      _libc_free(v96, v31);
    return a1;
  }
  if ( v96 != v98 )
    _libc_free(v96, v31);
LABEL_30:
  v36 = sub_D970F0((__int64)a2);
  sub_D97F80(a1, v36, v37, v38, v39, v40);
  return a1;
}
