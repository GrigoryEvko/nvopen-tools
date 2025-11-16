// Function: sub_DDB1C0
// Address: 0xddb1c0
//
char __fastcall sub_DDB1C0(
        __int64 *a1,
        unsigned __int64 a2,
        _BYTE *a3,
        _BYTE *a4,
        unsigned __int64 a5,
        _BYTE *a6,
        _BYTE *a7,
        __int64 a8)
{
  _BYTE *v9; // rcx
  int v10; // ebx
  unsigned int v11; // edi
  _BYTE *v13; // rdx
  char v14; // bl
  unsigned int v15; // eax
  unsigned int v16; // r13d
  int v17; // ebx
  int v18; // eax
  char v19; // r13
  int v20; // eax
  int v21; // r15d
  unsigned __int64 v22; // r13
  _BYTE *v23; // r14
  _BYTE *v24; // rbx
  unsigned __int64 v25; // rsi
  _BYTE *v26; // rbx
  _BYTE *v27; // r14
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  _BYTE *v36; // rbx
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  _BYTE *v40; // rax
  _BYTE *v41; // rbx
  _BYTE *v42; // r14
  _BYTE *v43; // r13
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  _BYTE *v47; // rax
  __int64 v48; // rax
  _BYTE *v49; // rax
  _BYTE *v50; // rax
  _QWORD *v51; // rax
  unsigned __int8 v52; // bl
  unsigned int v53; // eax
  _QWORD *v54; // rax
  unsigned __int8 v55; // bl
  unsigned int v56; // eax
  char v57; // al
  _BYTE *v58; // rax
  _BYTE *v59; // rax
  unsigned __int64 v60; // [rsp+0h] [rbp-A0h]
  _BYTE *v61; // [rsp+0h] [rbp-A0h]
  _BYTE *v62; // [rsp+0h] [rbp-A0h]
  _BYTE *v63; // [rsp+8h] [rbp-98h]
  _BYTE *v64; // [rsp+8h] [rbp-98h]
  _BYTE *v65; // [rsp+8h] [rbp-98h]
  _BYTE *v66; // [rsp+10h] [rbp-90h]
  _BYTE *v67; // [rsp+10h] [rbp-90h]
  _BYTE *v68; // [rsp+10h] [rbp-90h]
  _BYTE *v69; // [rsp+18h] [rbp-88h] BYREF
  unsigned __int64 v70; // [rsp+20h] [rbp-80h] BYREF
  _BYTE *v71; // [rsp+28h] [rbp-78h] BYREF
  _BYTE *v72; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v73; // [rsp+38h] [rbp-68h] BYREF
  __int64 v74; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v75; // [rsp+48h] [rbp-58h]
  __int64 v76; // [rsp+50h] [rbp-50h] BYREF
  int v77; // [rsp+58h] [rbp-48h]
  __int64 v78; // [rsp+60h] [rbp-40h] BYREF
  int v79; // [rsp+68h] [rbp-38h]

  v73 = a2;
  v72 = a3;
  v71 = a4;
  v70 = a5;
  v69 = a6;
  if ( (unsigned __int8)sub_DC7F30(a1, (__int64)&v73, (__int64 *)&v72, (__int64 *)&v71, 0) && v72 == v71 )
    return sub_B535D0(v73);
  if ( (unsigned __int8)sub_DC7F30(a1, (__int64)&v70, (__int64 *)&v69, (__int64 *)&a7, 0) && v69 == a7 )
    return sub_B53600(v70);
  v9 = v72;
  if ( v72 == a7 || v69 == v71 )
  {
    if ( !*((_WORD *)v71 + 12) )
    {
      v13 = v69;
      v14 = BYTE4(v70);
      v69 = a7;
      a7 = v13;
      v15 = sub_B52F50(v70);
      BYTE4(v70) = v14;
      v10 = v73;
      v11 = v15;
      LODWORD(v70) = v15;
      if ( v15 == (_DWORD)v73 )
        return sub_DDB0E0(a1, v73, v72, v71, v69, a7, a8);
      goto LABEL_8;
    }
    v19 = BYTE4(v73);
    v72 = v71;
    v71 = v9;
    v20 = sub_B52F50(v73);
    BYTE4(v73) = v19;
    v11 = v70;
    LODWORD(v73) = v20;
    v10 = v20;
  }
  else
  {
    v10 = v73;
    v11 = v70;
  }
  if ( v11 == v10 )
    return sub_DDB0E0(a1, v73, v72, v71, v69, a7, a8);
LABEL_8:
  if ( (unsigned int)sub_B52F50(v11) != v10 )
  {
    v16 = v70;
    v17 = v73;
    if ( sub_B52830(v70) || v17 != (unsigned int)sub_B53550(v16) )
      goto LABEL_21;
    if ( (!(unsigned __int8)sub_DBED40((__int64)a1, (__int64)v69)
       || !(unsigned __int8)sub_DBED40((__int64)a1, (__int64)a7))
      && (!(unsigned __int8)sub_DBEC00((__int64)a1, (__int64)v69)
       || !(unsigned __int8)sub_DBEC00((__int64)a1, (__int64)a7)) )
    {
      v21 = v73;
      v22 = (unsigned int)v70;
      v23 = v71;
      v60 = v70;
      v24 = v69;
      v66 = v72;
      v63 = a7;
      if ( (v73 & 0xFFFFFFFA) == 0x22 )
      {
        v21 = sub_B52F50(v73);
        v22 = (unsigned int)sub_B52F50(v22);
        v49 = v24;
        v24 = v63;
        v63 = v49;
        v50 = v66;
        v66 = v23;
        v23 = v50;
      }
      if ( sub_B532B0(v21) && (unsigned __int8)sub_DBED40((__int64)a1, (__int64)v23) )
      {
        v25 = v60 & 0xFFFFFFFF00000000LL;
        return sub_DDB0E0(a1, v22 | v25, v66, v23, v24, v63, 0);
      }
      if ( sub_B532A0(v21) && (unsigned __int8)sub_DBEC00((__int64)a1, (__int64)v23) )
      {
        v25 = (unsigned int)v22;
        v22 = v60 & 0xFFFFFFFF00000000LL;
        return sub_DDB0E0(a1, v22 | v25, v66, v23, v24, v63, 0);
      }
LABEL_21:
      v18 = v70;
      if ( (_DWORD)v70 != 33 )
      {
LABEL_22:
        if ( v18 == 32 && (unsigned __int8)sub_B535D0(v73) && sub_DDB0E0(a1, v73, v72, v71, v69, a7, a8) )
          return 1;
LABEL_23:
        if ( (_DWORD)v73 != 33 || (unsigned __int8)sub_B535D0(v70) || !sub_DDB0E0(a1, v70, v72, v71, v69, a7, a8) )
          return sub_DC15B0((__int64)a1, v73, (__int64)v72, (__int64)v71, v70, (__int64)v69, (__int64)a7);
        return 1;
      }
      v26 = v69;
      v27 = a7;
      if ( *((_WORD *)v69 + 12) )
      {
        if ( *((_WORD *)a7 + 12) )
          goto LABEL_23;
        v26 = a7;
        v27 = v69;
      }
      if ( sub_B532B0(v73) )
      {
        v48 = sub_DBB9F0((__int64)a1, (__int64)v27, 1u, 0);
        sub_AB14C0((__int64)&v74, v48);
      }
      else
      {
        v28 = sub_DBB9F0((__int64)a1, (__int64)v27, 0, 0);
        sub_AB0A00((__int64)&v74, v28);
      }
      v29 = *((_QWORD *)v26 + 4);
      if ( v75 <= 0x40 )
      {
        if ( v74 == *(_QWORD *)(v29 + 24) )
        {
LABEL_44:
          sub_9865C0((__int64)&v78, (__int64)&v74);
          sub_C46A40((__int64)&v78, 1);
          v77 = v79;
          v76 = v78;
          switch ( (int)v73 )
          {
            case '"':
            case '&':
              goto LABEL_67;
            case '#':
            case '\'':
              v59 = sub_DA26C0(a1, (__int64)&v76);
              if ( sub_DDB0E0(a1, v73, v72, v71, v27, v59, a8) )
                goto LABEL_69;
LABEL_67:
              v58 = sub_DA26C0(a1, (__int64)&v74);
              v57 = sub_DDB0E0(a1, v73, v72, v71, v27, v58, a8);
              goto LABEL_64;
            case '$':
            case '(':
              goto LABEL_63;
            case '%':
            case ')':
              v51 = sub_DA26C0(a1, (__int64)&v76);
              v52 = BYTE4(v73);
              v61 = v51;
              v64 = v72;
              v67 = v71;
              v53 = sub_B52F50(v73);
              if ( sub_DDB0E0(a1, ((unsigned __int64)v52 << 32) | v53, v67, v64, v27, v61, a8) )
                goto LABEL_69;
LABEL_63:
              v54 = sub_DA26C0(a1, (__int64)&v74);
              v55 = BYTE4(v73);
              v62 = v54;
              v65 = v72;
              v68 = v71;
              v56 = sub_B52F50(v73);
              v57 = sub_DDB0E0(a1, ((unsigned __int64)v55 << 32) | v56, v68, v65, v27, v62, a8);
LABEL_64:
              if ( !v57 )
              {
LABEL_65:
                sub_969240(&v76);
                goto LABEL_66;
              }
LABEL_69:
              sub_969240(&v76);
              sub_969240(&v74);
              break;
            default:
              goto LABEL_65;
          }
          return 1;
        }
      }
      else if ( sub_C43C50((__int64)&v74, (const void **)(v29 + 24)) )
      {
        goto LABEL_44;
      }
LABEL_66:
      sub_969240(&v74);
      v18 = v70;
      goto LABEL_22;
    }
    return sub_DDB0E0(a1, v73, v72, v71, v69, a7, a8);
  }
  if ( *((_WORD *)v71 + 12) && *((_WORD *)v72 + 12) != 8 )
    return sub_DDB0E0(a1, v70, v71, v72, v69, a7, a8);
  if ( *((_WORD *)a7 + 12) && *((_WORD *)v69 + 12) != 8 )
    return sub_DDB0E0(a1, v73, v72, v71, a7, v69, a8);
  if ( *(_BYTE *)(sub_D95540((__int64)v72) + 8) != 14 && *(_BYTE *)(sub_D95540((__int64)v71) + 8) != 14 )
  {
    v41 = v69;
    v42 = a7;
    v43 = sub_DD1D00(a1, v71, v30, v31, v32);
    v47 = sub_DD1D00(a1, v72, v44, v45, v46);
    if ( sub_DDB0E0(a1, v70, v47, v43, v41, v42, a8) )
      return 1;
  }
  if ( *(_BYTE *)(sub_D95540((__int64)v69) + 8) == 14 || *(_BYTE *)(sub_D95540((__int64)a7) + 8) == 14 )
    return 0;
  v36 = sub_DD1D00(a1, a7, v33, v34, v35);
  v40 = sub_DD1D00(a1, v69, v37, v38, v39);
  return sub_DDB0E0(a1, v73, v72, v71, v40, v36, a8);
}
