// Function: sub_9A0640
// Address: 0x9a0640
//
__int16 __fastcall sub_9A0640(__int64 a1, unsigned __int64 a2, unsigned __int8 *a3, unsigned __int8 *a4, char a5)
{
  unsigned __int8 *v6; // rbx
  unsigned __int8 *v7; // r14
  __int64 v8; // r13
  unsigned int v9; // r9d
  bool v10; // r15
  unsigned int v11; // edx
  unsigned __int64 v12; // rax
  __int64 v13; // rsi
  int v14; // eax
  __int64 v15; // r15
  unsigned int v16; // edi
  __int64 v17; // rax
  unsigned __int16 v18; // ax
  __int16 result; // ax
  unsigned __int64 v20; // rax
  __int64 v21; // rsi
  int v22; // eax
  unsigned int v23; // r15d
  __int64 v24; // rsi
  __int64 v25; // rax
  unsigned __int16 v26; // ax
  unsigned __int8 *v27; // rax
  int *v28; // rax
  int v29; // r14d
  unsigned __int8 v30; // al
  int *v31; // rax
  int v32; // r14d
  unsigned __int8 v33; // al
  unsigned int v34; // r14d
  __int64 v35; // rsi
  unsigned int v36; // eax
  __int64 v37; // rdx
  char v38; // cl
  char v39; // al
  unsigned int v40; // edx
  unsigned __int8 *v41; // rax
  unsigned __int8 *v42; // rdx
  char v43; // dl
  bool v44; // zf
  _BYTE *v45; // rax
  __int64 v46; // rax
  unsigned int v47; // r15d
  __int64 v48; // rdi
  __int64 v49; // rsi
  bool v50; // cl
  __int64 v51; // rax
  unsigned int v52; // r8d
  __int64 v53; // rdi
  int v54; // eax
  unsigned int v55; // eax
  unsigned __int8 *v56; // rax
  unsigned int v57; // ebx
  unsigned __int8 *v58; // r15
  __int64 v59; // rax
  unsigned int v60; // r8d
  __int64 v61; // rdx
  unsigned int v62; // edx
  bool v63; // r15
  __int64 v64; // rax
  unsigned int v65; // edx
  unsigned int v66; // r15d
  __int64 v67; // rdi
  __int64 v68; // rsi
  _BYTE *v69; // rax
  int v70; // eax
  int v71; // eax
  unsigned __int8 v72; // [rsp+5h] [rbp-CBh]
  unsigned __int8 v73; // [rsp+5h] [rbp-CBh]
  char v74; // [rsp+6h] [rbp-CAh]
  char v75; // [rsp+6h] [rbp-CAh]
  char v76; // [rsp+7h] [rbp-C9h]
  char v77; // [rsp+7h] [rbp-C9h]
  char v78; // [rsp+7h] [rbp-C9h]
  char v79; // [rsp+7h] [rbp-C9h]
  unsigned int v80; // [rsp+8h] [rbp-C8h]
  unsigned int v81; // [rsp+8h] [rbp-C8h]
  unsigned int v82; // [rsp+10h] [rbp-C0h]
  unsigned int v83; // [rsp+10h] [rbp-C0h]
  char v84; // [rsp+10h] [rbp-C0h]
  int v85; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v86; // [rsp+18h] [rbp-B8h]
  char v87; // [rsp+18h] [rbp-B8h]
  char v88; // [rsp+18h] [rbp-B8h]
  __int64 v89; // [rsp+18h] [rbp-B8h]
  __int64 v90; // [rsp+18h] [rbp-B8h]
  unsigned int v91; // [rsp+18h] [rbp-B8h]
  int v92; // [rsp+18h] [rbp-B8h]
  unsigned int v93; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v94; // [rsp+20h] [rbp-B0h]
  unsigned int v95; // [rsp+28h] [rbp-A8h]
  char v96; // [rsp+28h] [rbp-A8h]
  char v97; // [rsp+28h] [rbp-A8h]
  unsigned int v98; // [rsp+2Ch] [rbp-A4h]
  __int16 v99; // [rsp+2Ch] [rbp-A4h]
  __int16 v100; // [rsp+2Ch] [rbp-A4h]
  __int16 v101; // [rsp+2Ch] [rbp-A4h]
  __int16 v102; // [rsp+2Ch] [rbp-A4h]
  _BYTE *v103; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v104; // [rsp+38h] [rbp-98h] BYREF
  __int64 v105; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v106; // [rsp+48h] [rbp-88h]
  __int64 v107; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v108; // [rsp+58h] [rbp-78h]
  __int64 v109; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v110; // [rsp+68h] [rbp-68h]
  __int64 v111; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v112; // [rsp+78h] [rbp-58h]
  _QWORD *v113; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v114; // [rsp+88h] [rbp-48h]
  __int64 v115; // [rsp+90h] [rbp-40h]
  unsigned int v116; // [rsp+98h] [rbp-38h]

  v6 = a4;
  v94 = a2;
  v7 = *(unsigned __int8 **)(a1 - 64);
  v8 = *(_QWORD *)(a1 - 32);
  v98 = a2;
  v86 = HIDWORD(a2);
  v9 = *(_WORD *)(a1 + 2) & 0x3F;
  v10 = (*(_BYTE *)(a1 + 1) & 2) != 0;
  if ( a5 )
  {
    LODWORD(v104) = *(_WORD *)(a1 + 2) & 0x3F;
    BYTE4(v104) = v10;
    if ( v7 != a4 )
      goto LABEL_3;
  }
  else
  {
    BYTE4(v104) = (*(_BYTE *)(a1 + 1) & 2) != 0;
    LODWORD(v104) = sub_B52870(v9);
    if ( v7 != v6 )
    {
LABEL_3:
      if ( (unsigned __int8 *)v8 == a3 )
      {
        v8 = (__int64)v7;
        LODWORD(v104) = sub_B52F50((unsigned int)v104);
        goto LABEL_39;
      }
      if ( (unsigned __int8 *)v8 != v6 )
        goto LABEL_5;
      if ( v7 != a3 )
      {
LABEL_38:
        v8 = (__int64)v7;
        LODWORD(v104) = sub_B52F50((unsigned int)v104);
        v98 = sub_B52F50(v98);
        v27 = a3;
        a3 = v6;
        v6 = v27;
        goto LABEL_39;
      }
      goto LABEL_87;
    }
  }
  v98 = sub_B52F50((unsigned int)a2);
  if ( (unsigned __int8 *)v8 != v6 )
  {
    v44 = v8 == (_QWORD)a3;
    v6 = a3;
    a3 = v7;
    if ( !v44 )
    {
LABEL_5:
      if ( v7 != a3 )
      {
LABEL_6:
        v11 = sub_B538E0(&v104);
        if ( v11 - 38 > 1 )
        {
          if ( v11 - 40 <= 1 )
          {
            v12 = *a3;
            if ( (unsigned __int8)v12 <= 0x1Cu )
            {
              v15 = v98;
              if ( (_BYTE)v12 != 5 )
                goto LABEL_74;
              v14 = *((unsigned __int16 *)a3 + 1);
              if ( (*((_WORD *)a3 + 1) & 0xFFF7) != 0x11 && (v14 & 0xFFFD) != 0xD )
                goto LABEL_74;
            }
            else
            {
              if ( (unsigned __int8)v12 > 0x36u )
                goto LABEL_73;
              v13 = 0x40540000000000LL;
              if ( !_bittest64(&v13, v12) )
                goto LABEL_73;
              v14 = (unsigned __int8)v12 - 29;
            }
            if ( v14 == 15 )
            {
              v15 = v98;
              if ( (a3[1] & 4) == 0 || *((unsigned __int8 **)a3 - 8) != v7 || *((_QWORD *)a3 - 4) != v8 )
                goto LABEL_74;
              if ( *v6 == 17 )
              {
                v16 = *((_DWORD *)v6 + 8);
                v17 = *((_QWORD *)v6 + 3);
                if ( v16 > 0x40 )
                  v17 = *(_QWORD *)(v17 + 8LL * ((v16 - 1) >> 6));
                if ( (v17 & (1LL << ((unsigned __int8)v16 - 1))) == 0 )
                  goto LABEL_20;
                goto LABEL_73;
              }
              v83 = v11;
              v90 = *((_QWORD *)v6 + 1);
              if ( (unsigned int)*(unsigned __int8 *)(v90 + 8) - 17 > 1 || *v6 > 0x15u )
                goto LABEL_74;
              v51 = sub_AD7630(v6, 0);
              v11 = v83;
              if ( v51 && *(_BYTE *)v51 == 17 )
              {
                v52 = *(_DWORD *)(v51 + 32);
                v53 = *(_QWORD *)(v51 + 24);
                if ( v52 > 0x40 )
                  v53 = *(_QWORD *)(v53 + 8LL * ((v52 - 1) >> 6));
                if ( (v53 & (1LL << ((unsigned __int8)v52 - 1))) != 0 )
                  goto LABEL_73;
              }
              else
              {
                if ( *(_BYTE *)(v90 + 8) != 17 )
                  goto LABEL_73;
                v92 = *(_DWORD *)(v90 + 32);
                if ( !v92 )
                  goto LABEL_73;
                v84 = 0;
                v56 = v6;
                v80 = v11;
                v57 = 0;
                v58 = v56;
                do
                {
                  v59 = sub_AD69F0(v58, v57);
                  if ( !v59 )
                  {
LABEL_72:
                    v6 = v58;
                    goto LABEL_73;
                  }
                  if ( *(_BYTE *)v59 != 13 )
                  {
                    if ( *(_BYTE *)v59 != 17 )
                      goto LABEL_72;
                    v60 = *(_DWORD *)(v59 + 32);
                    v61 = *(_QWORD *)(v59 + 24);
                    if ( v60 > 0x40 )
                      v61 = *(_QWORD *)(v61 + 8LL * ((v60 - 1) >> 6));
                    if ( (v61 & (1LL << ((unsigned __int8)v60 - 1))) != 0 )
                      goto LABEL_72;
                    v84 = 1;
                  }
                  ++v57;
                }
                while ( v92 != v57 );
                v11 = v80;
                v6 = v58;
                if ( !v84 )
                  goto LABEL_73;
              }
LABEL_20:
              v15 = v98;
              v94 = v98 | v94 & 0xFFFFFFFF00000000LL;
              v18 = sub_B53860(v11, v94);
              if ( HIBYTE(v18) && (_BYTE)v18 )
                return 257;
              goto LABEL_74;
            }
          }
LABEL_73:
          v15 = v98;
          goto LABEL_74;
        }
        v20 = *a3;
        if ( (unsigned __int8)v20 <= 0x1Cu )
        {
          v15 = v98;
          if ( (_BYTE)v20 != 5 )
            goto LABEL_74;
          v22 = *((unsigned __int16 *)a3 + 1);
          if ( (*((_WORD *)a3 + 1) & 0xFFFD) != 0xD && (v22 & 0xFFF7) != 0x11 )
            goto LABEL_74;
        }
        else
        {
          if ( (unsigned __int8)v20 > 0x36u )
            goto LABEL_73;
          v21 = 0x40540000000000LL;
          if ( !_bittest64(&v21, v20) )
            goto LABEL_73;
          v22 = (unsigned __int8)v20 - 29;
        }
        if ( v22 != 15 )
          goto LABEL_73;
        v15 = v98;
        if ( (a3[1] & 4) == 0 || *((unsigned __int8 **)a3 - 8) != v7 || *((_QWORD *)a3 - 4) != v8 )
          goto LABEL_74;
        if ( *v6 == 17 )
        {
          v23 = *((_DWORD *)v6 + 8);
          v24 = *((_QWORD *)v6 + 3);
          v25 = 1LL << ((unsigned __int8)v23 - 1);
          if ( v23 > 0x40 )
          {
            if ( (*(_QWORD *)(v24 + 8LL * ((v23 - 1) >> 6)) & v25) != 0 )
              goto LABEL_34;
            v91 = v11;
            v54 = sub_C444A0(v6 + 24);
            v11 = v91;
            v50 = v23 == v54;
          }
          else
          {
            if ( (v25 & v24) != 0 )
              goto LABEL_34;
            v50 = v24 == 0;
          }
LABEL_136:
          v15 = v98;
          if ( !v50 )
          {
LABEL_74:
            if ( a3 == v7
              && (unsigned int)(v104 - 35) <= 1
              && v98 - 35 <= 1
              && *a3 == 42
              && ((v41 = (unsigned __int8 *)*((_QWORD *)a3 - 8),
                   v42 = (unsigned __int8 *)*((_QWORD *)a3 - 4),
                   v41 == (unsigned __int8 *)v8)
               && v42 == v6
               || v41 == v6 && v42 == (unsigned __int8 *)v8) )
            {
              sub_B53630(v104, v15 | v94 & 0xFFFFFFFF00000000LL);
              LOBYTE(result) = v43;
              HIBYTE(result) = 1;
            }
            else
            {
              v113 = (_QWORD *)sub_B53630(v104, v15 | v94 & 0xFFFFFFFF00000000LL);
              result = 0;
              v114 = v40;
              if ( (_BYTE)v40 )
                return sub_9959F0((unsigned int)v113, v7, (unsigned __int8 *)v8, a3, v6);
            }
            return result;
          }
LABEL_34:
          v15 = v98;
          v94 = v98 | v94 & 0xFFFFFFFF00000000LL;
          v26 = sub_B53860(v11, v94);
          if ( HIBYTE(v26) && !(_BYTE)v26 )
            return 256;
          goto LABEL_74;
        }
        v82 = v11;
        v89 = *((_QWORD *)v6 + 1);
        if ( (unsigned int)*(unsigned __int8 *)(v89 + 8) - 17 > 1 || *v6 > 0x15u )
          goto LABEL_74;
        v46 = sub_AD7630(v6, 0);
        v11 = v82;
        if ( v46 && *(_BYTE *)v46 == 17 )
        {
          v47 = *(_DWORD *)(v46 + 32);
          v48 = *(_QWORD *)(v46 + 24);
          v49 = 1LL << ((unsigned __int8)v47 - 1);
          if ( v47 > 0x40 )
          {
            if ( (*(_QWORD *)(v48 + 8LL * ((v47 - 1) >> 6)) & v49) != 0 )
              goto LABEL_34;
            v70 = sub_C444A0(v46 + 24);
            v11 = v82;
            v50 = v47 == v70;
          }
          else
          {
            if ( (v49 & v48) != 0 )
              goto LABEL_34;
            v50 = v48 == 0;
          }
          goto LABEL_136;
        }
        if ( *(_BYTE *)(v89 + 8) != 17 )
          goto LABEL_73;
        v85 = *(_DWORD *)(v89 + 32);
        if ( !v85 )
          goto LABEL_73;
        v81 = v11;
        v62 = 0;
        v63 = 0;
        while ( 1 )
        {
          v93 = v62;
          v64 = sub_AD69F0(v6, v62);
          v65 = v93;
          if ( !v64 )
            goto LABEL_73;
          if ( *(_BYTE *)v64 != 13 )
          {
            if ( *(_BYTE *)v64 != 17 )
              goto LABEL_73;
            v66 = *(_DWORD *)(v64 + 32);
            v67 = *(_QWORD *)(v64 + 24);
            v68 = 1LL << ((unsigned __int8)v66 - 1);
            if ( v66 > 0x40 )
            {
              if ( (*(_QWORD *)(v67 + 8LL * ((v66 - 1) >> 6)) & v68) == 0 )
              {
                v71 = sub_C444A0(v64 + 24);
                v65 = v93;
                v63 = v66 == v71;
                goto LABEL_189;
              }
            }
            else if ( (v68 & v67) == 0 )
            {
              v63 = v67 == 0;
LABEL_189:
              if ( !v63 )
                goto LABEL_73;
              goto LABEL_178;
            }
            v63 = 1;
          }
LABEL_178:
          v62 = v65 + 1;
          if ( v85 == v62 )
          {
            v11 = v81;
            v50 = v63;
            goto LABEL_136;
          }
        }
      }
      goto LABEL_39;
    }
LABEL_87:
    v8 = (__int64)v6;
    goto LABEL_88;
  }
  LODWORD(v104) = sub_B52F50((unsigned int)v104);
  if ( (unsigned __int8 *)v8 != a3 )
  {
    v6 = a3;
    a3 = (unsigned __int8 *)v8;
    goto LABEL_39;
  }
LABEL_88:
  if ( *a3 > 0x15u || *a3 == 5 )
  {
    v6 = (unsigned __int8 *)v8;
    goto LABEL_39;
  }
  v6 = (unsigned __int8 *)v8;
  if ( !(unsigned __int8)sub_AD6CA0(a3) )
  {
    v7 = a3;
    goto LABEL_38;
  }
LABEL_39:
  if ( *(_BYTE *)v8 == 17 )
  {
    v103 = (_BYTE *)(v8 + 24);
    goto LABEL_41;
  }
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17 <= 1 && *(_BYTE *)v8 <= 0x15u )
  {
    v45 = (_BYTE *)sub_AD7630(v8, 0);
    if ( v45 )
    {
      if ( *v45 == 17 )
        goto LABEL_118;
    }
  }
  if ( *v6 != 17 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v6 + 1) + 8LL) - 17 > 1 )
      goto LABEL_127;
    if ( *v6 > 0x15u )
      goto LABEL_127;
    v45 = (_BYTE *)sub_AD7630(v6, 0);
    if ( !v45 || *v45 != 17 )
      goto LABEL_127;
LABEL_118:
    v103 = v45 + 24;
    goto LABEL_41;
  }
  v103 = v6 + 24;
LABEL_41:
  v28 = (int *)sub_C94E20(qword_4F862D0);
  if ( v28 )
    v29 = *v28;
  else
    v29 = qword_4F862D0[2];
  v30 = sub_B532B0((unsigned int)v104);
  sub_99D930((__int64)&v105, (unsigned __int8 *)v8, v30, 1u, 0, 0, 0, v29 - 1);
  v31 = (int *)sub_C94E20(qword_4F862D0);
  if ( v31 )
    v32 = *v31;
  else
    v32 = qword_4F862D0[2];
  v33 = sub_B532B0(v98);
  sub_99D930((__int64)&v109, v6, v33, 1u, 0, 0, 0, v32 - 1);
  v34 = v104;
  sub_AB15A0(&v113, (unsigned int)v104, &v105);
  v35 = v98;
  LOBYTE(v36) = sub_ABB410(&v113, v98, &v109);
  v37 = 1;
  v38 = 1;
  if ( !(_BYTE)v36 )
  {
    v35 = (unsigned int)sub_B52870(v98);
    v36 = sub_ABB410(&v113, v35, &v109);
    v38 = 0;
    v37 = v36;
  }
  if ( v116 > 0x40 && v115 )
  {
    v72 = v37;
    v74 = v36;
    v76 = v38;
    j_j___libc_free_0_0(v115);
    v37 = v72;
    LOBYTE(v36) = v74;
    v38 = v76;
  }
  if ( v114 > 0x40 && v113 )
  {
    v73 = v37;
    v75 = v36;
    v77 = v38;
    j_j___libc_free_0_0(v113);
    v37 = v73;
    LOBYTE(v36) = v75;
    v38 = v77;
  }
  if ( !(_BYTE)v36 )
  {
    if ( BYTE4(v104) == (_BYTE)v86 )
      goto LABEL_68;
    if ( BYTE4(v104) )
      v34 = sub_B53550((unsigned int)v104, v35, v37);
    v95 = v98;
    if ( (_BYTE)v86 )
      v95 = sub_B53550(v98, v35, v37);
    sub_AB15A0(&v113, v34, &v105);
    v39 = sub_ABB410(&v113, v95, &v109);
    LOBYTE(v37) = 1;
    v38 = 1;
    if ( !v39 )
    {
      v55 = sub_B52870(v95);
      v39 = sub_ABB410(&v113, v55, &v109);
      v38 = 0;
      LOBYTE(v37) = v39;
    }
    if ( v116 > 0x40 && v115 )
    {
      v78 = v37;
      v96 = v39;
      v87 = v38;
      j_j___libc_free_0_0(v115);
      LOBYTE(v37) = v78;
      v39 = v96;
      v38 = v87;
    }
    if ( v114 > 0x40 && v113 )
    {
      v79 = v37;
      v97 = v39;
      v88 = v38;
      j_j___libc_free_0_0(v113);
      LOBYTE(v37) = v79;
      v39 = v97;
      v38 = v88;
    }
    if ( !v39 )
    {
LABEL_68:
      LOBYTE(v114) = 0;
      v113 = &v103;
      if ( (unsigned __int8)sub_991580((__int64)&v113, v8) )
      {
        if ( *v6 == 17 )
        {
          v103 = v6 + 24;
LABEL_71:
          result = 0;
          goto LABEL_102;
        }
        if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v6 + 1) + 8LL) - 17 <= 1 && *v6 <= 0x15u )
        {
          v69 = (_BYTE *)sub_AD7630(v6, 0);
          if ( v69 )
          {
            if ( *v69 == 17 )
            {
              v103 = v69 + 24;
              goto LABEL_71;
            }
          }
        }
      }
      sub_969240(&v111);
      sub_969240(&v109);
      sub_969240(&v107);
      sub_969240(&v105);
LABEL_127:
      v7 = a3;
      if ( (unsigned __int8 *)v8 == v6 )
        return sub_B53860(v104, v98 | v94 & 0xFFFFFFFF00000000LL);
      goto LABEL_6;
    }
  }
  LOBYTE(result) = v38;
  HIBYTE(result) = v37;
LABEL_102:
  if ( v112 > 0x40 && v111 )
  {
    v99 = result;
    j_j___libc_free_0_0(v111);
    result = v99;
  }
  if ( v110 > 0x40 && v109 )
  {
    v100 = result;
    j_j___libc_free_0_0(v109);
    result = v100;
  }
  if ( v108 > 0x40 && v107 )
  {
    v101 = result;
    j_j___libc_free_0_0(v107);
    result = v101;
  }
  if ( v106 > 0x40 && v105 )
  {
    v102 = result;
    j_j___libc_free_0_0(v105);
    return v102;
  }
  return result;
}
