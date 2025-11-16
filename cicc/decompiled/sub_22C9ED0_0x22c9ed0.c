// Function: sub_22C9ED0
// Address: 0x22c9ed0
//
__int64 __fastcall sub_22C9ED0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        unsigned __int8 a6,
        int a7)
{
  int v7; // r14d
  __int64 v10; // rbx
  char v11; // al
  int v12; // r15d
  _DWORD *v13; // rax
  __int64 v14; // rdx
  unsigned __int8 v15; // al
  __int64 v16; // rdi
  bool v17; // al
  __int64 v18; // rdx
  bool v19; // zf
  unsigned __int8 v20; // al
  __int64 v21; // rdi
  __int64 v22; // r8
  __int64 v23; // rcx
  _BYTE *v24; // rdi
  __int64 v25; // rbx
  __int64 v26; // rcx
  __int64 v28; // r15
  __int64 v29; // rax
  int v30; // eax
  __int64 v31; // r8
  unsigned int v32; // esi
  __int64 v33; // rdi
  bool v34; // cl
  __int64 v35; // rsi
  __int64 v36; // rax
  unsigned int v37; // r8d
  int v38; // eax
  __int64 v39; // r11
  _BYTE *v40; // rax
  unsigned int v41; // esi
  int v42; // eax
  bool v43; // al
  unsigned int v44; // esi
  int v45; // eax
  _BYTE *v46; // rax
  unsigned __int8 *v47; // r11
  unsigned int v48; // esi
  int v49; // eax
  bool v50; // al
  bool v51; // al
  __int64 *v52; // rsi
  __int64 *v53; // r10
  unsigned __int8 *v54; // r15
  unsigned int i; // ebx
  __int64 v56; // rax
  unsigned int v57; // edx
  unsigned __int8 *v58; // rdi
  __int64 v59; // rdx
  __int64 v60; // r13
  int v61; // ebx
  unsigned int v62; // eax
  _BYTE *v63; // rax
  unsigned int v64; // eax
  unsigned int v65; // eax
  int v66; // [rsp+4h] [rbp-ECh]
  char v67; // [rsp+4h] [rbp-ECh]
  int v68; // [rsp+4h] [rbp-ECh]
  __int64 v69; // [rsp+8h] [rbp-E8h]
  unsigned __int8 *v70; // [rsp+8h] [rbp-E8h]
  int v71; // [rsp+8h] [rbp-E8h]
  char v72; // [rsp+10h] [rbp-E0h]
  __int64 v73; // [rsp+10h] [rbp-E0h]
  __int64 v74; // [rsp+10h] [rbp-E0h]
  __int64 v75; // [rsp+10h] [rbp-E0h]
  __int64 v76; // [rsp+10h] [rbp-E0h]
  __int64 v77; // [rsp+18h] [rbp-D8h]
  int v78; // [rsp+18h] [rbp-D8h]
  bool v79; // [rsp+18h] [rbp-D8h]
  __int64 v80; // [rsp+18h] [rbp-D8h]
  __int64 v81; // [rsp+18h] [rbp-D8h]
  __int64 v82; // [rsp+18h] [rbp-D8h]
  __int64 v83; // [rsp+18h] [rbp-D8h]
  __int64 v84; // [rsp+18h] [rbp-D8h]
  __int64 v85; // [rsp+20h] [rbp-D0h]
  __int64 v86; // [rsp+20h] [rbp-D0h]
  __int64 v87; // [rsp+20h] [rbp-D0h]
  int v88; // [rsp+20h] [rbp-D0h]
  int v89; // [rsp+20h] [rbp-D0h]
  int v90; // [rsp+20h] [rbp-D0h]
  __int64 v91; // [rsp+20h] [rbp-D0h]
  __int64 v92; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v94; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v95; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v96; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v97; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v98; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v99; // [rsp+68h] [rbp-88h]
  unsigned __int64 v100; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v101; // [rsp+78h] [rbp-78h]
  char v102; // [rsp+88h] [rbp-68h]
  unsigned __int64 v103; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v104; // [rsp+98h] [rbp-58h]
  unsigned __int64 v105; // [rsp+A0h] [rbp-50h] BYREF
  unsigned int v106; // [rsp+A8h] [rbp-48h]
  char v107; // [rsp+B8h] [rbp-38h]

  v7 = a2;
  v10 = a4;
  v11 = *(_BYTE *)a4;
  if ( *(_BYTE *)a4 > 0x1Cu )
  {
    switch ( v11 )
    {
      case 'R':
        sub_22C85B0(a1, a2, a3, a4, a5, a6);
        return a1;
      case 'C':
        sub_22C1230((__int64)&v103, a2, a3, a4, a5);
        sub_22C0650(a1, (unsigned __int8 *)&v103);
        *(_BYTE *)(a1 + 40) = 1;
        sub_22C0090((unsigned __int8 *)&v103);
        return a1;
      case ']':
        v28 = *(_QWORD *)(a4 - 32);
        if ( *(_BYTE *)v28 == 85 )
        {
          v29 = *(_QWORD *)(v28 - 32);
          if ( v29 )
          {
            if ( !*(_BYTE *)v29 && *(_QWORD *)(v29 + 24) == *(_QWORD *)(v28 + 80) && (*(_BYTE *)(v29 + 33) & 0x20) != 0 )
            {
              v30 = *(_DWORD *)(v29 + 36);
              if ( v30 != 312 )
              {
                switch ( v30 )
                {
                  case 333:
                  case 339:
                  case 360:
                  case 369:
                  case 372:
                    break;
                  default:
                    goto LABEL_5;
                }
              }
              if ( *(_DWORD *)(a4 + 80) == 1 && **(_DWORD **)(a4 + 72) == 1 )
              {
                if ( a3 == *(_QWORD *)(v28 - 32LL * (*(_DWORD *)(v28 + 4) & 0x7FFFFFF))
                  && ((v58 = *(unsigned __int8 **)(v28 + 32 * (1LL - (*(_DWORD *)(v28 + 4) & 0x7FFFFFF))),
                       v59 = *v58,
                       v60 = (__int64)(v58 + 24),
                       (_BYTE)v59 == 17)
                   || (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v58 + 1) + 8LL) - 17 <= 1
                   && (unsigned __int8)v59 <= 0x15u
                   && (v63 = sub_AD7630((__int64)v58, 0, v59)) != 0
                   && (v60 = (__int64)(v63 + 24), *v63 == 17)) )
                {
                  v61 = sub_B5B690(v28);
                  v62 = sub_B5B5E0(v28);
                  sub_AB3450((__int64)&v94, v62, v60, v61);
                  if ( a5 )
                  {
                    sub_ABB300((__int64)&v103, (__int64)&v94);
                    if ( v95 > 0x40 && v94 )
                      j_j___libc_free_0_0(v94);
                    v94 = v103;
                    v64 = v104;
                    v104 = 0;
                    v95 = v64;
                    if ( v97 > 0x40 && v96 )
                      j_j___libc_free_0_0(v96);
                    v96 = v105;
                    v65 = v106;
                    v106 = 0;
                    v97 = v65;
                    sub_969240((__int64 *)&v105);
                    sub_969240((__int64 *)&v103);
                  }
                  v99 = v95;
                  if ( v95 > 0x40 )
                    sub_C43780((__int64)&v98, (const void **)&v94);
                  else
                    v98 = v94;
                  v101 = v97;
                  if ( v97 > 0x40 )
                    sub_C43780((__int64)&v100, (const void **)&v96);
                  else
                    v100 = v96;
                  sub_22C06B0((__int64)&v103, (__int64)&v98, 0);
                  sub_969240((__int64 *)&v100);
                  sub_969240((__int64 *)&v98);
                  sub_969240((__int64 *)&v96);
                  sub_969240((__int64 *)&v94);
                }
                else
                {
                  LOWORD(v103) = 6;
                }
                sub_22C0650(a1, (unsigned __int8 *)&v103);
                *(_BYTE *)(a1 + 40) = 1;
                sub_22C0090((unsigned __int8 *)&v103);
                return a1;
              }
            }
          }
        }
        break;
    }
  }
LABEL_5:
  v77 = a3;
  v12 = a7 + 1;
  v13 = sub_C94E20((__int64)qword_4F862D0);
  v14 = v77;
  if ( v13 )
  {
    if ( v12 == *v13 )
      goto LABEL_28;
  }
  else if ( v12 == LODWORD(qword_4F862D0[2]) )
  {
    goto LABEL_28;
  }
  v15 = *(_BYTE *)v10;
  if ( *(_BYTE *)v10 == 59 )
  {
    v31 = *(_QWORD *)(v10 - 64);
    if ( *(_BYTE *)v31 == 17 )
    {
      v32 = *(_DWORD *)(v31 + 32);
      if ( !v32 )
        goto LABEL_54;
      if ( v32 > 0x40 )
      {
        v33 = v31 + 24;
        v89 = *(_DWORD *)(v31 + 32);
LABEL_64:
        v42 = sub_C445E0(v33);
        v14 = v77;
        v43 = v89 == v42;
        goto LABEL_65;
      }
      v43 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v32) == *(_QWORD *)(v31 + 24);
    }
    else
    {
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v31 + 8) + 8LL) - 17 > 1 || *(_BYTE *)v31 > 0x15u )
        goto LABEL_66;
      v74 = v77;
      v80 = *(_QWORD *)(v31 + 8);
      v91 = *(_QWORD *)(v10 - 64);
      v40 = sub_AD7630(v91, 0, v14);
      v14 = v74;
      if ( !v40 || *v40 != 17 )
      {
        if ( *(_BYTE *)(v80 + 8) == 17 )
        {
          v66 = *(_DWORD *)(v80 + 32);
          if ( v66 )
          {
            v70 = (unsigned __int8 *)v91;
            v34 = 0;
            v35 = 0;
            while ( 1 )
            {
              v73 = v14;
              v79 = v34;
              v36 = sub_AD69F0(v70, v35);
              v34 = v79;
              v14 = v73;
              if ( !v36 )
                break;
              if ( *(_BYTE *)v36 != 13 )
              {
                if ( *(_BYTE *)v36 != 17 )
                  goto LABEL_66;
                v37 = *(_DWORD *)(v36 + 32);
                if ( v37 )
                {
                  if ( v37 <= 0x40 )
                  {
                    v34 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v37) == *(_QWORD *)(v36 + 24);
                  }
                  else
                  {
                    v90 = *(_DWORD *)(v36 + 32);
                    v38 = sub_C445E0(v36 + 24);
                    v14 = v73;
                    v34 = v90 == v38;
                  }
                  if ( !v34 )
                    goto LABEL_66;
                }
                else
                {
                  v34 = 1;
                }
              }
              v35 = (unsigned int)(v35 + 1);
              if ( v66 == (_DWORD)v35 )
              {
                if ( !v34 )
                  goto LABEL_66;
                goto LABEL_54;
              }
            }
          }
        }
        goto LABEL_66;
      }
      v41 = *((_DWORD *)v40 + 8);
      if ( !v41 )
      {
LABEL_54:
        v39 = *(_QWORD *)(v10 - 32);
        if ( v39 )
          goto LABEL_55;
        goto LABEL_67;
      }
      if ( v41 > 0x40 )
      {
        v77 = v74;
        v33 = (__int64)(v40 + 24);
        v89 = *((_DWORD *)v40 + 8);
        goto LABEL_64;
      }
      v43 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v41) == *((_QWORD *)v40 + 3);
    }
LABEL_65:
    if ( !v43 )
    {
LABEL_66:
      v39 = *(_QWORD *)(v10 - 32);
LABEL_67:
      if ( *(_BYTE *)v39 == 17 )
      {
        v44 = *(_DWORD *)(v39 + 32);
        if ( v44 )
        {
          if ( v44 <= 0x40 )
          {
            if ( *(_QWORD *)(v39 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v44) )
              goto LABEL_71;
          }
          else
          {
            v81 = v14;
            v45 = sub_C445E0(v39 + 24);
            v14 = v81;
            if ( v44 != v45 )
              goto LABEL_71;
          }
        }
      }
      else
      {
        v92 = *(_QWORD *)(v39 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v92 + 8) - 17 > 1 || *(_BYTE *)v39 > 0x15u )
          goto LABEL_71;
        v75 = v14;
        v82 = v39;
        v46 = sub_AD7630(v39, 0, v14);
        v47 = (unsigned __int8 *)v82;
        v14 = v75;
        if ( !v46 || *v46 != 17 )
        {
          if ( *(_BYTE *)(v92 + 8) == 17 )
          {
            v71 = *(_DWORD *)(v92 + 32);
            if ( v71 )
            {
              v67 = 0;
              v84 = v75;
              v54 = v47;
              v76 = v10;
              for ( i = 0; i != v71; ++i )
              {
                v56 = sub_AD69F0(v54, i);
                if ( !v56 )
                {
LABEL_131:
                  v12 = a7 + 1;
                  v14 = v84;
                  v10 = v76;
                  goto LABEL_71;
                }
                if ( *(_BYTE *)v56 != 13 )
                {
                  if ( *(_BYTE *)v56 != 17 )
                    goto LABEL_131;
                  v57 = *(_DWORD *)(v56 + 32);
                  if ( v57 )
                  {
                    if ( v57 <= 0x40 )
                    {
                      if ( *(_QWORD *)(v56 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v57) )
                        goto LABEL_131;
                    }
                    else
                    {
                      v68 = *(_DWORD *)(v56 + 32);
                      if ( v68 != (unsigned int)sub_C445E0(v56 + 24) )
                        goto LABEL_131;
                    }
                  }
                  v67 = 1;
                }
              }
              v12 = a7 + 1;
              v14 = v84;
              v10 = v76;
              if ( v67 )
                goto LABEL_80;
            }
          }
          goto LABEL_71;
        }
        v48 = *((_DWORD *)v46 + 8);
        if ( v48 )
        {
          if ( v48 <= 0x40 )
          {
            v50 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v48) == *((_QWORD *)v46 + 3);
          }
          else
          {
            v49 = sub_C445E0((__int64)(v46 + 24));
            v14 = v75;
            v50 = v48 == v49;
          }
          if ( !v50 )
          {
LABEL_71:
            v15 = *(_BYTE *)v10;
            goto LABEL_8;
          }
        }
      }
LABEL_80:
      v39 = *(_QWORD *)(v10 - 64);
      if ( !v39 )
        goto LABEL_71;
LABEL_55:
      sub_22C9ED0(a1, v7, v14, v39, a5 ^ 1, a6, v12);
      return a1;
    }
    goto LABEL_54;
  }
LABEL_8:
  if ( v15 <= 0x1Cu )
    goto LABEL_28;
  v16 = *(_QWORD *)(v10 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 <= 1 )
    v16 = **(_QWORD **)(v16 + 16);
  v85 = v14;
  v17 = sub_BCAC40(v16, 1);
  v18 = v85;
  v19 = !v17;
  v72 = v17;
  v20 = *(_BYTE *)v10;
  if ( !v19 )
  {
    if ( v20 == 57 )
    {
      if ( (*(_BYTE *)(v10 + 7) & 0x40) != 0 )
        v52 = *(__int64 **)(v10 - 8);
      else
        v52 = (__int64 *)(v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF));
      v26 = *v52;
      if ( *v52 )
      {
        v69 = v52[4];
        if ( v69 )
          goto LABEL_25;
      }
      goto LABEL_87;
    }
    if ( v20 == 86 )
    {
      v21 = *(_QWORD *)(v10 + 8);
      v86 = *(_QWORD *)(v10 - 96);
      if ( *(_QWORD *)(v86 + 8) != v21 || **(_BYTE **)(v10 - 32) > 0x15u )
        goto LABEL_15;
      v83 = v18;
      v69 = *(_QWORD *)(v10 - 64);
      v51 = sub_AC30F0(*(_QWORD *)(v10 - 32));
      v18 = v83;
      v72 = v51;
      if ( v51 )
      {
        LODWORD(v26) = v86;
        if ( v69 )
          goto LABEL_25;
      }
      v20 = *(_BYTE *)v10;
    }
  }
  if ( v20 <= 0x1Cu )
    goto LABEL_28;
LABEL_87:
  v21 = *(_QWORD *)(v10 + 8);
LABEL_15:
  if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 <= 1 )
    v21 = **(_QWORD **)(v21 + 16);
  v87 = v18;
  if ( !sub_BCAC40(v21, 1) )
    goto LABEL_28;
  LODWORD(v18) = v87;
  if ( *(_BYTE *)v10 != 58 )
  {
    if ( *(_BYTE *)v10 == 86 )
    {
      v23 = *(_QWORD *)(v10 - 96);
      v78 = v23;
      if ( *(_QWORD *)(v23 + 8) == *(_QWORD *)(v10 + 8) )
      {
        v24 = *(_BYTE **)(v10 - 64);
        if ( *v24 <= 0x15u )
        {
          v25 = *(_QWORD *)(v10 - 32);
          LODWORD(v69) = v25;
          if ( sub_AD7A80(v24, 1, v87, v23, v22) )
          {
            if ( v25 )
            {
              v72 = 0;
              LODWORD(v18) = v87;
              LODWORD(v26) = v78;
              goto LABEL_25;
            }
          }
        }
      }
    }
LABEL_28:
    *(_BYTE *)(a1 + 40) = 1;
    *(_WORD *)a1 = 6;
    LOWORD(v103) = 0;
    sub_22C0090((unsigned __int8 *)&v103);
    return a1;
  }
  if ( (*(_BYTE *)(v10 + 7) & 0x40) != 0 )
    v53 = *(__int64 **)(v10 - 8);
  else
    v53 = (__int64 *)(v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF));
  v26 = *v53;
  if ( !*v53 )
    goto LABEL_28;
  v69 = v53[4];
  if ( !v69 )
    goto LABEL_28;
  v72 = 0;
LABEL_25:
  v88 = v18;
  sub_22C9ED0((unsigned int)&v98, v7, v18, v26, a5, a6, v12);
  if ( v102 )
  {
    sub_22C9ED0((unsigned int)&v103, v7, v88, v69, a5, a6, v12);
    if ( v107 )
    {
      if ( v72 == a5 )
      {
        sub_22EACA0(&v94, &v98, &v103);
        sub_22C0650(a1, (unsigned __int8 *)&v94);
        *(_BYTE *)(a1 + 40) = 1;
        sub_22C0090((unsigned __int8 *)&v94);
      }
      else
      {
        sub_22C0C70((__int64)&v98, (__int64)&v103, 0, 0, 1u);
        sub_22C05A0(a1, (unsigned __int8 *)&v98);
        *(_BYTE *)(a1 + 40) = 1;
      }
      if ( v107 )
      {
        v107 = 0;
        sub_22C0090((unsigned __int8 *)&v103);
      }
    }
    else
    {
      *(_BYTE *)(a1 + 40) = 0;
    }
    if ( v102 )
    {
      v102 = 0;
      sub_22C0090((unsigned __int8 *)&v98);
    }
  }
  else
  {
    *(_BYTE *)(a1 + 40) = 0;
  }
  return a1;
}
