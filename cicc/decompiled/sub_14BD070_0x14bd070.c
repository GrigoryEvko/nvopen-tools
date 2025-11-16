// Function: sub_14BD070
// Address: 0x14bd070
//
char __fastcall sub_14BD070(_QWORD *a1, __int64 a2, int a3, __int64 a4, unsigned int a5)
{
  __int64 *v7; // r12
  unsigned __int8 v8; // bl
  unsigned __int8 v9; // al
  bool v10; // al
  __int64 v11; // rdx
  char result; // al
  __int64 v13; // rax
  int *v14; // rax
  int v15; // eax
  unsigned __int8 v16; // al
  unsigned int v17; // r15d
  int v18; // eax
  unsigned __int16 v19; // ax
  _QWORD *v20; // rax
  unsigned int v21; // r15d
  __int64 v22; // rax
  __int64 v23; // rax
  __int16 v24; // ax
  __int64 v25; // rax
  __int64 v26; // r15
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  unsigned int v30; // r8d
  char v31; // al
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // r15
  unsigned __int8 v35; // al
  int v36; // eax
  bool v37; // al
  char v38; // al
  unsigned __int8 v39; // al
  __int64 v40; // rax
  unsigned int v41; // r15d
  int v42; // eax
  unsigned int v43; // r15d
  int v44; // eax
  __int64 *v45; // rdi
  char v46; // al
  __int64 *v47; // rdi
  __int64 *v48; // rdi
  __int64 *v49; // r9
  char v50; // al
  char v51; // al
  unsigned int v52; // eax
  unsigned int v53; // eax
  unsigned int v54; // edx
  __int64 v55; // rax
  int v56; // eax
  bool v57; // al
  __int64 *v58; // rax
  __int64 v59; // rcx
  __int64 *v60; // rcx
  unsigned int v61; // edx
  __int64 v62; // rax
  int v63; // eax
  bool v64; // al
  int v65; // [rsp+18h] [rbp-C8h]
  int v66; // [rsp+18h] [rbp-C8h]
  int v67; // [rsp+20h] [rbp-C0h]
  int v68; // [rsp+20h] [rbp-C0h]
  int v69; // [rsp+20h] [rbp-C0h]
  int v70; // [rsp+28h] [rbp-B8h]
  int v71; // [rsp+28h] [rbp-B8h]
  unsigned int v72; // [rsp+28h] [rbp-B8h]
  unsigned int v73; // [rsp+28h] [rbp-B8h]
  unsigned int v74; // [rsp+28h] [rbp-B8h]
  __int64 *v75; // [rsp+30h] [rbp-B0h] BYREF
  __int64 *v76; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v77; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v78; // [rsp+48h] [rbp-98h]
  __int64 v79; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v80; // [rsp+58h] [rbp-88h]
  __int64 v81; // [rsp+60h] [rbp-80h] BYREF
  unsigned int v82; // [rsp+68h] [rbp-78h]
  __int64 v83[2]; // [rsp+70h] [rbp-70h] BYREF
  __int64 v84[2]; // [rsp+80h] [rbp-60h] BYREF
  __int64 **v85; // [rsp+90h] [rbp-50h] BYREF
  __int64 **v86; // [rsp+98h] [rbp-48h]
  __int64 v87[8]; // [rsp+A0h] [rbp-40h] BYREF

  v7 = (__int64 *)a4;
  v8 = a2;
  v9 = *((_BYTE *)a1 + 16);
  if ( v9 == 13 )
  {
    if ( *((_DWORD *)a1 + 8) <= 0x40u )
    {
      v13 = a1[3];
      if ( v13 && (v13 & (v13 - 1)) == 0 )
        return 1;
      goto LABEL_8;
    }
    v10 = (unsigned int)sub_16A5940(a1 + 3) == 1;
    goto LABEL_4;
  }
  v11 = *a1;
  if ( *(_BYTE *)(*a1 + 8LL) == 16 && v9 <= 0x10u )
  {
    v32 = sub_15A1020(a1);
    if ( v32 && *(_BYTE *)(v32 + 16) == 13 )
    {
      if ( *(_DWORD *)(v32 + 32) > 0x40u )
      {
        v10 = (unsigned int)sub_16A5940(v32 + 24) == 1;
LABEL_4:
        if ( v10 )
          return 1;
        goto LABEL_37;
      }
      v33 = *(_QWORD *)(v32 + 24);
      if ( v33 )
      {
        v11 = v33 - 1;
        if ( (v33 & (v33 - 1)) == 0 )
          return 1;
      }
    }
    else
    {
      v70 = *(_QWORD *)(*a1 + 32LL);
      if ( !v70 )
        return 1;
      v21 = 0;
      while ( 1 )
      {
        a2 = v21;
        v22 = sub_15A0A60(a1, v21);
        if ( !v22 )
          break;
        v11 = *(unsigned __int8 *)(v22 + 16);
        if ( (_BYTE)v11 != 9 )
        {
          if ( (_BYTE)v11 != 13 )
            break;
          if ( *(_DWORD *)(v22 + 32) > 0x40u )
          {
            if ( (unsigned int)sub_16A5940(v22 + 24) != 1 )
              break;
          }
          else
          {
            v23 = *(_QWORD *)(v22 + 24);
            if ( !v23 )
              break;
            v11 = v23 - 1;
            if ( (v23 & (v23 - 1)) != 0 )
              break;
          }
        }
        if ( v70 == ++v21 )
          return 1;
      }
    }
LABEL_37:
    v9 = *((_BYTE *)a1 + 16);
  }
  switch ( v9 )
  {
    case 0x2Fu:
      v34 = *(a1 - 6);
      v35 = *(_BYTE *)(v34 + 16);
      if ( v35 != 13 )
      {
        if ( *(_BYTE *)(*(_QWORD *)v34 + 8LL) != 16 || v35 > 0x10u )
          goto LABEL_8;
        v40 = sub_15A1020(*(a1 - 6));
        if ( !v40 || *(_BYTE *)(v40 + 16) != 13 )
        {
          v67 = *(_QWORD *)(*(_QWORD *)v34 + 32LL);
          if ( !v67 )
            return 1;
          v54 = 0;
          while ( 1 )
          {
            a2 = v54;
            v73 = v54;
            v55 = sub_15A0A60(v34, v54);
            if ( !v55 )
              break;
            a4 = *(unsigned __int8 *)(v55 + 16);
            v11 = v73;
            if ( (_BYTE)a4 != 9 )
            {
              if ( (_BYTE)a4 != 13 )
                break;
              a4 = *(unsigned int *)(v55 + 32);
              if ( (unsigned int)a4 <= 0x40 )
              {
                v57 = *(_QWORD *)(v55 + 24) == 1;
              }
              else
              {
                v65 = *(_DWORD *)(v55 + 32);
                v56 = sub_16A57B0(v55 + 24);
                v11 = v73;
                a4 = (unsigned int)(v65 - 1);
                v57 = (_DWORD)a4 == v56;
              }
              if ( !v57 )
                break;
            }
            v54 = v11 + 1;
            if ( v67 == v54 )
              return 1;
          }
          goto LABEL_59;
        }
        v41 = *(_DWORD *)(v40 + 32);
        if ( v41 > 0x40 )
        {
          v42 = sub_16A57B0(v40 + 24);
          v11 = v41 - 1;
          if ( v42 == (_DWORD)v11 )
            return 1;
          goto LABEL_59;
        }
LABEL_181:
        v37 = *(_QWORD *)(v40 + 24) == 1;
        goto LABEL_58;
      }
      break;
    case 5u:
      v24 = *((_WORD *)a1 + 9);
      if ( v24 != 23 )
        goto LABEL_41;
      v34 = a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
      if ( *(_BYTE *)(v34 + 16) != 13 )
      {
        if ( *(_BYTE *)(*(_QWORD *)v34 + 8LL) != 16 )
          goto LABEL_8;
        v40 = sub_15A1020(v34);
        if ( !v40 || *(_BYTE *)(v40 + 16) != 13 )
        {
          v69 = *(_QWORD *)(*(_QWORD *)v34 + 32LL);
          if ( !v69 )
            return 1;
          v61 = 0;
          while ( 1 )
          {
            a2 = v61;
            v74 = v61;
            v62 = sub_15A0A60(v34, v61);
            if ( !v62 )
              break;
            a4 = *(unsigned __int8 *)(v62 + 16);
            v11 = v74;
            if ( (_BYTE)a4 != 9 )
            {
              if ( (_BYTE)a4 != 13 )
                break;
              a4 = *(unsigned int *)(v62 + 32);
              if ( (unsigned int)a4 <= 0x40 )
              {
                v64 = *(_QWORD *)(v62 + 24) == 1;
              }
              else
              {
                v66 = *(_DWORD *)(v62 + 32);
                v63 = sub_16A57B0(v62 + 24);
                v11 = v74;
                a4 = (unsigned int)(v66 - 1);
                v64 = (_DWORD)a4 == v63;
              }
              if ( !v64 )
                break;
            }
            v61 = v11 + 1;
            if ( v69 == v61 )
              return 1;
          }
          goto LABEL_59;
        }
        v43 = *(_DWORD *)(v40 + 32);
        if ( v43 > 0x40 )
        {
          v44 = sub_16A57B0(v40 + 24);
          v11 = v43 - 1;
          v37 = (_DWORD)v11 == v44;
          goto LABEL_58;
        }
        goto LABEL_181;
      }
      break;
    case 0x30u:
LABEL_60:
      v26 = *(a1 - 6);
      v39 = *(_BYTE *)(v26 + 16);
      if ( v39 != 13 )
      {
        if ( *(_BYTE *)(*(_QWORD *)v26 + 8LL) != 16 || v39 > 0x10u )
          goto LABEL_8;
        v27 = sub_15A1020(*(a1 - 6));
        if ( !v27 || *(_BYTE *)(v27 + 16) != 13 )
        {
LABEL_46:
          v31 = sub_14A9430(v26);
          goto LABEL_62;
        }
        goto LABEL_99;
      }
LABEL_61:
      v31 = sub_13CFF40((__int64 *)(v26 + 24), a2, v11, a4, a5);
      goto LABEL_62;
    default:
      goto LABEL_8;
  }
  v11 = *(unsigned int *)(v34 + 32);
  if ( (unsigned int)v11 <= 0x40 )
  {
    v37 = *(_QWORD *)(v34 + 24) == 1;
  }
  else
  {
    v71 = *(_DWORD *)(v34 + 32);
    v36 = sub_16A57B0(v34 + 24);
    v11 = (unsigned int)(v71 - 1);
    v37 = (_DWORD)v11 == v36;
  }
LABEL_58:
  if ( v37 )
    return 1;
LABEL_59:
  v38 = *((_BYTE *)a1 + 16);
  if ( v38 == 48 )
    goto LABEL_60;
  if ( v38 != 5 )
    goto LABEL_8;
  v24 = *((_WORD *)a1 + 9);
LABEL_41:
  if ( v24 != 24 )
    goto LABEL_8;
  v25 = *((_DWORD *)a1 + 5) & 0xFFFFFFF;
  v11 = 4 * v25;
  v26 = a1[-3 * v25];
  if ( *(_BYTE *)(v26 + 16) == 13 )
    goto LABEL_61;
  if ( *(_BYTE *)(*(_QWORD *)v26 + 8LL) != 16 )
    goto LABEL_8;
  v27 = sub_15A1020(v26);
  if ( !v27 || *(_BYTE *)(v27 + 16) != 13 )
    goto LABEL_46;
LABEL_99:
  v31 = sub_13CFF40((__int64 *)(v27 + 24), a2, v28, v29, v30);
LABEL_62:
  if ( v31 )
    return 1;
LABEL_8:
  v14 = (int *)sub_16D40F0(qword_4FBB370);
  if ( v14 )
    v15 = *v14;
  else
    v15 = qword_4FBB370[2];
  if ( a3 == v15 )
    return 0;
  v75 = 0;
  v16 = *((_BYTE *)a1 + 16);
  v17 = a3 + 1;
  v76 = 0;
  if ( !v8 )
  {
    if ( v16 <= 0x17u )
    {
LABEL_84:
      if ( v16 == 5 )
      {
LABEL_19:
        v18 = *((unsigned __int16 *)a1 + 9);
        if ( (_WORD)v18 != 11
          || (v59 = *((_DWORD *)a1 + 5) & 0xFFFFFFF, !a1[-3 * v59])
          || (v75 = (__int64 *)a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)], (v60 = (__int64 *)a1[3 * (1 - v59)]) == 0) )
        {
LABEL_20:
          if ( ((unsigned __int16)(v18 - 24) <= 1u || (unsigned int)(v18 - 17) <= 1) && (*((_BYTE *)a1 + 17) & 2) != 0 )
          {
            v19 = *((_WORD *)a1 + 9);
            if ( v19 == 24
              || ((unsigned int)v19 - 17 <= 1 || (unsigned __int16)(v19 - 24) <= 1u)
              && (*((_BYTE *)a1 + 17) & 2) != 0
              && v19 == 17 )
            {
LABEL_28:
              v20 = (_QWORD *)sub_13CF970((__int64)a1);
              return sub_14BDDF0(*v20, v8, v17, v7);
            }
          }
          return 0;
        }
        v76 = v60;
        v16 = 5;
LABEL_116:
        if ( !v8 && (*((_BYTE *)a1 + 17) & 2) == 0 && ((*((_BYTE *)a1 + 17) >> 1) & 2) == 0 )
        {
LABEL_119:
          if ( v16 <= 0x17u )
          {
            if ( v16 != 5 )
              return 0;
            v18 = *((unsigned __int16 *)a1 + 9);
            goto LABEL_20;
          }
LABEL_86:
          if ( ((unsigned __int8)(v16 - 48) <= 1u || (unsigned int)v16 - 41 <= 1)
            && (*((_BYTE *)a1 + 17) & 2) != 0
            && (v16 == 48
             || ((unsigned __int8)(v16 - 48) <= 1u || (unsigned int)v16 - 41 <= 1)
             && (*((_BYTE *)a1 + 17) & 2) != 0
             && v16 == 41) )
          {
            goto LABEL_28;
          }
          return 0;
        }
        v48 = v75;
        v49 = v76;
        v50 = *((_BYTE *)v75 + 16);
        if ( v50 == 50 )
        {
          if ( v76 != (__int64 *)*(v75 - 6) && v76 != (__int64 *)*(v75 - 3) )
            goto LABEL_125;
        }
        else if ( v50 != 5
               || *((_WORD *)v75 + 9) != 26
               || v76 != (__int64 *)v75[-3 * (*((_DWORD *)v75 + 5) & 0xFFFFFFF)]
               && v76 != (__int64 *)v75[3 * (1LL - (*((_DWORD *)v75 + 5) & 0xFFFFFFF))] )
        {
          goto LABEL_125;
        }
        if ( (unsigned __int8)sub_14BDDF0(v76, v8, v17, v7) )
          return 1;
        v48 = v75;
        v49 = v76;
LABEL_125:
        v51 = *((_BYTE *)v49 + 16);
        if ( v51 == 50 )
        {
          if ( v48 != (__int64 *)*(v49 - 6) && v48 != (__int64 *)*(v49 - 3) )
            goto LABEL_128;
        }
        else if ( v51 != 5
               || *((_WORD *)v49 + 9) != 26
               || v48 != (__int64 *)v49[-3 * (*((_DWORD *)v49 + 5) & 0xFFFFFFF)]
               && v48 != (__int64 *)v49[3 * (1LL - (*((_DWORD *)v49 + 5) & 0xFFFFFFF))] )
        {
          goto LABEL_128;
        }
        if ( (unsigned __int8)sub_14BDDF0(v48, v8, v17, v7) )
          return 1;
LABEL_128:
        v72 = sub_16431D0(*a1);
        sub_14AA4E0((__int64)v83, v72);
        sub_14B86A0(v75, (__int64)v83, v17, v7);
        sub_14AA4E0((__int64)&v85, v72);
        sub_14B86A0(v76, (__int64)&v85, v17, v7);
        sub_13A38D0((__int64)&v77, (__int64)v83);
        sub_14A9240((__int64)&v77, (__int64 *)&v85);
        v52 = v78;
        v78 = 0;
        v80 = v52;
        v79 = v77;
        sub_13D0570((__int64)&v79);
        v53 = v80;
        v80 = 0;
        v82 = v53;
        v81 = v79;
        if ( v53 > 0x40 )
        {
          v68 = sub_16A5940(&v81);
          sub_135E100(&v81);
          sub_135E100(&v79);
          sub_135E100(&v77);
          if ( v68 == 1 )
            goto LABEL_132;
        }
        else
        {
          if ( v79 && (v79 & (v79 - 1)) == 0 )
          {
            sub_135E100(&v81);
            sub_135E100(&v79);
            sub_135E100(&v77);
LABEL_132:
            if ( v8 || !sub_13D01C0((__int64)v87) || !sub_13D01C0((__int64)v84) )
            {
              sub_135E100(v87);
              sub_135E100((__int64 *)&v85);
              sub_135E100(v84);
              sub_135E100(v83);
              return 1;
            }
            goto LABEL_135;
          }
          sub_135E100(&v81);
          sub_135E100(&v79);
          sub_135E100(&v77);
        }
LABEL_135:
        sub_135E100(v87);
        sub_135E100((__int64 *)&v85);
        sub_135E100(v84);
        sub_135E100(v83);
        v16 = *((_BYTE *)a1 + 16);
        goto LABEL_119;
      }
      goto LABEL_85;
    }
    goto LABEL_13;
  }
  if ( v16 != 47 )
  {
    if ( v16 == 5 )
    {
      if ( *((_WORD *)a1 + 9) != 23 || (v45 = (__int64 *)a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)]) == 0 )
      {
        if ( *((_WORD *)a1 + 9) != 24 )
          goto LABEL_16;
        v45 = (__int64 *)a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
        if ( !v45 )
          goto LABEL_16;
      }
    }
    else
    {
      if ( v16 != 48 )
      {
        if ( v16 <= 0x17u )
          goto LABEL_16;
LABEL_13:
        if ( v16 != 61 )
        {
          if ( v16 != 79 )
            goto LABEL_15;
          if ( !(unsigned __int8)sub_14BDDF0(*(a1 - 6), v8, v17, v7) )
            return 0;
        }
        return sub_14BDDF0(*(a1 - 3), v8, v17, v7);
      }
      v45 = (__int64 *)*(a1 - 6);
      if ( !v45 )
        goto LABEL_16;
    }
LABEL_108:
    v75 = v45;
    return sub_14BDDF0(v45, 1, v17, v7);
  }
  v45 = (__int64 *)*(a1 - 6);
  if ( v45 )
    goto LABEL_108;
LABEL_15:
  if ( !v8 )
    goto LABEL_112;
LABEL_16:
  v85 = &v75;
  v86 = &v76;
  switch ( v16 )
  {
    case 0x32u:
      v47 = (__int64 *)*(a1 - 6);
      if ( !v47 )
        return 0;
      v58 = (__int64 *)*(a1 - 3);
      v75 = (__int64 *)*(a1 - 6);
      if ( !v58 )
        return 0;
      v76 = v58;
      break;
    case 5u:
      if ( *((_WORD *)a1 + 9) != 26 )
        goto LABEL_19;
      v46 = sub_14A9030(&v85, (__int64)a1);
      v47 = v75;
      if ( !v46 )
      {
        v16 = *((_BYTE *)a1 + 16);
LABEL_112:
        if ( v16 == 35 )
          goto LABEL_113;
        goto LABEL_84;
      }
      break;
    case 0x23u:
LABEL_113:
      if ( !*(a1 - 6) )
        return 0;
      v75 = (__int64 *)*(a1 - 6);
      if ( !*(a1 - 3) )
        return 0;
      v76 = (__int64 *)*(a1 - 3);
      v16 = 35;
      goto LABEL_116;
    default:
LABEL_85:
      if ( v16 <= 0x17u )
        return 0;
      goto LABEL_86;
  }
  if ( (unsigned __int8)sub_14BDDF0(v47, 1, v17, v7) )
    return 1;
  if ( (unsigned __int8)sub_14BDDF0(v76, 1, v17, v7) )
    return 1;
  v86 = (__int64 **)v76;
  if ( sub_13D52E0((__int64)&v85, (__int64)v75) )
    return 1;
  v83[1] = (__int64)v75;
  result = sub_13D52E0((__int64)v83, (__int64)v76);
  if ( result )
    return 1;
  return result;
}
