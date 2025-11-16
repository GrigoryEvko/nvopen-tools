// Function: sub_995040
// Address: 0x995040
//
__int64 __fastcall sub_995040(int a1, __int64 a2, unsigned __int8 *a3)
{
  unsigned int v5; // edx
  unsigned int v6; // r13d
  unsigned __int64 v8; // rax
  unsigned int v9; // esi
  unsigned __int64 *v10; // rdx
  __int64 v11; // rax
  unsigned __int8 v12; // r12
  unsigned __int8 *v13; // rax
  unsigned __int8 *v14; // rax
  __int64 v15; // rax
  __int64 v16; // r12
  char v17; // r8
  unsigned __int8 v18; // al
  __int64 v19; // rdx
  __int64 v20; // r12
  __int64 v21; // rcx
  __int64 v22; // r15
  __int16 v23; // ax
  __int64 v24; // rcx
  int v25; // edx
  unsigned __int8 *v26; // rax
  char v27; // al
  unsigned __int8 v28; // dl
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // rsi
  unsigned __int64 **v33; // r15
  unsigned int v34; // r12d
  unsigned __int64 v35; // rax
  __int64 v36; // rdx
  unsigned __int8 *v37; // rcx
  unsigned __int8 *v38; // r15
  unsigned __int8 *v39; // rax
  unsigned __int8 *v40; // rsi
  __int16 v41; // ax
  int v42; // eax
  __int64 v43; // rax
  unsigned __int8 *v44; // rcx
  unsigned __int8 *v45; // rax
  unsigned __int8 v46; // [rsp+8h] [rbp-B8h]
  unsigned __int8 *v47; // [rsp+8h] [rbp-B8h]
  __int64 v48; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v49; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v50; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v51; // [rsp+28h] [rbp-98h] BYREF
  __int64 *v52; // [rsp+30h] [rbp-90h]
  __int64 *v53; // [rsp+38h] [rbp-88h] BYREF
  char v54; // [rsp+40h] [rbp-80h]
  __int64 *v55; // [rsp+48h] [rbp-78h]
  __int64 *v56; // [rsp+50h] [rbp-70h] BYREF
  char v57; // [rsp+58h] [rbp-68h]
  unsigned __int8 *v58; // [rsp+60h] [rbp-60h] BYREF
  __int64 *v59; // [rsp+68h] [rbp-58h] BYREF
  char v60; // [rsp+70h] [rbp-50h]
  __int64 v61; // [rsp+78h] [rbp-48h]
  __int64 *v62; // [rsp+80h] [rbp-40h] BYREF
  char v63; // [rsp+88h] [rbp-38h]

  LOBYTE(v5) = sub_B535D0() & (a2 == (_QWORD)a3);
  v6 = v5;
  if ( (_BYTE)v5 )
    return v6;
  if ( a1 == 37 )
  {
    LODWORD(v11) = *a3;
    if ( (_BYTE)v11 == 42 )
    {
      if ( a2 == *((_QWORD *)a3 - 8) || a2 == *((_QWORD *)a3 - 4) )
      {
        LODWORD(v11) = (a3[1] & 2) != 0;
        if ( (a3[1] & 2) != 0 )
          return (unsigned int)v11;
      }
      goto LABEL_16;
    }
    if ( (_BYTE)v11 == 58 )
    {
      if ( a2 == *((_QWORD *)a3 - 8) || a2 == *((_QWORD *)a3 - 4) )
        return 1;
    }
    else if ( (unsigned __int8)v11 > 0x1Cu )
    {
      if ( (_BYTE)v11 == 85 )
      {
        v11 = *((_QWORD *)a3 - 4);
        if ( v11 )
        {
          if ( !*(_BYTE *)v11
            && *(_QWORD *)(v11 + 24) == *((_QWORD *)a3 + 10)
            && (*(_BYTE *)(v11 + 33) & 0x20) != 0
            && *(_DWORD *)(v11 + 36) == 365 )
          {
            if ( a2 == *(_QWORD *)&a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)] )
              return 1;
            v11 = 32 * (1LL - (*((_DWORD *)a3 + 1) & 0x7FFFFFF));
            if ( a2 == *(_QWORD *)&a3[v11] )
              return 1;
          }
        }
        goto LABEL_16;
      }
      if ( (_BYTE)v11 == 86 )
      {
        v11 = *((_QWORD *)a3 - 12);
        if ( *(_BYTE *)v11 == 82 )
        {
          v19 = *((_QWORD *)a3 - 8);
          v20 = *(_QWORD *)(v11 - 64);
          v21 = *((_QWORD *)a3 - 4);
          v22 = *(_QWORD *)(v11 - 32);
          if ( v19 == v20 && v21 == v22 )
          {
            v23 = *(_WORD *)(v11 + 2);
            goto LABEL_46;
          }
          if ( v19 == v22 && v21 == v20 )
          {
            v23 = *(_WORD *)(v11 + 2);
            if ( v19 != v20 )
            {
              LODWORD(v11) = sub_B52870(v23 & 0x3F);
              goto LABEL_47;
            }
LABEL_46:
            LODWORD(v11) = v23 & 0x3F;
LABEL_47:
            LODWORD(v11) = v11 - 34;
            if ( (unsigned int)v11 <= 1 && (a2 == v20 || a2 == v22) )
              return 1;
          }
        }
      }
    }
LABEL_16:
    v12 = *(_BYTE *)a2;
    if ( *(_BYTE *)a2 != 55 )
    {
      v58 = a3;
      v59 = &v48;
      v60 = 0;
      if ( v12 != 48 )
        goto LABEL_18;
      v26 = *(unsigned __int8 **)(a2 - 64);
      if ( a3 != v26 || !v26 )
        goto LABEL_23;
      v46 = sub_991580((__int64)&v59, *(_QWORD *)(a2 - 32));
      if ( !v46 )
      {
LABEL_69:
        v12 = *(_BYTE *)a2;
LABEL_18:
        if ( v12 == 57 )
        {
          v13 = *(unsigned __int8 **)(a2 - 64);
          if ( a3 == v13 && v13 )
            return 1;
          v14 = *(unsigned __int8 **)(a2 - 32);
          if ( v14 )
          {
            if ( a3 == v14 )
              return 1;
          }
LABEL_23:
          v54 = 0;
          v52 = &v49;
          v53 = &v50;
          v55 = &v49;
          v56 = &v50;
          v57 = 0;
          if ( !(unsigned __int8)sub_987880((unsigned __int8 *)a2) )
            goto LABEL_24;
          goto LABEL_126;
        }
        if ( v12 <= 0x1Cu )
        {
          v54 = 0;
          v52 = &v49;
          v53 = &v50;
          v55 = &v49;
          v56 = &v50;
          v57 = 0;
          if ( !(unsigned __int8)sub_987880((unsigned __int8 *)a2) )
            return v6;
LABEL_95:
          if ( *(_WORD *)(a2 + 2) != 13 )
            return v6;
LABEL_96:
          if ( (*(_BYTE *)(a2 + 1) & 2) == 0 || !*(_QWORD *)(a2 - 64) )
            return v6;
          v32 = *(_QWORD *)(a2 - 32);
          v49 = *(_QWORD *)(a2 - 64);
          if ( (unsigned __int8)sub_991580((__int64)&v53, v32) )
          {
LABEL_28:
            v16 = v49;
            v60 = 0;
            v59 = &v51;
            v58 = (unsigned __int8 *)v49;
            v61 = v49;
            v62 = &v51;
            v63 = 0;
            v17 = sub_987880(a3);
            v18 = *a3;
            if ( v17 )
            {
              if ( v18 <= 0x1Cu )
              {
                if ( *((_WORD *)a3 + 1) != 13 )
                  return v6;
              }
              else if ( v18 != 42 )
              {
                goto LABEL_29;
              }
              if ( (a3[1] & 2) == 0 || *((_QWORD *)a3 - 8) != v16 )
                return v6;
              if ( (unsigned __int8)sub_991580((__int64)&v59, *((_QWORD *)a3 - 4)) )
              {
LABEL_33:
                LOBYTE(v6) = (int)sub_C49970(v50, v51) <= 0;
                return v6;
              }
              v18 = *a3;
            }
LABEL_29:
            if ( v18 != 58
              || (a3[1] & 2) == 0
              || *((_QWORD *)a3 - 8) != v61
              || !(unsigned __int8)sub_991580((__int64)&v62, *((_QWORD *)a3 - 4)) )
            {
              return v6;
            }
            goto LABEL_33;
          }
          v12 = *(_BYTE *)a2;
LABEL_24:
          if ( v12 != 58 )
            return v6;
          if ( (*(_BYTE *)(a2 + 1) & 2) == 0 )
            return v6;
          v15 = *(_QWORD *)(a2 - 64);
          if ( !v15 )
            return v6;
          *v55 = v15;
          if ( !(unsigned __int8)sub_991580((__int64)&v56, *(_QWORD *)(a2 - 32)) )
            return v6;
          goto LABEL_28;
        }
        if ( v12 == 85 )
        {
          v43 = *(_QWORD *)(a2 - 32);
          if ( !v43
            || *(_BYTE *)v43
            || *(_QWORD *)(v43 + 24) != *(_QWORD *)(a2 + 80)
            || (*(_BYTE *)(v43 + 33) & 0x20) == 0
            || *(_DWORD *)(v43 + 36) != 366 )
          {
            goto LABEL_23;
          }
          v44 = *(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
          v45 = *(unsigned __int8 **)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
          if ( v45 )
          {
            if ( a3 == v44 || a3 == v45 )
              return 1;
          }
          else if ( a3 == v44 )
          {
            return 1;
          }
          goto LABEL_137;
        }
        if ( v12 != 86 )
          goto LABEL_23;
        v36 = *(_QWORD *)(a2 - 96);
        if ( *(_BYTE *)v36 == 82 )
        {
          v37 = *(unsigned __int8 **)(a2 - 64);
          v38 = *(unsigned __int8 **)(v36 - 64);
          v39 = *(unsigned __int8 **)(a2 - 32);
          v40 = *(unsigned __int8 **)(v36 - 32);
          if ( v37 == v38 && v39 == v40 )
          {
            v41 = *(_WORD *)(v36 + 2);
            goto LABEL_120;
          }
          if ( v37 == v40 && v39 == v38 )
          {
            v41 = *(_WORD *)(v36 + 2);
            if ( v37 != v38 )
            {
              v47 = *(unsigned __int8 **)(v36 - 32);
              v42 = sub_B52870(*(_WORD *)(v36 + 2) & 0x3F);
              v40 = v47;
              goto LABEL_121;
            }
LABEL_120:
            v42 = v41 & 0x3F;
LABEL_121:
            if ( (unsigned int)(v42 - 36) <= 1 && (a3 == v38 || a3 == v40) )
              return 1;
            v54 = 0;
            v12 = *(_BYTE *)a2;
            v52 = &v49;
            v53 = &v50;
            v55 = &v49;
            v56 = &v50;
            v57 = 0;
            if ( !(unsigned __int8)sub_987880((unsigned __int8 *)a2) )
              goto LABEL_24;
            if ( v12 <= 0x1Cu )
              goto LABEL_95;
LABEL_126:
            if ( v12 != 42 )
              goto LABEL_24;
            goto LABEL_96;
          }
        }
LABEL_137:
        v54 = 0;
        v52 = &v49;
        v53 = &v50;
        v55 = &v49;
        v56 = &v50;
        v57 = 0;
        if ( !(unsigned __int8)sub_987880((unsigned __int8 *)a2) )
          return v6;
        goto LABEL_126;
      }
      v33 = (unsigned __int64 **)v48;
      v34 = *(_DWORD *)(v48 + 8);
      if ( v34 > 0x40 )
      {
        if ( v34 - (unsigned int)sub_C444A0(v48) > 0x40 )
          return v46;
        v35 = **v33;
      }
      else
      {
        v35 = *(_QWORD *)v48;
      }
      if ( v35 <= 1 )
        goto LABEL_69;
      return v46;
    }
    LOBYTE(v11) = a3 == *(unsigned __int8 **)(a2 - 64) && *(_QWORD *)(a2 - 64) != 0;
    if ( !(_BYTE)v11 )
      goto LABEL_23;
    return (unsigned int)v11;
  }
  if ( a1 != 41 )
    return v6;
  v8 = *a3;
  v52 = (__int64 *)a2;
  v53 = &v48;
  v54 = 0;
  if ( (unsigned __int8)v8 <= 0x1Cu )
  {
    if ( (_BYTE)v8 != 5
      || (v25 = *((unsigned __int16 *)a3 + 1), (*((_WORD *)a3 + 1) & 0xFFF7) != 0x11) && (v25 & 0xFFFD) != 0xD )
    {
LABEL_35:
      v58 = (unsigned __int8 *)a2;
      if ( !(unsigned __int8)sub_994E00((__int64 *)&v58, (char *)a3) )
      {
        v58 = a3;
        v6 = sub_994F20((__int64 *)&v58, (char *)a2);
        if ( !(_BYTE)v6 )
        {
          v54 = 0;
          v52 = &v49;
          v55 = &v49;
          v53 = &v50;
          v56 = &v50;
          v57 = 0;
          v27 = sub_987880((unsigned __int8 *)a2);
          v28 = *(_BYTE *)a2;
          if ( a2 && v27 )
          {
            if ( v28 <= 0x1Cu )
            {
              if ( *(_WORD *)(a2 + 2) != 13 )
                return v6;
            }
            else if ( v28 != 42 )
            {
              goto LABEL_79;
            }
            if ( (*(_BYTE *)(a2 + 1) & 4) == 0 )
              return v6;
            if ( *(_QWORD *)(a2 - 64) )
            {
              v29 = *(_QWORD *)(a2 - 32);
              v49 = *(_QWORD *)(a2 - 64);
              if ( (unsigned __int8)sub_991580((__int64)&v53, v29) )
                goto LABEL_83;
              v28 = *(_BYTE *)a2;
            }
          }
LABEL_79:
          if ( v28 != 58 )
            return v6;
          if ( (*(_BYTE *)(a2 + 1) & 2) == 0 )
            return v6;
          v30 = *(_QWORD *)(a2 - 64);
          if ( !v30 )
            return v6;
          *v55 = v30;
          if ( !(unsigned __int8)sub_991580((__int64)&v56, *(_QWORD *)(a2 - 32)) )
            return v6;
LABEL_83:
          v31 = v49;
          v60 = 0;
          v59 = &v51;
          v58 = (unsigned __int8 *)v49;
          v61 = v49;
          v62 = &v51;
          v63 = 0;
          if ( (unsigned __int8)sub_987880(a3) )
          {
            if ( *a3 <= 0x1Cu )
            {
              if ( *((_WORD *)a3 + 1) != 13 )
                return v6;
LABEL_86:
              if ( (a3[1] & 4) == 0 )
                return v6;
              if ( *((_QWORD *)a3 - 8) == v31 && (unsigned __int8)sub_991580((__int64)&v59, *((_QWORD *)a3 - 4)) )
                goto LABEL_92;
              goto LABEL_88;
            }
            if ( *a3 == 42 )
              goto LABEL_86;
          }
LABEL_88:
          if ( *a3 != 58
            || (a3[1] & 2) == 0
            || *((_QWORD *)a3 - 8) != v61
            || !(unsigned __int8)sub_991580((__int64)&v62, *((_QWORD *)a3 - 4)) )
          {
            return v6;
          }
LABEL_92:
          LOBYTE(v6) = (int)sub_C4C880(v50, v51) <= 0;
          return v6;
        }
      }
      return 1;
    }
  }
  else
  {
    if ( (unsigned __int8)v8 > 0x36u )
      goto LABEL_7;
    v24 = 0x40540000000000LL;
    v25 = (unsigned __int8)v8 - 29;
    if ( !_bittest64(&v24, v8) )
      goto LABEL_35;
  }
  if ( v25 != 13 )
    goto LABEL_35;
  if ( (a3[1] & 4) != 0 && a2 == *((_QWORD *)a3 - 8) )
  {
    if ( (unsigned __int8)sub_991580((__int64)&v53, *((_QWORD *)a3 - 4)) )
      goto LABEL_10;
    LOBYTE(v8) = *a3;
  }
LABEL_7:
  v58 = (unsigned __int8 *)a2;
  v59 = &v48;
  v60 = 0;
  if ( (_BYTE)v8 != 58 || a2 != *((_QWORD *)a3 - 8) || !(unsigned __int8)sub_991580((__int64)&v59, *((_QWORD *)a3 - 4)) )
    goto LABEL_35;
LABEL_10:
  v9 = *(_DWORD *)(v48 + 8);
  if ( v9 > 0x40 )
    v10 = *(unsigned __int64 **)(*(_QWORD *)v48 + 8LL * ((v9 - 1) >> 6));
  else
    v10 = *(unsigned __int64 **)v48;
  LOBYTE(v6) = ((unsigned __int64)v10 & (1LL << ((unsigned __int8)v9 - 1))) == 0;
  return v6;
}
