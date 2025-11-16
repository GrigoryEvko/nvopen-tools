// Function: sub_22185C0
// Address: 0x22185c0
//
_QWORD *__fastcall sub_22185C0(
        __int64 a1,
        _QWORD *a2,
        int a3,
        _QWORD *a4,
        int a5,
        unsigned __int64 a6,
        _DWORD *a7,
        __int64 a8)
{
  unsigned __int64 v9; // r14
  unsigned __int64 v11; // rbp
  __int64 v13; // r13
  unsigned __int8 *v14; // rax
  unsigned __int64 v15; // rbx
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  char v19; // al
  char v20; // r14
  unsigned __int8 *v21; // rax
  unsigned __int64 v22; // rax
  unsigned __int64 i; // r14
  int v24; // eax
  char v25; // bp
  _BYTE *v26; // rax
  int v27; // eax
  unsigned __int8 *v28; // rax
  char v29; // al
  char v30; // al
  char v31; // al
  char v32; // r13
  int v33; // eax
  unsigned __int8 *v34; // rax
  unsigned __int64 v35; // rax
  int v36; // eax
  _BYTE *v37; // rax
  __int64 v38; // rbx
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rax
  bool v41; // zf
  char v42; // al
  char v43; // al
  char v45; // al
  int v46; // eax
  int v47; // edx
  int v48; // edx
  char v49; // al
  char v50; // r13
  int v51; // edx
  __int64 v52; // rbp
  char v53; // r14
  unsigned __int64 v54; // rdx
  unsigned __int64 v55; // rax
  char v56; // al
  char v57; // r13
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rdx
  int v60; // eax
  int v61; // ecx
  int v62; // r8d
  int v63; // r9d
  _BYTE *v64; // rax
  int v65; // eax
  int v66; // eax
  _BYTE *v67; // rax
  int v68; // eax
  __int64 v70; // [rsp+8h] [rbp-110h]
  __int64 *v71; // [rsp+10h] [rbp-108h]
  int v72; // [rsp+10h] [rbp-108h]
  _BYTE *s; // [rsp+18h] [rbp-100h]
  __int64 v74; // [rsp+20h] [rbp-F8h]
  unsigned __int64 v75; // [rsp+38h] [rbp-E0h]
  __int64 v76; // [rsp+48h] [rbp-D0h]
  char v77; // [rsp+50h] [rbp-C8h]
  unsigned __int64 v78; // [rsp+50h] [rbp-C8h]
  char v81; // [rsp+88h] [rbp-90h]
  unsigned __int8 v82; // [rsp+8Dh] [rbp-8Bh]
  char v83; // [rsp+8Eh] [rbp-8Ah]
  bool v84; // [rsp+8Fh] [rbp-89h]
  int v85; // [rsp+9Ch] [rbp-7Ch]
  _QWORD *v86; // [rsp+A0h] [rbp-78h] BYREF
  __int64 v87; // [rsp+A8h] [rbp-70h]
  _QWORD v88[2]; // [rsp+B0h] [rbp-68h] BYREF
  _QWORD *v89; // [rsp+C0h] [rbp-58h] BYREF
  unsigned __int64 v90; // [rsp+C8h] [rbp-50h]
  _QWORD v91[9]; // [rsp+D0h] [rbp-48h] BYREF

  LODWORD(v9) = a6 + 208;
  v11 = a6;
  v76 = sub_222F790(a6 + 208);
  v13 = sub_22091A0(&qword_4FD6880);
  v71 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(v11 + 208) + 24LL) + 8 * v13);
  v70 = *v71;
  if ( !*v71 )
  {
    v11 = sub_22077B0(0x70u);
    *(_DWORD *)(v11 + 8) = 0;
    *(_QWORD *)(v11 + 16) = 0;
    *(_QWORD *)(v11 + 24) = 0;
    *(_WORD *)(v11 + 32) = 0;
    *(_QWORD *)v11 = off_4A04860;
    *(_BYTE *)(v11 + 34) = 0;
    *(_QWORD *)(v11 + 40) = 0;
    *(_QWORD *)(v11 + 48) = 0;
    *(_QWORD *)(v11 + 56) = 0;
    *(_QWORD *)(v11 + 64) = 0;
    *(_QWORD *)(v11 + 72) = 0;
    *(_QWORD *)(v11 + 80) = 0;
    *(_QWORD *)(v11 + 88) = 0;
    *(_DWORD *)(v11 + 96) = 0;
    *(_BYTE *)(v11 + 111) = 0;
    sub_22303B0(v11, v9, 0, v61, v62, v63);
    sub_2209690(*(_QWORD *)(a6 + 208), (volatile signed __int32 *)v11, v13);
    v70 = *v71;
  }
  if ( *(_QWORD *)(v70 + 64) )
    v84 = *(_QWORD *)(v70 + 80) != 0;
  else
    v84 = 0;
  LOBYTE(v88[0]) = 0;
  v86 = v88;
  v87 = 0;
  if ( *(_BYTE *)(v70 + 32) )
    sub_2240E30(&v86, 32);
  v90 = 0;
  v89 = v91;
  LOBYTE(v91[0]) = 0;
  sub_2240E30(&v89, 32);
  v82 = 0;
  v74 = 0;
  v72 = 0;
  v85 = *(_DWORD *)(v70 + 96);
  v81 = 0;
  v75 = 0;
  v83 = 0;
  while ( 2 )
  {
    switch ( *((_BYTE *)&v85 + v74) )
    {
      case 0:
        LODWORD(v13) = 1;
        goto LABEL_28;
      case 1:
        LOBYTE(v9) = a3 == -1;
        LOBYTE(v11) = v9 & (a2 != 0);
        if ( (_BYTE)v11 )
        {
          if ( a2[2] >= a2[3] )
          {
            if ( (*(unsigned int (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2) == -1 )
              a2 = 0;
            else
              LODWORD(v11) = 0;
          }
          else
          {
            LODWORD(v11) = 0;
          }
        }
        else
        {
          LODWORD(v11) = v9;
        }
        v31 = a5 == -1;
        v32 = v31 & (a4 != 0);
        if ( v32 )
        {
          v31 = 0;
          if ( a4[2] >= a4[3] )
          {
            v41 = (*(unsigned int (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4) == -1;
            v31 = 0;
            if ( v41 )
            {
              v31 = v32;
              a4 = 0;
            }
          }
        }
        LODWORD(v13) = 0;
        if ( v31 == (_BYTE)v11 )
          goto LABEL_28;
        if ( !a2 || a3 != -1 )
        {
          LOBYTE(v33) = a3;
          goto LABEL_66;
        }
        v34 = (unsigned __int8 *)a2[2];
        if ( (unsigned __int64)v34 >= a2[3] )
        {
          v33 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
          if ( v33 == -1 )
            a2 = 0;
LABEL_66:
          LODWORD(v13) = 0;
          if ( (*(_BYTE *)(*(_QWORD *)(v76 + 48) + 2LL * (unsigned __int8)v33 + 1) & 0x20) == 0 )
            goto LABEL_28;
          v34 = (unsigned __int8 *)a2[2];
          if ( (unsigned __int64)v34 >= a2[3] )
          {
            (*(void (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
            goto LABEL_69;
          }
LABEL_68:
          a2[2] = v34 + 1;
LABEL_69:
          a3 = -1;
          LODWORD(v13) = 1;
          goto LABEL_28;
        }
        a3 = -1;
        if ( (*(_BYTE *)(*(_QWORD *)(v76 + 48) + 2LL * *v34 + 1) & 0x20) != 0 )
          goto LABEL_68;
LABEL_28:
        if ( v74 != 3 )
        {
          while ( 1 )
          {
            LOBYTE(v9) = a3 == -1;
            LOBYTE(v11) = v9 & (a2 != 0);
            if ( (_BYTE)v11 )
            {
              if ( a2[2] >= a2[3] )
              {
                if ( (*(unsigned int (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2) == -1 )
                  a2 = 0;
                else
                  LODWORD(v11) = 0;
              }
              else
              {
                LODWORD(v11) = 0;
              }
            }
            else
            {
              LODWORD(v11) = v9;
            }
            v29 = a5 == -1;
            v77 = v29 & (a4 != 0);
            if ( v77 )
            {
              v29 = 0;
              if ( a4[2] >= a4[3] )
              {
                v41 = (*(unsigned int (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4) == -1;
                v29 = 0;
                if ( v41 )
                {
                  v29 = v77;
                  a4 = 0;
                }
              }
            }
            if ( v29 == (_BYTE)v11 )
              goto LABEL_127;
            if ( a2 && a3 == -1 )
            {
              v28 = (unsigned __int8 *)a2[2];
              if ( (unsigned __int64)v28 < a2[3] )
              {
                if ( (*(_BYTE *)(*(_QWORD *)(v76 + 48) + 2LL * *v28 + 1) & 0x20) == 0 )
                {
                  a3 = -1;
                  v30 = v13 ^ 1;
                  goto LABEL_128;
                }
LABEL_48:
                a2[2] = v28 + 1;
                goto LABEL_49;
              }
              v27 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
              if ( v27 == -1 )
                a2 = 0;
            }
            else
            {
              LOBYTE(v27) = a3;
            }
            if ( (*(_BYTE *)(*(_QWORD *)(v76 + 48) + 2LL * (unsigned __int8)v27 + 1) & 0x20) == 0 )
              goto LABEL_127;
            v28 = (unsigned __int8 *)a2[2];
            if ( (unsigned __int64)v28 < a2[3] )
              goto LABEL_48;
            (*(void (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
LABEL_49:
            a3 = -1;
          }
        }
LABEL_29:
        if ( ((unsigned __int8)v13 & (v75 > 1)) != 0 )
        {
          if ( v83 )
            v22 = *(_QWORD *)(v70 + 72);
          else
            v22 = *(_QWORD *)(v70 + 56);
          s = (_BYTE *)v22;
          for ( i = 1; ; ++i )
          {
            LOBYTE(v13) = a3 == -1;
            LOBYTE(v11) = v13 & (a2 != 0);
            if ( (_BYTE)v11 )
            {
              v22 = a2[3];
              if ( a2[2] >= v22 )
              {
                v41 = (*(unsigned int (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2) == -1;
                LODWORD(v22) = 0;
                if ( v41 )
                  a2 = 0;
                else
                  LODWORD(v11) = 0;
              }
              else
              {
                LODWORD(v11) = 0;
              }
            }
            else
            {
              LODWORD(v11) = v13;
            }
            if ( a4 && a5 == -1 )
            {
              if ( a4[2] >= a4[3] )
              {
                v46 = (*(__int64 (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4);
                LOBYTE(v47) = v46 == -1;
                v48 = v11 ^ v47;
                if ( v46 == -1 )
                {
                  LODWORD(v11) = v48;
                  a4 = 0;
                }
              }
            }
            else
            {
              LOBYTE(v22) = a5 == -1;
              LODWORD(v11) = v22 ^ v11;
            }
            if ( i >= v75 || !(_BYTE)v11 )
            {
              if ( i == v75 )
              {
                if ( v90 <= 1 )
                  goto LABEL_170;
                goto LABEL_200;
              }
LABEL_112:
              v25 = v13 & (a2 != 0);
LABEL_113:
              *a7 |= 4u;
              if ( !v25 )
                goto LABEL_114;
LABEL_185:
              LOBYTE(v13) = 0;
              if ( a2[2] >= a2[3] && (*(unsigned int (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2) == -1 )
              {
                LOBYTE(v13) = v25;
                a2 = 0;
              }
              goto LABEL_114;
            }
            LOBYTE(v24) = a3;
            v25 = v13 & (a2 != 0);
            if ( v25 )
            {
              v26 = (_BYTE *)a2[2];
              if ( (unsigned __int64)v26 < a2[3] )
              {
                if ( *v26 != s[i] )
                {
                  LOBYTE(v13) = v13 & (a2 != 0);
                  goto LABEL_113;
                }
LABEL_43:
                v22 = (unsigned __int64)(v26 + 1);
                a2[2] = v22;
                goto LABEL_44;
              }
              v24 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
              if ( v24 == -1 )
                a2 = 0;
            }
            if ( s[i] != (_BYTE)v24 )
              goto LABEL_112;
            v26 = (_BYTE *)a2[2];
            if ( (unsigned __int64)v26 < a2[3] )
              goto LABEL_43;
            v22 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
LABEL_44:
            a3 = -1;
          }
        }
        if ( !(_BYTE)v13 )
          goto LABEL_111;
        if ( v90 <= 1 )
          goto LABEL_170;
LABEL_200:
        v58 = sub_2241A40(&v89, 48, 0);
        if ( v58 )
        {
          v59 = v90;
          if ( v58 != -1 )
            goto LABEL_202;
          v58 = v90 - 1;
          if ( !v90 )
          {
            v90 = 0;
            *(_BYTE *)v89 = 0;
            goto LABEL_170;
          }
          if ( v90 != 1 )
          {
LABEL_202:
            if ( v90 > v58 )
              v59 = v58;
            sub_2240CE0(&v89, 0, v59);
          }
        }
LABEL_170:
        if ( v83 && *(_BYTE *)v89 != 48 )
          sub_2240FD0(&v89, 0, 0, 1, 45);
        v52 = v87;
        if ( v87 )
        {
          v53 = v81;
          v54 = (unsigned __int64)v86;
          if ( !v82 )
            v53 = v72;
          v55 = 15;
          if ( v86 != v88 )
            v55 = v88[0];
          if ( v87 + 1 > v55 )
          {
            sub_2240BB0(&v86, v87, 0, 0, 1);
            v54 = (unsigned __int64)v86;
          }
          *(_BYTE *)(v54 + v52) = v53;
          v87 = v52 + 1;
          *((_BYTE *)v86 + v52 + 1) = 0;
          if ( !(unsigned __int8)sub_2255C00(*(_QWORD *)(v70 + 16), *(_QWORD *)(v70 + 24), &v86) )
            *a7 |= 4u;
        }
        LOBYTE(v13) = a3 == -1;
        v25 = v13 & (a2 != 0);
        if ( v82 && *(_DWORD *)(v70 + 88) != v72 )
          goto LABEL_113;
        sub_22415E0(a8, &v89);
        if ( v25 )
          goto LABEL_185;
LABEL_114:
        v43 = a5 == -1;
        if ( a4 )
        {
          if ( a5 == -1 )
          {
            v43 = 0;
            if ( a4[2] >= a4[3] )
              v43 = (*(unsigned int (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4) == -1;
          }
        }
        if ( v43 == (_BYTE)v13 )
          *a7 |= 2u;
        if ( v89 != v91 )
          j___libc_free_0((unsigned __int64)v89);
        if ( v86 != v88 )
          j___libc_free_0((unsigned __int64)v86);
        return a2;
      case 2:
        if ( (*(_BYTE *)(a6 + 25) & 2) != 0 )
          goto LABEL_80;
        v30 = (_DWORD)v74 == 0 || v75 > 1;
        if ( v30 )
          goto LABEL_80;
        if ( (_DWORD)v74 == 1 )
        {
          if ( v84 || (_BYTE)v85 == 3 || BYTE2(v85) == 1 )
          {
LABEL_80:
            v9 = 0;
            v35 = *(_QWORD *)(v70 + 48);
            v78 = v35;
            while ( 1 )
            {
              LOBYTE(v13) = a3 == -1;
              LOBYTE(v11) = v13 & (a2 != 0);
              if ( (_BYTE)v11 )
              {
                v35 = a2[3];
                if ( a2[2] >= v35 )
                {
                  v41 = (*(unsigned int (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2) == -1;
                  LODWORD(v35) = 0;
                  if ( v41 )
                    a2 = 0;
                  else
                    LODWORD(v11) = 0;
                }
                else
                {
                  LODWORD(v11) = 0;
                }
              }
              else
              {
                LODWORD(v11) = v13;
              }
              if ( a4 && a5 == -1 )
              {
                if ( a4[2] >= a4[3] )
                {
                  LOBYTE(v51) = (*(unsigned int (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4) == -1;
                  LODWORD(v11) = v51 ^ v11;
                  if ( (_BYTE)v51 )
                    a4 = 0;
                }
              }
              else
              {
                LOBYTE(v35) = a5 == -1;
                LODWORD(v11) = v35 ^ v11;
              }
              v30 = v11 & (v9 < v78);
              if ( !v30 )
              {
                if ( v9 == v78 )
                  break;
LABEL_157:
                if ( !v9 )
                {
                  LOBYTE(v13) = (*(_BYTE *)(a6 + 25) & 2) == 0;
                  v30 = (*(_BYTE *)(a6 + 25) & 2) != 0;
                  goto LABEL_128;
                }
LABEL_111:
                LOBYTE(v13) = a3 == -1;
                goto LABEL_112;
              }
              if ( a2 && a3 == -1 )
              {
                v37 = (_BYTE *)a2[2];
                if ( (unsigned __int64)v37 < a2[3] )
                {
                  if ( *(_BYTE *)(*(_QWORD *)(v70 + 40) + v9) != *v37 )
                  {
                    a3 = -1;
                    goto LABEL_157;
                  }
LABEL_92:
                  v35 = (unsigned __int64)(v37 + 1);
                  a2[2] = v35;
                  goto LABEL_93;
                }
                v36 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
                if ( v36 == -1 )
                  a2 = 0;
              }
              else
              {
                LOBYTE(v36) = a3;
              }
              if ( *(_BYTE *)(*(_QWORD *)(v70 + 40) + v9) != (_BYTE)v36 )
                goto LABEL_157;
              v37 = (_BYTE *)a2[2];
              if ( (unsigned __int64)v37 < a2[3] )
                goto LABEL_92;
              v35 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
LABEL_93:
              ++v9;
              a3 = -1;
            }
          }
LABEL_191:
          LODWORD(v13) = 1;
          goto LABEL_128;
        }
        LODWORD(v13) = 1;
        if ( (_DWORD)v74 != 2 )
          goto LABEL_128;
        if ( HIBYTE(v85) == 4 || HIBYTE(v85) == 3 && v84 )
          goto LABEL_80;
LABEL_130:
        ++v74;
        continue;
      case 3:
        if ( !*(_QWORD *)(v70 + 64) )
        {
          if ( !*(_QWORD *)(v70 + 80) )
            goto LABEL_72;
LABEL_192:
          LOBYTE(v11) = a3 == -1;
          if ( !a2 || a3 != -1 )
          {
            LODWORD(v9) = v11;
            goto LABEL_195;
          }
LABEL_237:
          LODWORD(v9) = 1;
          LODWORD(v11) = 0;
          if ( a2[2] >= a2[3] )
          {
            v65 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
            if ( v65 == -1 )
              a2 = 0;
            LOBYTE(v11) = v65 == -1;
          }
LABEL_195:
          v56 = a5 == -1;
          v57 = v56 & (a4 != 0);
          if ( v57 )
          {
            v56 = 0;
            if ( a4[2] >= a4[3] )
            {
              v41 = (*(unsigned int (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4) == -1;
              v56 = 0;
              if ( v41 )
              {
                v56 = v57;
                a4 = 0;
              }
            }
          }
          if ( (_BYTE)v11 == v56 )
          {
LABEL_197:
            if ( !*(_QWORD *)(v70 + 64) || *(_QWORD *)(v70 + 80) )
            {
LABEL_72:
              v30 = v84;
              LODWORD(v13) = !v84;
              goto LABEL_128;
            }
            goto LABEL_149;
          }
          if ( a2 && (_BYTE)v9 )
          {
            v67 = (_BYTE *)a2[2];
            if ( (unsigned __int64)v67 < a2[3] )
            {
              if ( *v67 != **(_BYTE **)(v70 + 72) )
                goto LABEL_197;
              v75 = *(_QWORD *)(v70 + 80);
              goto LABEL_251;
            }
            v68 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
            if ( v68 == -1 )
              a2 = 0;
          }
          else
          {
            LOBYTE(v68) = a3;
          }
          if ( **(_BYTE **)(v70 + 72) != (_BYTE)v68 )
            goto LABEL_197;
          v75 = *(_QWORD *)(v70 + 80);
          v67 = (_BYTE *)a2[2];
          if ( (unsigned __int64)v67 >= a2[3] )
          {
            (*(void (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
            goto LABEL_252;
          }
LABEL_251:
          a2[2] = v67 + 1;
LABEL_252:
          v83 = 1;
          v30 = 0;
          a3 = -1;
          LODWORD(v13) = 1;
          goto LABEL_128;
        }
        LOBYTE(v9) = a3 == -1;
        LOBYTE(v11) = v9 & (a2 != 0);
        if ( (_BYTE)v11 )
        {
          if ( a2[2] >= a2[3] )
          {
            if ( (*(unsigned int (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2) == -1 )
              a2 = 0;
            else
              LODWORD(v11) = 0;
          }
          else
          {
            LODWORD(v11) = 0;
          }
        }
        else
        {
          LODWORD(v11) = v9;
        }
        v49 = a5 == -1;
        v50 = v49 & (a4 != 0);
        if ( v50 )
        {
          v49 = 0;
          if ( a4[2] >= a4[3] )
          {
            v41 = (*(unsigned int (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4) == -1;
            v49 = 0;
            if ( v41 )
            {
              v49 = v50;
              a4 = 0;
            }
          }
        }
        if ( v49 == (_BYTE)v11 )
          goto LABEL_147;
        if ( a2 && a3 == -1 )
        {
          v64 = (_BYTE *)a2[2];
          if ( (unsigned __int64)v64 < a2[3] )
          {
            if ( **(_BYTE **)(v70 + 56) != *v64 )
            {
              if ( *(_QWORD *)(v70 + 80) )
                goto LABEL_237;
              goto LABEL_148;
            }
            v75 = *(_QWORD *)(v70 + 64);
            goto LABEL_244;
          }
          v66 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
          if ( v66 == -1 )
            a2 = 0;
        }
        else
        {
          LOBYTE(v66) = a3;
        }
        if ( **(_BYTE **)(v70 + 56) != (_BYTE)v66 )
        {
LABEL_147:
          if ( *(_QWORD *)(v70 + 80) )
            goto LABEL_192;
LABEL_148:
          if ( !*(_QWORD *)(v70 + 64) )
            goto LABEL_72;
LABEL_149:
          v83 = 1;
          v30 = 0;
          LODWORD(v13) = 1;
          goto LABEL_128;
        }
        v75 = *(_QWORD *)(v70 + 64);
        v64 = (_BYTE *)a2[2];
        if ( (unsigned __int64)v64 >= a2[3] )
        {
          (*(void (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
          goto LABEL_245;
        }
LABEL_244:
        a2[2] = v64 + 1;
LABEL_245:
        v30 = 0;
        a3 = -1;
        goto LABEL_191;
      case 4:
        while ( 2 )
        {
          LOBYTE(v13) = a3 == -1;
          LOBYTE(v11) = v13 & (a2 != 0);
          if ( (_BYTE)v11 )
          {
            if ( a2[2] >= a2[3] )
            {
              if ( (*(unsigned int (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2) == -1 )
                a2 = 0;
              else
                LODWORD(v11) = 0;
            }
            else
            {
              LODWORD(v11) = 0;
            }
          }
          else
          {
            LODWORD(v11) = v13;
          }
          v19 = a5 == -1;
          v20 = v19 & (a4 != 0);
          if ( v20 && (v19 = 0, a4[2] >= a4[3]) )
          {
            v41 = (*(unsigned int (__fastcall **)(_QWORD *))(*a4 + 72LL))(a4) == -1;
            v42 = 0;
            if ( v41 )
            {
              v42 = v20;
              a4 = 0;
            }
            if ( (_BYTE)v11 == v42 )
              goto LABEL_109;
          }
          else if ( (_BYTE)v11 == v19 )
          {
            goto LABEL_109;
          }
          if ( a2 && a3 == -1 )
          {
            v21 = (unsigned __int8 *)a2[2];
            if ( (unsigned __int64)v21 >= a2[3] )
            {
              v60 = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 72LL))(a2);
              LODWORD(v11) = -1;
              if ( v60 == -1 )
                a2 = 0;
              else
                LODWORD(v11) = v60;
            }
            else
            {
              LODWORD(v11) = *v21;
            }
          }
          else
          {
            LODWORD(v11) = a3;
          }
          v14 = (unsigned __int8 *)memchr((const void *)(v70 + 101), (char)v11, 0xAu);
          if ( v14 )
          {
            v15 = v90;
            v11 = v90 + 1;
            LODWORD(v13) = v14[(_QWORD)off_4CDFAD0 - 100 - v70];
            v16 = (unsigned __int64)v89;
            v17 = 15;
            if ( v89 != v91 )
              v17 = v91[0];
            if ( v11 > v17 )
            {
              sub_2240BB0(&v89, v90, 0, 0, 1);
              v16 = (unsigned __int64)v89;
            }
            *(_BYTE *)(v16 + v15) = v13;
            ++v72;
            v90 = v11;
            *((_BYTE *)v89 + v15 + 1) = 0;
LABEL_16:
            v18 = a2[2];
            if ( v18 < a2[3] )
            {
LABEL_17:
              a2[2] = v18 + 1;
LABEL_18:
              a3 = -1;
              continue;
            }
LABEL_104:
            (*(void (__fastcall **)(_QWORD *))(*a2 + 80LL))(a2);
            goto LABEL_18;
          }
          break;
        }
        LOBYTE(v13) = v82 | (*(_BYTE *)(v70 + 33) != (unsigned __int8)v11);
        if ( (_BYTE)v13 )
        {
          if ( !*(_BYTE *)(v70 + 32) )
          {
LABEL_110:
            if ( !v90 )
              goto LABEL_111;
LABEL_127:
            v30 = v13 ^ 1;
LABEL_128:
            if ( (int)v74 + 1 > 3 || v30 )
              goto LABEL_29;
            goto LABEL_130;
          }
          if ( *(_BYTE *)(v70 + 34) != (_BYTE)v11 )
          {
            LODWORD(v13) = *(unsigned __int8 *)(v70 + 32);
            goto LABEL_110;
          }
          if ( v82 )
          {
            LODWORD(v13) = v82;
            goto LABEL_110;
          }
          if ( !v72 )
          {
            LODWORD(v13) = 0;
            goto LABEL_110;
          }
          v38 = v87;
          v39 = (unsigned __int64)v86;
          LODWORD(v13) = v72;
          v40 = 15;
          if ( v86 != v88 )
            v40 = v88[0];
          LODWORD(v11) = v87 + 1;
          if ( v87 + 1 > v40 )
          {
            sub_2240BB0(&v86, v87, 0, 0, 1);
            v39 = (unsigned __int64)v86;
          }
          *(_BYTE *)(v39 + v38) = v72;
          v87 = v38 + 1;
          v72 = 0;
          *((_BYTE *)v86 + v38 + 1) = 0;
          v18 = a2[2];
          if ( v18 < a2[3] )
            goto LABEL_17;
          goto LABEL_104;
        }
        if ( *(int *)(v70 + 88) > 0 )
        {
          v45 = v72;
          v82 = 1;
          v72 = 0;
          v81 = v45;
          goto LABEL_16;
        }
        v82 = 0;
LABEL_109:
        LODWORD(v13) = 1;
        goto LABEL_110;
      default:
        v30 = 0;
        LODWORD(v13) = 1;
        goto LABEL_128;
    }
  }
}
