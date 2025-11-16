// Function: sub_1183020
// Address: 0x1183020
//
unsigned __int8 *__fastcall sub_1183020(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  _BYTE *v3; // rdi
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  unsigned int v7; // eax
  __int64 v8; // rsi
  unsigned int v9; // r8d
  int v10; // r15d
  int v11; // ecx
  unsigned int v12; // r9d
  unsigned int v13; // ecx
  __int64 v14; // rdx
  unsigned int v15; // r9d
  char v16; // cl
  unsigned int v17; // edx
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rax
  char v22; // al
  char v23; // al
  int v24; // eax
  __int64 v25; // rdi
  int v26; // eax
  bool v27; // al
  char v28; // al
  char v29; // si
  char v30; // al
  int v31; // r9d
  int v32; // ecx
  _QWORD *v33; // rax
  __int64 v34; // rdi
  int v35; // eax
  bool v36; // al
  char v37; // al
  _BYTE **v38; // rax
  char v39; // al
  __int64 v40; // rax
  char v41; // al
  __int64 v42; // rax
  char v43; // al
  int v44; // r14d
  unsigned int v45; // esi
  __int64 v46; // r15
  int v47; // eax
  bool v48; // al
  __int64 v49; // rax
  _QWORD *v50; // rax
  unsigned int v51; // r15d
  int v52; // eax
  bool v53; // al
  bool v54; // al
  _BYTE *v55; // rax
  __int64 v56; // rdx
  _BYTE *v57; // rax
  unsigned int i; // ebx
  __int64 v59; // rax
  __int64 v60; // rsi
  bool v61; // r10
  __int64 v62; // rax
  int v63; // eax
  unsigned int v64; // eax
  unsigned int v65; // esi
  _QWORD *v66; // rax
  unsigned int v67; // [rsp+8h] [rbp-E8h]
  unsigned int v68; // [rsp+10h] [rbp-E0h]
  unsigned int v69; // [rsp+10h] [rbp-E0h]
  _QWORD *v70; // [rsp+10h] [rbp-E0h]
  bool v71; // [rsp+10h] [rbp-E0h]
  char v72; // [rsp+18h] [rbp-D8h]
  int v73; // [rsp+18h] [rbp-D8h]
  int v74; // [rsp+18h] [rbp-D8h]
  bool v75; // [rsp+1Ch] [rbp-D4h]
  int v76; // [rsp+1Ch] [rbp-D4h]
  unsigned int v77; // [rsp+1Ch] [rbp-D4h]
  unsigned int v78; // [rsp+1Ch] [rbp-D4h]
  unsigned int v79; // [rsp+20h] [rbp-D0h]
  unsigned int v80; // [rsp+20h] [rbp-D0h]
  int v81; // [rsp+20h] [rbp-D0h]
  unsigned int v82; // [rsp+20h] [rbp-D0h]
  unsigned int v83; // [rsp+20h] [rbp-D0h]
  int v84; // [rsp+20h] [rbp-D0h]
  char v85; // [rsp+20h] [rbp-D0h]
  unsigned int v86; // [rsp+20h] [rbp-D0h]
  unsigned int v87; // [rsp+20h] [rbp-D0h]
  unsigned int v88; // [rsp+28h] [rbp-C8h]
  char v89; // [rsp+28h] [rbp-C8h]
  char v90; // [rsp+28h] [rbp-C8h]
  char v91; // [rsp+28h] [rbp-C8h]
  unsigned int v92; // [rsp+28h] [rbp-C8h]
  char v93; // [rsp+28h] [rbp-C8h]
  __int64 v94; // [rsp+28h] [rbp-C8h]
  __int64 v95; // [rsp+28h] [rbp-C8h]
  int v96; // [rsp+28h] [rbp-C8h]
  unsigned int v97; // [rsp+28h] [rbp-C8h]
  int v98; // [rsp+28h] [rbp-C8h]
  _BYTE *v100; // [rsp+38h] [rbp-B8h]
  char v101; // [rsp+38h] [rbp-B8h]
  char v102; // [rsp+38h] [rbp-B8h]
  char v103; // [rsp+38h] [rbp-B8h]
  __int64 v104; // [rsp+48h] [rbp-A8h] BYREF
  __int64 v105; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v106; // [rsp+58h] [rbp-98h] BYREF
  char v107; // [rsp+5Ch] [rbp-94h]
  unsigned int v108; // [rsp+60h] [rbp-90h] BYREF
  char v109; // [rsp+64h] [rbp-8Ch]
  unsigned int v110; // [rsp+68h] [rbp-88h]
  int v111; // [rsp+6Ch] [rbp-84h]
  _QWORD *v112[2]; // [rsp+70h] [rbp-80h] BYREF
  unsigned int *v113; // [rsp+80h] [rbp-70h] BYREF
  _QWORD *v114; // [rsp+88h] [rbp-68h]
  _QWORD *v115; // [rsp+90h] [rbp-60h]
  __int64 *v116; // [rsp+98h] [rbp-58h] BYREF
  __int16 v117; // [rsp+A0h] [rbp-50h]
  __int64 *v118; // [rsp+A8h] [rbp-48h] BYREF
  char v119; // [rsp+B0h] [rbp-40h]

  v2 = *(_QWORD *)(a2 - 64);
  v3 = *(_BYTE **)(a2 - 96);
  v100 = *(_BYTE **)(a2 - 32);
  if ( *v3 != 82 )
    return 0;
  v4 = (_QWORD *)*((_QWORD *)v3 - 8);
  if ( !v4 )
    return 0;
  v5 = (_QWORD *)*((_QWORD *)v3 - 4);
  if ( !v5 )
    return 0;
  v7 = sub_B53900((__int64)v3);
  v8 = v4[1];
  v9 = v7;
  v10 = v7;
  v11 = *(unsigned __int8 *)(v8 + 8);
  if ( (unsigned int)(v11 - 17) <= 1 )
    LOBYTE(v11) = *(_BYTE *)(**(_QWORD **)(v8 + 16) + 8LL);
  if ( (_BYTE)v11 != 12 )
    return 0;
  v12 = v7 & 0xFFFFFFFB;
  if ( (v7 & 0xFFFFFFFB) == 0x22 )
  {
    v82 = v7;
    v113 = 0;
    v30 = sub_995B10((_QWORD **)&v113, v2);
    v12 = 34;
    v9 = v82;
    if ( v30 )
    {
LABEL_33:
      v31 = sub_B52F50(v9);
      v32 = v31 - 36;
      v10 = v31;
      v33 = v4;
      v12 = v31 & 0xFFFFFFFB;
      v4 = v5;
      v13 = v32 & 0xFFFFFFFB;
      v5 = v33;
      goto LABEL_9;
    }
  }
  v13 = (v9 - 36) & 0xFFFFFFFB;
  if ( !v13 )
  {
    if ( *(_BYTE *)v2 == 17 )
    {
      if ( *(_DWORD *)(v2 + 32) > 0x40u )
      {
        v68 = v9;
        v34 = v2 + 24;
        v76 = *(_DWORD *)(v2 + 32);
        v83 = 0;
        v92 = v12;
LABEL_37:
        v35 = sub_C444A0(v34);
        v12 = v92;
        v13 = v83;
        v9 = v68;
        v36 = v76 - 1 == v35;
        goto LABEL_38;
      }
      v36 = *(_QWORD *)(v2 + 24) == 1;
    }
    else
    {
      v56 = *(_QWORD *)(v2 + 8);
      v86 = v9;
      v95 = v56;
      if ( (unsigned int)*(unsigned __int8 *)(v56 + 8) - 17 > 1 || *(_BYTE *)v2 > 0x15u )
        goto LABEL_9;
      v69 = (v9 - 36) & 0xFFFFFFFB;
      v77 = v12;
      v57 = sub_AD7630(v2, 0, v56);
      v12 = v77;
      v13 = v69;
      v9 = v86;
      if ( !v57 || *v57 != 17 )
      {
        if ( *(_BYTE *)(v95 + 8) == 17 )
        {
          v74 = *(_DWORD *)(v95 + 32);
          if ( v74 )
          {
            v67 = v86;
            v60 = 0;
            v61 = 0;
            while ( 1 )
            {
              v87 = v13;
              v97 = v12;
              v71 = v61;
              v62 = sub_AD69F0((unsigned __int8 *)v2, v60);
              v12 = v97;
              v13 = v87;
              if ( !v62 )
                break;
              v61 = v71;
              if ( *(_BYTE *)v62 != 13 )
              {
                if ( *(_BYTE *)v62 != 17 )
                  break;
                if ( *(_DWORD *)(v62 + 32) <= 0x40u )
                {
                  v61 = *(_QWORD *)(v62 + 24) == 1;
                }
                else
                {
                  v78 = v97;
                  v98 = *(_DWORD *)(v62 + 32);
                  v63 = sub_C444A0(v62 + 24);
                  v12 = v78;
                  v13 = v87;
                  v61 = v98 - 1 == v63;
                }
                if ( !v61 )
                  break;
              }
              v60 = (unsigned int)(v60 + 1);
              if ( v74 == (_DWORD)v60 )
              {
                v9 = v67;
                if ( v61 )
                  goto LABEL_33;
                goto LABEL_9;
              }
            }
          }
        }
        goto LABEL_9;
      }
      if ( *((_DWORD *)v57 + 8) > 0x40u )
      {
        v68 = v86;
        v34 = (__int64)(v57 + 24);
        v76 = *((_DWORD *)v57 + 8);
        v83 = v13;
        v92 = v12;
        goto LABEL_37;
      }
      v36 = *((_QWORD *)v57 + 3) == 1;
    }
LABEL_38:
    if ( !v36 )
      goto LABEL_9;
    goto LABEL_33;
  }
LABEL_9:
  v79 = v13;
  v88 = v12;
  v106 = 42;
  v75 = sub_B532B0(v10);
  v15 = v88;
  v107 = 0;
  if ( v79 )
    goto LABEL_10;
  v112[0] = 0;
  v22 = sub_995B10(v112, v2);
  v15 = v88;
  if ( !v22 )
    goto LABEL_10;
  v114 = v4;
  v113 = &v106;
  v115 = v5;
  if ( *v100 != 68 )
    goto LABEL_10;
  v23 = sub_1181140((__int64 *)&v113, *((_BYTE **)v100 - 4));
  v15 = v88;
  v16 = v23;
  if ( !v23 || v106 != 33 && (v80 = v88, v89 = v23, v24 = sub_B52F50(v106), v16 = v89, v15 = v80, v24 != v10) )
LABEL_10:
    v16 = 0;
  if ( v15 != 34 )
    goto LABEL_12;
  if ( *(_BYTE *)v2 == 17 )
  {
    if ( *(_DWORD *)(v2 + 32) > 0x40u )
    {
      v81 = *(_DWORD *)(v2 + 32);
      v25 = v2 + 24;
      v90 = v16;
LABEL_25:
      v26 = sub_C444A0(v25);
      v16 = v90;
      v27 = v81 - 1 == v26;
      goto LABEL_26;
    }
    v27 = *(_QWORD *)(v2 + 24) == 1;
    goto LABEL_26;
  }
  v94 = *(_QWORD *)(v2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v94 + 8) - 17 <= 1 && *(_BYTE *)v2 <= 0x15u )
  {
    v85 = v16;
    v55 = sub_AD7630(v2, 0, v14);
    v16 = v85;
    if ( !v55 || *v55 != 17 )
    {
      if ( *(_BYTE *)(v94 + 8) == 17 )
      {
        v96 = *(_DWORD *)(v94 + 32);
        if ( v96 )
        {
          v72 = 0;
          v70 = v4;
          for ( i = 0; i != v96; ++i )
          {
            v59 = sub_AD69F0((unsigned __int8 *)v2, i);
            if ( !v59 )
            {
LABEL_97:
              v16 = v85;
              v4 = v70;
              goto LABEL_12;
            }
            if ( *(_BYTE *)v59 != 13 )
            {
              if ( *(_BYTE *)v59 != 17 )
                goto LABEL_97;
              if ( *(_DWORD *)(v59 + 32) <= 0x40u )
              {
                if ( *(_QWORD *)(v59 + 24) != 1 )
                  goto LABEL_97;
              }
              else
              {
                v73 = *(_DWORD *)(v59 + 32);
                if ( (unsigned int)sub_C444A0(v59 + 24) != v73 - 1 )
                  goto LABEL_97;
              }
              v72 = 1;
            }
          }
          v16 = v85;
          v4 = v70;
          if ( v72 )
            goto LABEL_27;
        }
      }
      goto LABEL_12;
    }
    if ( *((_DWORD *)v55 + 8) > 0x40u )
    {
      v81 = *((_DWORD *)v55 + 8);
      v25 = (__int64)(v55 + 24);
      v90 = v16;
      goto LABEL_25;
    }
    v27 = *((_QWORD *)v55 + 3) == 1;
LABEL_26:
    if ( v27 )
    {
LABEL_27:
      v114 = v4;
      v113 = &v106;
      v115 = v5;
      if ( *v100 == 69 )
      {
        v91 = v16;
        v28 = sub_1181140((__int64 *)&v113, *((_BYTE **)v100 - 4));
        v16 = v91;
        v29 = v28;
        if ( v28 )
        {
          if ( v106 == 33 || (v29 = v28, (unsigned int)sub_B52F50(v106) == v10) )
            v16 = v29;
          else
            v16 = v91;
        }
      }
    }
  }
LABEL_12:
  v108 = 42;
  v109 = 0;
  if ( v10 == 32 )
  {
    v93 = v16;
    v37 = sub_1178DE0(v2);
    v16 = v93;
    if ( v37 )
    {
      v114 = v4;
      v113 = &v108;
      v116 = &v104;
      v118 = &v105;
      v115 = v5;
      LOBYTE(v117) = 0;
      v119 = 0;
      if ( *v100 == 86 )
      {
        v38 = (_BYTE **)sub_986520((__int64)v100);
        v39 = sub_1181140((__int64 *)&v113, *v38);
        v16 = v93;
        if ( v39 )
        {
          v40 = sub_986520((__int64)v100);
          v41 = sub_991580((__int64)&v116, *(_QWORD *)(v40 + 32));
          v16 = v93;
          if ( v41 )
          {
            v42 = sub_986520((__int64)v100);
            v43 = sub_991580((__int64)&v118, *(_QWORD *)(v42 + 64));
            v16 = v93;
            if ( v43 )
            {
              v44 = v108;
              v45 = v108 & 0xFFFFFFFB;
              if ( (v108 & 0xFFFFFFFB) != 0x22 )
              {
                v64 = sub_B52F50(v108);
                v109 = 0;
                v65 = v64;
                v108 = v64;
                v44 = v64;
                v66 = v4;
                v16 = v93;
                v4 = v5;
                v45 = v65 & 0xFFFFFFFB;
                v5 = v66;
              }
              v46 = v104;
              if ( *(_DWORD *)(v104 + 8) <= 0x40u )
              {
                v48 = *(_QWORD *)v104 == 1;
              }
              else
              {
                v84 = *(_DWORD *)(v104 + 8);
                v101 = v16;
                v47 = sub_C444A0(v104);
                v16 = v101;
                v48 = v84 - 1 == v47;
              }
              if ( !v48 )
              {
                v49 = v105;
                v105 = v46;
                v104 = v49;
                v50 = v4;
                v4 = v5;
                v5 = v50;
              }
              if ( v45 == 34 )
              {
                v51 = *(_DWORD *)(v104 + 8);
                if ( v51 <= 0x40 )
                {
                  v53 = *(_QWORD *)v104 == 1;
                }
                else
                {
                  v102 = v16;
                  v52 = sub_C444A0(v104);
                  v16 = v102;
                  v53 = v51 - 1 == v52;
                }
                if ( v53 )
                {
                  v103 = v16;
                  v54 = sub_986760(v105);
                  v16 = v103;
                  if ( v54 )
                  {
                    v17 = !sub_B532B0(v44) ? 362 : 313;
                    goto LABEL_14;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  v17 = !v75 ? 362 : 313;
  if ( v16 )
  {
LABEL_14:
    v18 = *(_QWORD *)(a2 + 8);
    v111 = 0;
    v19 = *(_QWORD *)(a1 + 32);
    v112[0] = v4;
    v112[1] = v5;
    v117 = 257;
    v20 = sub_B35180(v19, v18, v17, (__int64)v112, 2u, v110, (__int64)&v113);
    return sub_F162A0(a1, a2, v20);
  }
  return 0;
}
