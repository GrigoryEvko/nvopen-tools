// Function: sub_1018220
// Address: 0x1018220
//
unsigned __int8 *__fastcall sub_1018220(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __m128i *a5)
{
  unsigned int v5; // r15d
  __int64 v6; // r13
  __int64 v7; // rax
  unsigned int v8; // r14d
  _QWORD *v9; // r12
  bool v10; // bl
  __int64 v11; // r12
  _QWORD *v12; // rsi
  __int64 *v13; // r11
  _BYTE *v17; // rax
  unsigned __int8 *v18; // r15
  _BYTE *v19; // rdx
  __int64 v20; // r13
  __int64 v21; // r14
  __int64 v22; // rax
  unsigned int v23; // r13d
  char v24; // al
  __int64 v25; // rdi
  unsigned int v26; // r15d
  char v27; // bl
  char v28; // al
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r12
  unsigned __int8 *v33; // r13
  unsigned __int8 *v34; // r14
  char v35; // al
  __int64 *v36; // r11
  __int64 v37; // r9
  char v38; // al
  char v39; // r9
  char v40; // r8
  __int64 v41; // rcx
  char v42; // r13
  char v43; // r12
  char v44; // al
  char v45; // r13
  char v46; // r12
  char v47; // al
  char v48; // r13
  char v49; // r12
  char v50; // al
  char v51; // r13
  char v52; // al
  unsigned __int8 *v53; // rbx
  unsigned __int8 *v54; // rdx
  int v55; // ecx
  char v56; // r13
  char v57; // r12
  char v58; // al
  char v59; // r13
  char v60; // r12
  char v61; // al
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rax
  unsigned __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  unsigned int v69; // eax
  unsigned __int8 *v70; // r13
  _BYTE *v71; // r14
  __int64 v72; // r15
  __int64 v73; // rbx
  unsigned int v74; // eax
  __int64 v75; // r15
  __int64 v76; // r14
  unsigned __int64 v77; // rax
  __int64 v78; // rdi
  __int64 v79; // rsi
  char v80; // al
  __int64 v81; // r9
  __int64 *v82; // r11
  unsigned int v83; // ebx
  __int64 v84; // r9
  __int64 *v85; // r11
  int v86; // eax
  bool v87; // bl
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rax
  char v92; // al
  __int64 v93; // r9
  char v94; // al
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rax
  __int64 *v98; // [rsp+8h] [rbp-98h]
  __int64 *v99; // [rsp+8h] [rbp-98h]
  __int64 *v100; // [rsp+10h] [rbp-90h]
  __int64 v101; // [rsp+10h] [rbp-90h]
  __int64 v102; // [rsp+10h] [rbp-90h]
  __int64 v103; // [rsp+10h] [rbp-90h]
  __int64 v104; // [rsp+10h] [rbp-90h]
  __int64 v106; // [rsp+18h] [rbp-88h]
  _QWORD *v113; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v114; // [rsp+28h] [rbp-78h]
  unsigned __int64 v115; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v116; // [rsp+38h] [rbp-68h]
  unsigned __int64 v117; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v118; // [rsp+48h] [rbp-58h]
  _QWORD *v119; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v120; // [rsp+58h] [rbp-48h]
  unsigned __int64 *v121; // [rsp+60h] [rbp-40h]

  v5 = *(_DWORD *)(a2 + 36);
  if ( !(_DWORD)a4 )
  {
    if ( v5 == 493 )
    {
      v6 = **(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL);
      v7 = sub_B43CB0(a1);
      sub_988CD0((__int64)&v117, v7, 0x40u);
      v114 = v118;
      if ( v118 > 0x40 )
        sub_C43780((__int64)&v113, (const void **)&v117);
      else
        v113 = (_QWORD *)v117;
      sub_C46A40((__int64)&v113, 1);
      v8 = v114;
      v9 = v113;
      v114 = 0;
      v116 = v8;
      v115 = (unsigned __int64)v113;
      if ( v120 <= 0x40 )
        v10 = v119 == v113;
      else
        v10 = sub_C43C50((__int64)&v119, (const void **)&v115);
      if ( v8 > 0x40 )
      {
        if ( v9 )
        {
          j_j___libc_free_0_0(v9);
          if ( v114 > 0x40 )
          {
            if ( v113 )
              j_j___libc_free_0_0(v113);
          }
        }
      }
      v11 = 0;
      if ( v10 )
      {
        v12 = (_QWORD *)v117;
        if ( v118 > 0x40 )
          v12 = *(_QWORD **)v117;
        v11 = sub_AD64C0(v6, (__int64)v12, 0);
      }
      if ( v120 > 0x40 && v119 )
        j_j___libc_free_0_0(v119);
LABEL_19:
      if ( v118 > 0x40 )
      {
        if ( v117 )
          j_j___libc_free_0_0(v117);
      }
      return (unsigned __int8 *)v11;
    }
    return 0;
  }
  v13 = a3;
  if ( (_DWORD)a4 == 1 )
  {
    v11 = *a3;
    LOBYTE(v23) = 1;
    if ( v5 != 172 )
    {
      if ( v5 <= 0xAC )
      {
        if ( v5 > 0x15 )
        {
          if ( v5 != 170 )
          {
LABEL_87:
            if ( v5 <= 0xAA )
            {
              if ( v5 == 88 )
              {
                if ( !sub_B451B0(a1) )
                  return 0;
                if ( *(_BYTE *)v11 != 85 )
                  return 0;
                v90 = *(_QWORD *)(v11 - 32);
                if ( !v90
                  || *(_BYTE *)v90
                  || *(_QWORD *)(v90 + 24) != *(_QWORD *)(v11 + 80)
                  || *(_DWORD *)(v90 + 36) != 218 )
                {
                  return 0;
                }
                goto LABEL_98;
              }
              if ( v5 > 0x58 )
              {
                if ( v5 == 89 )
                {
                  if ( !sub_B451B0(a1) )
                    return 0;
                  if ( *(_BYTE *)v11 != 85 )
                    return 0;
                  v89 = *(_QWORD *)(v11 - 32);
                  if ( !v89
                    || *(_BYTE *)v89
                    || *(_QWORD *)(v89 + 24) != *(_QWORD *)(v11 + 80)
                    || *(_DWORD *)(v89 + 36) != 219 )
                  {
                    return 0;
                  }
                }
                else
                {
                  if ( v5 != 90 )
                    return 0;
                  if ( !sub_B451B0(a1) )
                    return 0;
                  if ( *(_BYTE *)v11 != 85 )
                    return 0;
                  v30 = *(_QWORD *)(v11 - 32);
                  if ( !v30
                    || *(_BYTE *)v30
                    || *(_QWORD *)(v30 + 24) != *(_QWORD *)(v11 + 80)
                    || *(_DWORD *)(v30 + 36) != 220 )
                  {
                    return 0;
                  }
                }
                goto LABEL_98;
              }
              goto LABEL_55;
            }
            if ( v5 == 219 )
            {
              if ( !sub_B451B0(a1) )
                return 0;
              if ( *(_BYTE *)v11 != 85
                || (v95 = *(_QWORD *)(v11 - 32)) == 0
                || *(_BYTE *)v95
                || *(_QWORD *)(v95 + 24) != *(_QWORD *)(v11 + 80)
                || *(_DWORD *)(v95 + 36) != 89
                || (v96 = *(_QWORD *)(v11 - 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF))) == 0 )
              {
                LODWORD(v117) = 284;
                v64 = 0x4024000000000000LL;
                v118 = 0;
LABEL_139:
                v119 = (_QWORD *)v64;
                v120 = 1;
                v121 = &v115;
                if ( *(_BYTE *)v11 != 85 )
                  return 0;
                v65 = *(_QWORD *)(v11 - 32);
                if ( !v65 )
                  return 0;
                if ( *(_BYTE *)v65 )
                  return 0;
                if ( *(_QWORD *)(v65 + 24) != *(_QWORD *)(v11 + 80) )
                  return 0;
                if ( *(_DWORD *)(v65 + 36) != 284 )
                  return 0;
                if ( !(unsigned __int8)sub_1009690(
                                         (double *)&v119,
                                         *(_QWORD *)(v11 - 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF))) )
                  return 0;
                if ( *(_BYTE *)v11 != 85 )
                  return 0;
                v66 = *(_QWORD *)(v11 + 32 * (v120 - (unsigned __int64)(*(_DWORD *)(v11 + 4) & 0x7FFFFFF)));
                if ( !v66 )
                  return 0;
                *v121 = v66;
                return (unsigned __int8 *)v115;
              }
              return (unsigned __int8 *)v96;
            }
            if ( v5 <= 0xDB )
            {
              if ( v5 == 179 )
              {
                if ( *(_BYTE *)v11 == 93 && *(_DWORD *)(v11 + 80) == 1 && !**(_DWORD **)(v11 + 72) )
                {
                  v11 = *(_QWORD *)(v11 - 32);
                  if ( v11 )
                  {
                    v115 = v11;
                    if ( *(_BYTE *)v11 == 85 )
                    {
                      v88 = *(_QWORD *)(v11 - 32);
                      if ( v88 )
                      {
                        if ( !*(_BYTE *)v88
                          && *(_QWORD *)(v88 + 24) == *(_QWORD *)(v11 + 80)
                          && *(_DWORD *)(v88 + 36) == 179 )
                        {
                          return (unsigned __int8 *)v11;
                        }
                      }
                    }
                  }
                }
                return 0;
              }
              if ( v5 != 218 )
                return 0;
              if ( !sub_B451B0(a1) )
                return 0;
              if ( *(_BYTE *)v11 != 85 )
                return 0;
              v67 = *(_QWORD *)(v11 - 32);
              if ( !v67
                || *(_BYTE *)v67
                || *(_QWORD *)(v67 + 24) != *(_QWORD *)(v11 + 80)
                || *(_DWORD *)(v67 + 36) != 88 )
              {
                return 0;
              }
              goto LABEL_98;
            }
            if ( v5 == 220 )
            {
              if ( !sub_B451B0(a1) )
                return 0;
              if ( *(_BYTE *)v11 != 85
                || (v97 = *(_QWORD *)(v11 - 32)) == 0
                || *(_BYTE *)v97
                || *(_QWORD *)(v97 + 24) != *(_QWORD *)(v11 + 80)
                || *(_DWORD *)(v97 + 36) != 90
                || (v96 = *(_QWORD *)(v11 - 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF))) == 0 )
              {
                LODWORD(v117) = 284;
                v64 = 0x4000000000000000LL;
                v118 = 0;
                goto LABEL_139;
              }
              return (unsigned __int8 *)v96;
            }
LABEL_72:
            if ( v5 == 402 )
            {
              if ( *(_BYTE *)v11 == 85 )
              {
                v62 = *(_QWORD *)(v11 - 32);
                if ( v62 )
                {
                  if ( !*(_BYTE *)v62 && *(_QWORD *)(v62 + 24) == *(_QWORD *)(v11 + 80) && *(_DWORD *)(v62 + 36) == 402 )
                  {
                    v63 = *(_QWORD *)(v11 - 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF));
                    if ( v63 )
                      return (unsigned __int8 *)v63;
                  }
                }
              }
              if ( sub_9B7DA0((char *)v11, 0xFFFFFFFF, 0) )
                return (unsigned __int8 *)v11;
            }
            return 0;
          }
          if ( *(_BYTE *)v11 != 85 )
            goto LABEL_83;
          goto LABEL_159;
        }
        if ( v5 <= 0x13 )
        {
          if ( v5 != 8 )
          {
LABEL_55:
            switch ( v5 )
            {
              case 0xFu:
                if ( *(_BYTE *)v11 != 85 )
                  return 0;
                v91 = *(_QWORD *)(v11 - 32);
                if ( !v91
                  || *(_BYTE *)v91
                  || *(_QWORD *)(v91 + 24) != *(_QWORD *)(v11 + 80)
                  || *(_DWORD *)(v91 + 36) != 15 )
                {
                  return 0;
                }
                break;
              case 0x42u:
                v24 = sub_9B64A0(
                        v11,
                        a5->m128i_i64[0],
                        0,
                        0,
                        a5[2].m128i_i64[0],
                        a5[2].m128i_i64[1],
                        a5[1].m128i_i64[1],
                        1);
                v25 = *(_QWORD *)(v11 + 8);
                if ( v24 )
                  return (unsigned __int8 *)sub_AD64C0(v25, 1, 0);
                v118 = sub_BCB060(v25);
                v26 = v118;
                if ( v118 > 0x40 )
                {
                  sub_C43690((__int64)&v117, 0, 0);
                  v23 = v118 - v26 + 1;
                  if ( v23 == v118 )
                    goto LABEL_61;
                  if ( v23 > 0x3F || v118 > 0x40 )
                  {
                    sub_C43C90(&v117, v23, v118);
                    goto LABEL_61;
                  }
                }
                else
                {
                  v117 = 0;
                  if ( v118 == 1 )
                  {
LABEL_61:
                    v27 = sub_9AC230(v11, (__int64)&v117, a5, 0);
                    if ( v118 > 0x40 && v117 )
                      j_j___libc_free_0_0(v117);
                    if ( v27 )
                      return (unsigned __int8 *)v11;
                    return 0;
                  }
                }
                v117 |= 0xFFFFFFFFFFFFFFFFLL >> (65 - (unsigned __int8)v26) << v23;
                goto LABEL_61;
              case 0xEu:
                if ( *(_BYTE *)v11 != 85 )
                  return 0;
                v31 = *(_QWORD *)(v11 - 32);
                if ( !v31
                  || *(_BYTE *)v31
                  || *(_QWORD *)(v31 + 24) != *(_QWORD *)(v11 + 80)
                  || *(_DWORD *)(v31 + 36) != 14 )
                {
                  return 0;
                }
                break;
              default:
                return 0;
            }
LABEL_98:
            v11 = *(_QWORD *)(v11 - 32LL * (*(_DWORD *)(v11 + 4) & 0x7FFFFFF));
            if ( v11 )
              return (unsigned __int8 *)v11;
            return 0;
          }
          if ( *(_BYTE *)v11 != 85 )
            return 0;
          goto LABEL_159;
        }
        if ( *(_BYTE *)v11 != 85 )
        {
          if ( v5 != 21 )
            goto LABEL_82;
          goto LABEL_79;
        }
LABEL_159:
        v68 = *(_QWORD *)(v11 - 32);
        if ( v68
          && !*(_BYTE *)v68
          && *(_QWORD *)(v68 + 24) == *(_QWORD *)(v11 + 80)
          && (*(_BYTE *)(v68 + 33) & 0x20) != 0
          && v5 == *(_DWORD *)(v68 + 36) )
        {
          return (unsigned __int8 *)v11;
        }
        if ( v5 <= 0xF9 && v5 != 21 && v5 != 172 )
          goto LABEL_82;
        if ( !v68 )
          return 0;
        goto LABEL_166;
      }
      if ( v5 > 0x136 )
      {
        if ( v5 != 355 )
          goto LABEL_72;
      }
      else if ( v5 <= 0x133 && v5 != 250 )
      {
        goto LABEL_69;
      }
    }
    if ( *(_BYTE *)v11 == 85 )
      goto LABEL_159;
    if ( v5 == 250 )
    {
LABEL_79:
      v28 = *(_BYTE *)v11;
      if ( *(_BYTE *)v11 <= 0x1Cu )
        return 0;
      if ( v28 != 85 )
      {
        if ( (unsigned __int8)(v28 - 72) <= 1u )
          return (unsigned __int8 *)v11;
LABEL_82:
        if ( v5 == 170 )
        {
LABEL_83:
          v29 = sub_9B4030((__int64 *)v11, 1023, 0, a5);
          if ( BYTE5(v29) && !BYTE4(v29) )
            return (unsigned __int8 *)v11;
          return 0;
        }
        goto LABEL_87;
      }
      v68 = *(_QWORD *)(v11 - 32);
      if ( !v68 )
        goto LABEL_82;
LABEL_166:
      if ( !*(_BYTE *)v68 && *(_QWORD *)(v68 + 24) == *(_QWORD *)(v11 + 80) && (*(_BYTE *)(v68 + 33) & 0x20) != 0 )
      {
        v69 = *(_DWORD *)(v68 + 36);
        if ( v69 == 250 )
          return (unsigned __int8 *)v11;
        if ( v69 > 0xFA )
        {
          if ( v69 > 0x136 )
          {
            if ( v69 == 355 )
              return (unsigned __int8 *)v11;
          }
          else if ( v69 > 0x133 )
          {
            return (unsigned __int8 *)v11;
          }
        }
        else if ( v69 == 21 || v69 == 172 )
        {
          return (unsigned __int8 *)v11;
        }
      }
      goto LABEL_82;
    }
LABEL_69:
    if ( v5 <= 0xFA )
    {
      if ( v5 != 172 )
        goto LABEL_82;
    }
    else if ( v5 < 0x134 )
    {
      return 0;
    }
    goto LABEL_79;
  }
  if ( (_DWORD)a4 == 2 )
    return sub_1016D00(v5, **(__int64 ****)(*(_QWORD *)(a2 + 24) + 16LL), *a3, a3[1], a5, a1);
  if ( v5 <= 0xB5 )
  {
    if ( v5 > 0x65 )
    {
      switch ( v5 )
      {
        case 0x66u:
          v59 = sub_B59DB0((unsigned __int8 *)a1, a2);
          v60 = sub_B59EF0((unsigned __int8 *)a1, a2);
          v61 = sub_B45210(a1);
          return sub_100EA30((__int64 *)*a3, (_BYTE *)a3[1], v61, a5, v60, v59);
        case 0x69u:
          v56 = sub_B59DB0((unsigned __int8 *)a1, a2);
          v57 = sub_B59EF0((unsigned __int8 *)a1, a2);
          v58 = sub_B45210(a1);
          return sub_100A740((_BYTE *)*a3, (_BYTE *)a3[1], v58, a5->m128i_i64, v57, v56);
        case 0x6Bu:
          v51 = sub_B59DB0((unsigned __int8 *)a1, a2);
          v52 = sub_B59EF0((unsigned __int8 *)a1, a2);
          v39 = v51;
          v41 = (__int64)a5;
          v13 = a3;
          v40 = v52;
          return sub_1003820(v13, a4, 0, v41, v40, v39);
        case 0x6Cu:
          v48 = sub_B59DB0((unsigned __int8 *)a1, a2);
          v49 = sub_B59EF0((unsigned __int8 *)a1, a2);
          v50 = sub_B45210(a1);
          return sub_1009EB0((_BYTE *)*a3, (_BYTE *)a3[1], v50, a5, v49, v48);
        case 0x72u:
          v45 = sub_B59DB0((unsigned __int8 *)a1, a2);
          v46 = sub_B59EF0((unsigned __int8 *)a1, a2);
          v47 = sub_B45210(a1);
          return sub_1009300((__int64 *)*a3, (_BYTE *)a3[1], v47, a5->m128i_i64, v46, v45);
        case 0x73u:
          v42 = sub_B59DB0((unsigned __int8 *)a1, a2);
          v43 = sub_B59EF0((unsigned __int8 *)a1, a2);
          v44 = sub_B45210(a1);
          return (unsigned __int8 *)sub_10091C0((__int64 *)*a3, (__int64 *)a3[1], v44, a5, v43, v42);
        case 0x74u:
          return sub_1003530(*a3, a3[1], (__int64)a5, 1);
        case 0x95u:
          v53 = (unsigned __int8 *)sub_B5B890(a1);
          v54 = (unsigned __int8 *)sub_B5B740(a1);
          v55 = *v53;
          if ( (unsigned int)(v55 - 12) <= 1 || (unsigned int)*v54 - 12 <= 1 )
            return (unsigned __int8 *)sub_ACA8A0(*(__int64 ***)(a1 + 8));
          v78 = *(_QWORD *)(a1 + 8);
          if ( *(_BYTE *)(v78 + 8) != 14 || (_BYTE)v55 != 20 )
            return 0;
          return (unsigned __int8 *)sub_AC9EC0((__int64 **)v78);
        case 0xADu:
        case 0xAEu:
          v39 = 1;
          v40 = 0;
          v41 = (__int64)a5;
          return sub_1003820(v13, a4, 0, v41, v40, v39);
        case 0xB4u:
        case 0xB5u:
          v32 = *a3;
          v33 = (unsigned __int8 *)a3[1];
          v34 = (unsigned __int8 *)a3[2];
          v35 = sub_1003090((__int64)a5, (unsigned __int8 *)*a3);
          v36 = a3;
          v37 = a2;
          if ( v35 )
          {
            v38 = sub_1003090((__int64)a5, v33);
            v37 = a2;
            v36 = a3;
            if ( v38 )
              return (unsigned __int8 *)sub_ACA8A0(**(__int64 ****)(*(_QWORD *)(a2 + 24) + 16LL));
          }
          v100 = v36;
          v106 = v37;
          if ( (unsigned __int8)sub_1003090((__int64)a5, v34) )
            return (unsigned __int8 *)v100[v5 != 180];
          v79 = (__int64)v34;
          LOBYTE(v118) = 0;
          v117 = (unsigned __int64)&v113;
          v80 = sub_991580((__int64)&v117, (__int64)v34);
          v81 = v106;
          if ( v80 )
          {
            v79 = (__int64)v113;
            v82 = v100;
            v116 = *((_DWORD *)v113 + 2);
            if ( v116 > 0x40 )
            {
              sub_C43690((__int64)&v115, v116, 0);
              v79 = (__int64)v113;
              v82 = v100;
              v81 = v106;
            }
            else
            {
              v115 = v116;
            }
            v98 = v82;
            v101 = v81;
            sub_C4B490((__int64)&v117, v79, (__int64)&v115);
            v83 = v118;
            v84 = v101;
            v85 = v98;
            if ( v118 <= 0x40 )
            {
              v87 = v117 == 0;
            }
            else
            {
              v86 = sub_C444A0((__int64)&v117);
              v85 = v98;
              v84 = v101;
              v87 = v83 == v86;
            }
            v102 = v84;
            v99 = v85;
            sub_969240((__int64 *)&v117);
            if ( v87 )
            {
              v11 = v99[v5 != 180];
              sub_969240((__int64 *)&v115);
              return (unsigned __int8 *)v11;
            }
            sub_969240((__int64 *)&v115);
            v81 = v102;
          }
          v103 = v81;
          v92 = sub_FFFE90(v32);
          v93 = v103;
          if ( v92 )
          {
            v94 = sub_FFFE90((__int64)v33);
            v93 = v103;
            if ( v94 )
              return (unsigned __int8 *)sub_AD6530(**(_QWORD **)(*(_QWORD *)(v103 + 24) + 16LL), v79);
          }
          v104 = v93;
          v115 = 0;
          if ( (unsigned __int8)sub_995B10((_QWORD **)&v115, v32) )
          {
            v117 = 0;
            if ( (unsigned __int8)sub_995B10((_QWORD **)&v117, (__int64)v33) )
              return (unsigned __int8 *)sub_AD62B0(**(_QWORD **)(*(_QWORD *)(v104 + 24) + 16LL));
          }
          break;
        default:
          return 0;
      }
    }
    return 0;
  }
  if ( v5 > 0x14C )
  {
    if ( v5 == 382 )
    {
      v17 = (_BYTE *)a3[2];
      v18 = (unsigned __int8 *)*a3;
      v19 = (_BYTE *)a3[1];
      v20 = **(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL);
      v21 = *((_QWORD *)v17 + 3);
      if ( *((_DWORD *)v17 + 8) > 0x40u )
        v21 = **((_QWORD **)v17 + 3);
      if ( *v19 == 85 )
      {
        v22 = *((_QWORD *)v19 - 4);
        if ( v22 )
        {
          if ( !*(_BYTE *)v22 && *(_QWORD *)(v22 + 24) == *((_QWORD *)v19 + 10) && *(_DWORD *)(v22 + 36) == 381 )
          {
            v11 = *(_QWORD *)&v19[-32 * (*((_DWORD *)v19 + 1) & 0x7FFFFFF)];
            if ( v11 )
            {
              if ( (unsigned __int8)sub_FFFE90(*(_QWORD *)&v19[32 * (1LL - (*((_DWORD *)v19 + 1) & 0x7FFFFFF))])
                && ((unsigned __int8)sub_1003090((__int64)a5, v18) || v18 == (unsigned __int8 *)v11)
                && !(_DWORD)v21
                && v20 == *(_QWORD *)(v11 + 8) )
              {
                return (unsigned __int8 *)v11;
              }
            }
          }
        }
      }
    }
    return 0;
  }
  if ( v5 <= 0x14A )
  {
    if ( v5 - 227 <= 1 )
    {
      v11 = a3[3];
      if ( (unsigned __int8)sub_9B9FC0((unsigned __int8 *)a3[2]) )
        return (unsigned __int8 *)v11;
    }
    return 0;
  }
  v11 = *a3;
  v70 = (unsigned __int8 *)a3[1];
  v71 = (_BYTE *)a3[2];
  v72 = **(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL);
  if ( *(_BYTE *)*a3 <= 0x15u )
  {
    v11 = a3[1];
    v70 = (unsigned __int8 *)*a3;
  }
  if ( !(unsigned __int8)sub_FFFE90((__int64)v70) )
  {
    a2 = (__int64)v70;
    if ( !(unsigned __int8)sub_1003090((__int64)a5, v70) )
    {
      v73 = *((_QWORD *)v71 + 3);
      if ( *((_DWORD *)v71 + 8) > 0x40u )
        v73 = **((_QWORD **)v71 + 3);
      v74 = sub_BCB060(v72);
      v118 = v74;
      v75 = 1LL << v73;
      if ( v74 > 0x40 )
      {
        sub_C43690((__int64)&v117, 0, 0);
        LOBYTE(v74) = v118;
        if ( v118 > 0x40 )
        {
          *(_QWORD *)(v117 + 8LL * ((unsigned int)v73 >> 6)) |= v75;
          v76 = 1LL << ((unsigned __int8)v118 - 1);
          if ( v118 > 0x40 )
          {
            v77 = *(_QWORD *)(v117 + 8LL * ((v118 - 1) >> 6));
            goto LABEL_189;
          }
LABEL_188:
          v77 = v117;
LABEL_189:
          if ( (v76 & v77) != 0 || (v115 = (unsigned __int64)&v117, !sub_10080A0((const void ***)&v115, (__int64)v70)) )
            v11 = 0;
          goto LABEL_19;
        }
      }
      else
      {
        v117 = 0;
      }
      v117 |= v75;
      v76 = 1LL << ((unsigned __int8)v74 - 1);
      goto LABEL_188;
    }
  }
  return (unsigned __int8 *)sub_AD6530(v72, a2);
}
