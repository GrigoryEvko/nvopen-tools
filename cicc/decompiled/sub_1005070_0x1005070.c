// Function: sub_1005070
// Address: 0x1005070
//
__int64 __fastcall sub_1005070(unsigned int a1, _BYTE *a2, __int64 a3, __m128i *a4)
{
  __int64 v8; // rdx
  _QWORD *v9; // rdi
  int v10; // ecx
  int v11; // eax
  __int64 *v12; // rax
  __int64 v13; // r14
  char v14; // al
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rdi
  _QWORD *v18; // rax
  __int64 result; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  unsigned __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rdi
  _BYTE *v27; // rsi
  __int64 v28; // rax
  unsigned int v29; // eax
  _QWORD *v30; // rdx
  _BYTE *v31; // rdx
  __int64 v32; // rax
  unsigned int v33; // edx
  _BYTE *v34; // rdx
  __int64 v35; // rax
  unsigned __int64 v36; // rcx
  unsigned int v37; // eax
  __int64 v38; // rdx
  unsigned int v39; // esi
  unsigned __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  _BYTE *v43; // rax
  _BYTE *v44; // rax
  unsigned __int64 v45; // rcx
  __int64 v46; // rdi
  __int64 v47; // rdi
  unsigned __int64 *v48; // r10
  unsigned int v49; // eax
  bool v50; // zf
  _QWORD *v51; // r12
  unsigned int v52; // ebx
  bool v53; // r15
  __int64 v54; // rdi
  __int64 v55; // rdi
  _QWORD *v56; // rax
  __int64 v57; // rdx
  _BYTE *v58; // rax
  __int64 v59; // rdx
  _BYTE *v60; // rax
  _BYTE *v61; // rax
  __int64 v62; // rdi
  __int64 v63; // r8
  unsigned int v64; // eax
  __int64 v65; // rdx
  _BYTE *v66; // rax
  unsigned __int64 *v67; // [rsp+10h] [rbp-140h]
  char v68; // [rsp+20h] [rbp-130h]
  __int64 v69; // [rsp+20h] [rbp-130h]
  int v70; // [rsp+28h] [rbp-128h]
  __int64 v71; // [rsp+28h] [rbp-128h]
  __int64 v72; // [rsp+28h] [rbp-128h]
  __int64 v73; // [rsp+28h] [rbp-128h]
  __int64 v74; // [rsp+28h] [rbp-128h]
  __int64 v75; // [rsp+28h] [rbp-128h]
  char v76; // [rsp+28h] [rbp-128h]
  int v77; // [rsp+28h] [rbp-128h]
  __int64 v78; // [rsp+28h] [rbp-128h]
  __int64 v79; // [rsp+38h] [rbp-118h] BYREF
  unsigned __int64 *v80; // [rsp+40h] [rbp-110h] BYREF
  __int64 v81; // [rsp+48h] [rbp-108h]
  unsigned __int64 v82; // [rsp+50h] [rbp-100h] BYREF
  unsigned int v83; // [rsp+58h] [rbp-F8h]
  unsigned __int64 v84; // [rsp+60h] [rbp-F0h] BYREF
  unsigned int v85; // [rsp+68h] [rbp-E8h]
  __int64 v86; // [rsp+70h] [rbp-E0h] BYREF
  unsigned int v87; // [rsp+78h] [rbp-D8h]
  __int64 v88; // [rsp+80h] [rbp-D0h] BYREF
  unsigned int v89; // [rsp+88h] [rbp-C8h]
  __int64 v90; // [rsp+90h] [rbp-C0h]
  __int64 *v91; // [rsp+98h] [rbp-B8h] BYREF
  char v92; // [rsp+A0h] [rbp-B0h]
  unsigned __int64 **v93; // [rsp+A8h] [rbp-A8h]
  unsigned __int8 v94; // [rsp+B0h] [rbp-A0h]
  _QWORD *v95; // [rsp+C0h] [rbp-90h] BYREF
  __int64 *v96; // [rsp+C8h] [rbp-88h] BYREF
  __int64 v97; // [rsp+D0h] [rbp-80h]
  unsigned __int64 **v98; // [rsp+D8h] [rbp-78h]
  unsigned __int8 v99; // [rsp+E0h] [rbp-70h]
  _QWORD *v100; // [rsp+F0h] [rbp-60h] BYREF
  __int64 *v101; // [rsp+F8h] [rbp-58h] BYREF
  __int64 v102; // [rsp+100h] [rbp-50h]
  unsigned __int64 **v103; // [rsp+108h] [rbp-48h]
  unsigned __int8 v104; // [rsp+110h] [rbp-40h]

  v8 = *(_QWORD *)(a3 + 8);
  v9 = *(_QWORD **)v8;
  v10 = *(unsigned __int8 *)(v8 + 8);
  if ( (unsigned int)(v10 - 17) > 1 )
  {
    v13 = sub_BCB2A0(v9);
    v14 = *a2;
    if ( *a2 != 58 )
      goto LABEL_3;
  }
  else
  {
    v11 = *(_DWORD *)(v8 + 32);
    BYTE4(v81) = (_BYTE)v10 == 18;
    LODWORD(v81) = v11;
    v12 = (__int64 *)sub_BCB2A0(v9);
    v13 = sub_BCE1B0(v12, v81);
    v14 = *a2;
    if ( *a2 != 58 )
      goto LABEL_3;
  }
  v20 = *((_QWORD *)a2 - 8);
  v21 = *((_QWORD *)a2 - 4);
  v22 = v20;
  if ( !v20 )
    goto LABEL_18;
  if ( a3 == v21 )
  {
    if ( !v21 )
      goto LABEL_18;
  }
  else
  {
    if ( !v21 || a3 != v20 )
      goto LABEL_18;
    v22 = *((_QWORD *)a2 - 4);
  }
  v71 = v22;
  if ( a1 - 39 > 1 )
    goto LABEL_18;
  sub_9AC330((__int64)&v95, a3, 0, a4);
  sub_9AC330((__int64)&v100, v71, 0, a4);
  if ( (unsigned int)v96 > 0x40 )
    v23 = v95[(unsigned int)((_DWORD)v96 - 1) >> 6];
  else
    v23 = (unsigned __int64)v95;
  if ( (v23 & (1LL << ((unsigned __int8)v96 - 1))) == 0 )
    goto LABEL_109;
  v24 = v102;
  if ( (unsigned int)v103 > 0x40 )
    v24 = *(_QWORD *)(v102 + 8LL * ((unsigned int)((_DWORD)v103 - 1) >> 6));
  if ( (v24 & (1LL << ((unsigned __int8)v103 - 1))) == 0 )
  {
LABEL_109:
    v38 = v97;
    if ( (unsigned int)v98 > 0x40 )
      v38 = *(_QWORD *)(v97 + 8LL * ((unsigned int)((_DWORD)v98 - 1) >> 6));
    if ( (v38 & (1LL << ((unsigned __int8)v98 - 1))) != 0 )
      goto LABEL_115;
    v39 = (unsigned int)v101;
    v40 = (unsigned __int64)v100;
    if ( (unsigned int)v101 > 0x40 )
      v40 = v100[(unsigned int)((_DWORD)v101 - 1) >> 6];
    if ( (v40 & (1LL << ((unsigned __int8)v101 - 1))) != 0 )
    {
LABEL_115:
      v25 = v13;
      if ( a1 == 40 )
        goto LABEL_36;
      goto LABEL_116;
    }
    if ( (unsigned int)v103 > 0x40 && v102 )
    {
      j_j___libc_free_0_0(v102);
      v39 = (unsigned int)v101;
    }
    if ( v39 > 0x40 && v100 )
      j_j___libc_free_0_0(v100);
    if ( (unsigned int)v98 > 0x40 && v97 )
      j_j___libc_free_0_0(v97);
    if ( (unsigned int)v96 > 0x40 && v95 )
      j_j___libc_free_0_0(v95);
    v14 = *a2;
LABEL_3:
    if ( v14 == 51 )
    {
      v15 = *((_QWORD *)a2 - 4);
      if ( a3 == v15 && v15 )
      {
        switch ( a1 )
        {
          case ' ':
          case '"':
          case '#':
            return sub_AD6450(v13);
          case '!':
          case '$':
          case '%':
            return sub_AD6400(v13);
          case '&':
          case '\'':
            sub_9AC330((__int64)&v100, a3, 0, a4);
            v33 = (unsigned int)v101;
            if ( (unsigned int)v101 > 0x40 )
              v36 = v100[(unsigned int)((_DWORD)v101 - 1) >> 6];
            else
              v36 = (unsigned __int64)v100;
            v37 = (unsigned int)v103;
            if ( (v36 & (1LL << ((unsigned __int8)v101 - 1))) == 0 )
              goto LABEL_81;
            if ( (unsigned int)v103 > 0x40 && v102 )
            {
              j_j___libc_free_0_0(v102);
              v33 = (unsigned int)v101;
            }
            if ( v33 > 0x40 )
            {
              if ( v100 )
                j_j___libc_free_0_0(v100);
            }
            return sub_AD6450(v13);
          case '(':
          case ')':
            sub_9AC330((__int64)&v100, a3, 0, a4);
            v33 = (unsigned int)v101;
            if ( (unsigned int)v101 > 0x40 )
              v45 = v100[(unsigned int)((_DWORD)v101 - 1) >> 6];
            else
              v45 = (unsigned __int64)v100;
            v37 = (unsigned int)v103;
            if ( (v45 & (1LL << ((unsigned __int8)v101 - 1))) == 0 )
            {
LABEL_81:
              if ( v37 > 0x40 && v102 )
              {
                j_j___libc_free_0_0(v102);
                v33 = (unsigned int)v101;
              }
              if ( v33 > 0x40 && v100 )
                j_j___libc_free_0_0(v100);
              v14 = *a2;
              goto LABEL_88;
            }
            if ( (unsigned int)v103 > 0x40 && v102 )
            {
              j_j___libc_free_0_0(v102);
              v33 = (unsigned int)v101;
            }
            if ( v33 > 0x40 && v100 )
              j_j___libc_free_0_0(v100);
            break;
          default:
            goto LABEL_18;
        }
        return sub_AD6400(v13);
      }
      goto LABEL_18;
    }
LABEL_88:
    if ( v14 != 55 )
      goto LABEL_119;
    v34 = (_BYTE *)*((_QWORD *)a2 - 8);
    if ( !v34 || (_BYTE *)a3 != v34 )
      goto LABEL_91;
    v16 = *((_QWORD *)a2 - 4);
    if ( *(_BYTE *)v16 == 17 )
    {
      v17 = v16 + 24;
    }
    else
    {
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v16 + 8) + 8LL) - 17 > 1 || *(_BYTE *)v16 > 0x15u )
      {
LABEL_91:
        v90 = a3;
        v91 = &v79;
        v92 = 0;
        v93 = &v80;
        v94 = 0;
        v95 = (_QWORD *)a3;
        v96 = &v79;
        LOBYTE(v97) = 0;
        v98 = &v80;
        v99 = 0;
        goto LABEL_92;
      }
      v44 = sub_AD7630(v16, 0, (__int64)v34);
      if ( !v44 || *v44 != 17 )
      {
LABEL_118:
        v14 = *a2;
LABEL_119:
        if ( v14 != 48 )
          goto LABEL_18;
        v31 = (_BYTE *)*((_QWORD *)a2 - 8);
        if ( (_BYTE *)a3 != v31 || !v31 )
        {
LABEL_122:
          v90 = a3;
          v91 = &v79;
          v92 = 0;
          v93 = &v80;
          v94 = 0;
          goto LABEL_64;
        }
        v54 = *((_QWORD *)a2 - 4);
        if ( *(_BYTE *)v54 == 17 )
        {
          v55 = v54 + 24;
        }
        else
        {
          if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v54 + 8) + 8LL) - 17 > 1 || *(_BYTE *)v54 > 0x15u )
            goto LABEL_122;
          v61 = sub_AD7630(v54, 0, (__int64)v31);
          if ( !v61 || *v61 != 17 )
            goto LABEL_17;
          v55 = (__int64)(v61 + 24);
        }
        if ( *(_DWORD *)(v55 + 8) <= 0x40u )
        {
          v56 = *(_QWORD **)v55;
          goto LABEL_179;
        }
        v77 = *(_DWORD *)(v55 + 8);
        if ( v77 - (unsigned int)sub_C444A0(v55) <= 0x40 )
        {
          v56 = **(_QWORD ***)v55;
LABEL_179:
          if ( v56 == (_QWORD *)1 )
            goto LABEL_17;
        }
LABEL_12:
        if ( (unsigned __int8)sub_9B6260(a3, a4, 0) )
        {
          if ( a1 <= 0x23 )
          {
            if ( a1 > 0x21 || a1 == 32 )
              return sub_AD6450(v13);
            if ( a1 != 33 )
              goto LABEL_17;
            return sub_AD6400(v13);
          }
          if ( a1 - 36 <= 1 )
            return sub_AD6400(v13);
        }
LABEL_17:
        v14 = *a2;
LABEL_18:
        v90 = a3;
        v91 = &v79;
        v92 = 0;
        v93 = &v80;
        v94 = 0;
        if ( v14 != 48 )
        {
LABEL_19:
          v95 = (_QWORD *)a3;
          v96 = &v79;
          LOBYTE(v97) = 0;
          v98 = &v80;
          v99 = 0;
          if ( v14 != 55 )
          {
LABEL_20:
            v100 = (_QWORD *)a3;
            v101 = &v79;
            LOBYTE(v102) = 0;
            v103 = &v80;
            v104 = 0;
            if ( v14 != 48 )
              goto LABEL_21;
            v31 = (_BYTE *)*((_QWORD *)a2 - 8);
            goto LABEL_66;
          }
          v34 = (_BYTE *)*((_QWORD *)a2 - 8);
LABEL_92:
          if ( *v34 != 46 )
            goto LABEL_21;
          v35 = *((_QWORD *)v34 - 8);
          if ( a3 != v35 || !v35 )
            goto LABEL_21;
          v76 = sub_991580((__int64)&v96, *((_QWORD *)v34 - 4));
          if ( !v76 )
            goto LABEL_96;
          v47 = *((_QWORD *)a2 - 4);
          if ( *(_BYTE *)v47 == 17 )
          {
            *v98 = (unsigned __int64 *)(v47 + 24);
          }
          else
          {
            v59 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v47 + 8) + 8LL) - 17;
            if ( (unsigned int)v59 > 1 || *(_BYTE *)v47 > 0x15u || (v60 = sub_AD7630(v47, v99, v59)) == 0 || *v60 != 17 )
            {
LABEL_96:
              v14 = *a2;
              goto LABEL_20;
            }
            *v98 = (unsigned __int64 *)(v60 + 24);
          }
          v48 = v80;
          v69 = v79;
          v49 = *((_DWORD *)v80 + 2);
          v85 = v49;
          if ( v49 > 0x40 )
          {
            sub_C43690((__int64)&v84, 1, 0);
            v48 = v80;
            v83 = v85;
            if ( v85 > 0x40 )
            {
              v67 = v80;
              sub_C43780((__int64)&v82, (const void **)&v84);
              v48 = v67;
              goto LABEL_159;
            }
          }
          else
          {
            v84 = 1;
            v83 = v49;
          }
          v82 = v84;
LABEL_159:
          sub_C47AC0((__int64)&v82, (__int64)v48);
          if ( (int)sub_C49970(v69, &v82) <= 0 )
            goto LABEL_71;
          v50 = *a2 == 48;
          v100 = (_QWORD *)a3;
          LOBYTE(v102) = 0;
          v101 = &v79;
          v103 = &v80;
          v104 = 0;
          if ( !v50 )
            goto LABEL_70;
          v31 = (_BYTE *)*((_QWORD *)a2 - 8);
          if ( *v31 != 54 )
            goto LABEL_70;
          v68 = v76;
LABEL_67:
          v32 = *((_QWORD *)v31 - 8);
          if ( a3 == v32 && v32 && (unsigned __int8)sub_991580((__int64)&v101, *((_QWORD *)v31 - 4)) )
          {
            v62 = *((_QWORD *)a2 - 4);
            if ( *(_BYTE *)v62 == 17 )
            {
              *v103 = (unsigned __int64 *)(v62 + 24);
              goto LABEL_217;
            }
            v65 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v62 + 8) + 8LL) - 17;
            if ( (unsigned int)v65 <= 1 && *(_BYTE *)v62 <= 0x15u )
            {
              v66 = sub_AD7630(v62, v104, v65);
              if ( v66 )
              {
                if ( *v66 == 17 )
                {
                  *v103 = (unsigned __int64 *)(v66 + 24);
LABEL_217:
                  v63 = v79;
                  v64 = *(_DWORD *)(v79 + 8);
                  v89 = v64;
                  if ( v64 > 0x40 )
                  {
                    sub_C43690((__int64)&v88, 1, 0);
                    v63 = v79;
                    v87 = v89;
                    if ( v89 > 0x40 )
                    {
                      v78 = v79;
                      sub_C43780((__int64)&v86, (const void **)&v88);
                      v63 = v78;
                      goto LABEL_220;
                    }
                  }
                  else
                  {
                    v88 = 1;
                    v87 = v64;
                  }
                  v86 = v88;
LABEL_220:
                  sub_C47AC0((__int64)&v86, v63);
                  v76 = (int)sub_C49970((__int64)&v86, v80) <= 0;
                  if ( v87 > 0x40 && v86 )
                    j_j___libc_free_0_0(v86);
                  if ( v89 > 0x40 && v88 )
                    j_j___libc_free_0_0(v88);
                  if ( !v68 )
                  {
LABEL_77:
                    if ( !v76 )
                      goto LABEL_21;
                    goto LABEL_78;
                  }
LABEL_71:
                  if ( v83 > 0x40 && v82 )
                    j_j___libc_free_0_0(v82);
                  if ( v85 > 0x40 && v84 )
                    j_j___libc_free_0_0(v84);
                  goto LABEL_77;
                }
              }
            }
          }
          if ( !v68 )
            goto LABEL_21;
LABEL_70:
          v76 = 0;
          goto LABEL_71;
        }
        v31 = (_BYTE *)*((_QWORD *)a2 - 8);
LABEL_64:
        if ( *v31 != 46 || (v41 = *((_QWORD *)v31 - 8)) == 0 || a3 != v41 )
        {
          v95 = (_QWORD *)a3;
          v96 = &v79;
          LOBYTE(v97) = 0;
          v98 = &v80;
          v99 = 0;
          v100 = (_QWORD *)a3;
          v101 = &v79;
          LOBYTE(v102) = 0;
          v103 = &v80;
          v104 = 0;
LABEL_66:
          v68 = 0;
          if ( *v31 != 54 )
            goto LABEL_21;
          goto LABEL_67;
        }
        if ( !(unsigned __int8)sub_991580((__int64)&v91, *((_QWORD *)v31 - 4)) )
          goto LABEL_126;
        v46 = *((_QWORD *)a2 - 4);
        if ( *(_BYTE *)v46 == 17 )
        {
          *v93 = (unsigned __int64 *)(v46 + 24);
        }
        else
        {
          v57 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v46 + 8) + 8LL) - 17;
          if ( (unsigned int)v57 > 1 || *(_BYTE *)v46 > 0x15u || (v58 = sub_AD7630(v46, v94, v57)) == 0 || *v58 != 17 )
          {
LABEL_126:
            v14 = *a2;
            goto LABEL_19;
          }
          *v93 = (unsigned __int64 *)(v58 + 24);
        }
        if ( (int)sub_C49970(v79, v80) <= 0 )
        {
LABEL_78:
          if ( a1 == 34 )
            return sub_AD6450(v13);
          if ( a1 == 37 )
            return sub_AD6400(v13);
LABEL_21:
          if ( *a2 != 44 )
            return 0;
          v26 = *((_QWORD *)a2 - 8);
          v27 = (_BYTE *)(v26 + 24);
          if ( *(_BYTE *)v26 != 17 )
          {
            v42 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v26 + 8) + 8LL) - 17;
            if ( (unsigned int)v42 > 1 )
              return 0;
            if ( *(_BYTE *)v26 > 0x15u )
              return 0;
            v43 = sub_AD7630(v26, 1, v42);
            if ( !v43 || *v43 != 17 )
              return 0;
            v27 = v43 + 24;
          }
          v28 = *((_QWORD *)a2 - 4);
          if ( a3 != v28 || !v28 )
            return 0;
          v29 = *((_DWORD *)v27 + 2);
          LODWORD(v101) = v29;
          if ( v29 > 0x40 )
          {
            sub_C43780((__int64)&v100, (const void **)v27);
            v29 = (unsigned int)v101;
            if ( (unsigned int)v101 > 0x40 )
            {
              *v100 &= 1uLL;
              v51 = v100;
              memset(v100 + 1, 0, 8 * (unsigned int)(((unsigned __int64)(unsigned int)v101 + 63) >> 6) - 8);
              v52 = (unsigned int)v101;
              v95 = v51;
              v30 = v51;
              LODWORD(v101) = 0;
              LODWORD(v96) = v52;
              if ( v52 > 0x40 )
              {
                v53 = 0;
                if ( v52 - (unsigned int)sub_C444A0((__int64)&v95) <= 0x40 && *v51 == 1 )
                  v53 = a1 - 32 <= 1;
                if ( v95 )
                  j_j___libc_free_0_0(v95);
                goto LABEL_170;
              }
LABEL_55:
              if ( v30 != (_QWORD *)1 )
                return 0;
              v53 = a1 - 32 <= 1;
LABEL_170:
              if ( (unsigned int)v101 > 0x40 && v100 )
                j_j___libc_free_0_0(v100);
              if ( !v53 )
                return 0;
              if ( a1 != 32 )
                return sub_AD6400(v13);
              return sub_AD6450(v13);
            }
          }
          else
          {
            v100 = *(_QWORD **)v27;
          }
          LODWORD(v96) = v29;
          LODWORD(v101) = 0;
          v30 = (_QWORD *)((unsigned __int8)v100 & 1);
          v100 = v30;
          v95 = v30;
          goto LABEL_55;
        }
        v14 = *a2;
        goto LABEL_19;
      }
      v17 = (__int64)(v44 + 24);
    }
    if ( *(_DWORD *)(v17 + 8) > 0x40u )
    {
      v70 = *(_DWORD *)(v17 + 8);
      if ( v70 - (unsigned int)sub_C444A0(v17) > 0x40 )
        goto LABEL_12;
      v18 = **(_QWORD ***)v17;
    }
    else
    {
      v18 = *(_QWORD **)v17;
    }
    if ( v18 )
      goto LABEL_12;
    goto LABEL_118;
  }
  v25 = v13;
  if ( a1 != 40 )
  {
LABEL_36:
    result = sub_AD6450(v25);
    goto LABEL_37;
  }
LABEL_116:
  result = sub_AD6400(v25);
LABEL_37:
  if ( (unsigned int)v103 > 0x40 && v102 )
  {
    v72 = result;
    j_j___libc_free_0_0(v102);
    result = v72;
  }
  if ( (unsigned int)v101 > 0x40 && v100 )
  {
    v73 = result;
    j_j___libc_free_0_0(v100);
    result = v73;
  }
  if ( (unsigned int)v98 > 0x40 && v97 )
  {
    v74 = result;
    j_j___libc_free_0_0(v97);
    result = v74;
  }
  if ( (unsigned int)v96 > 0x40 && v95 )
  {
    v75 = result;
    j_j___libc_free_0_0(v95);
    return v75;
  }
  return result;
}
