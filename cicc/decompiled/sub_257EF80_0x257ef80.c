// Function: sub_257EF80
// Address: 0x257ef80
//
__int64 __fastcall sub_257EF80(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 i)
{
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // r12
  _QWORD *v11; // rax
  _QWORD *v12; // rcx
  int v13; // r8d
  unsigned int v14; // r12d
  __int64 v16; // r12
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r13
  __int64 v20; // r14
  _QWORD *v21; // rax
  unsigned __int64 v22; // rax
  __int64 v23; // r14
  __int64 v24; // r12
  unsigned __int64 v25; // r13
  _QWORD *v26; // rax
  _QWORD *v27; // rcx
  unsigned __int8 v28; // r9
  int v29; // r8d
  __int64 v30; // rax
  __int64 v31; // r15
  unsigned int v32; // eax
  __int64 *v33; // r12
  unsigned __int8 v34; // dl
  unsigned __int64 v35; // rax
  __int64 v36; // r13
  unsigned int j; // ebx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // r14
  __int64 v41; // rdi
  char *v42; // rax
  char *v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // rax
  unsigned __int64 v46; // rdx
  int v47; // esi
  __int64 *v48; // rdx
  int v49; // eax
  _QWORD *v50; // rax
  unsigned __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // r12
  __int64 v54; // r13
  _QWORD *v55; // rax
  _QWORD *v56; // rsi
  __int64 *v57; // rax
  __int64 v58; // r12
  __int64 *v59; // r15
  __int64 *v60; // rbx
  _QWORD *v61; // rdx
  int v62; // eax
  unsigned int v63; // esi
  int v64; // eax
  __int64 v65; // r12
  __int64 *v66; // rax
  unsigned int v67; // esi
  int v68; // eax
  __int64 *v69; // rdx
  int v70; // eax
  __int64 v71; // [rsp-10h] [rbp-2B0h]
  __int64 *v73; // [rsp+30h] [rbp-270h]
  unsigned __int8 v74; // [rsp+38h] [rbp-268h]
  unsigned __int8 v75; // [rsp+3Eh] [rbp-262h]
  unsigned __int8 v76; // [rsp+3Fh] [rbp-261h]
  bool v77; // [rsp+3Fh] [rbp-261h]
  int v78; // [rsp+40h] [rbp-260h]
  unsigned __int64 v79; // [rsp+40h] [rbp-260h]
  __int64 v81; // [rsp+58h] [rbp-248h] BYREF
  __int64 *v82; // [rsp+60h] [rbp-240h] BYREF
  __int64 *v83; // [rsp+68h] [rbp-238h] BYREF
  __int64 *v84; // [rsp+70h] [rbp-230h] BYREF
  __int64 v85; // [rsp+78h] [rbp-228h]
  __int64 v86; // [rsp+80h] [rbp-220h] BYREF
  __int64 *v87; // [rsp+88h] [rbp-218h]
  __int64 v88; // [rsp+90h] [rbp-210h]
  __int64 v89; // [rsp+98h] [rbp-208h]
  _QWORD *v90; // [rsp+A0h] [rbp-200h] BYREF
  __int64 v91; // [rsp+A8h] [rbp-1F8h]
  _QWORD v92[16]; // [rsp+B0h] [rbp-1F0h] BYREF
  __int64 v93; // [rsp+130h] [rbp-170h] BYREF
  char *v94; // [rsp+138h] [rbp-168h]
  __int64 v95; // [rsp+140h] [rbp-160h]
  int v96; // [rsp+148h] [rbp-158h]
  char v97; // [rsp+14Ch] [rbp-154h]
  char v98; // [rsp+150h] [rbp-150h] BYREF
  unsigned __int64 v99; // [rsp+1D0h] [rbp-D0h] BYREF
  char *v100; // [rsp+1D8h] [rbp-C8h]
  __int64 v101; // [rsp+1E0h] [rbp-C0h]
  int v102; // [rsp+1E8h] [rbp-B8h]
  char v103; // [rsp+1ECh] [rbp-B4h]
  char v104; // [rsp+1F0h] [rbp-B0h] BYREF

  v4 = *(_QWORD *)a3;
  v5 = *(_QWORD *)(a3 + 8);
  v6 = *(_QWORD *)(v5 + 40);
  v7 = *(_QWORD *)(*(_QWORD *)a3 + 40LL);
  v73 = (__int64 *)a3;
  v74 = i;
  v81 = v6;
  if ( v7 == v6 )
  {
    if ( v4 == v5 )
      return (unsigned int)sub_25736B0(a1, a2, 1, (__int64)v73, 0, v74);
    a3 = *(_QWORD *)(a3 + 16);
    v52 = v7;
    v53 = v4;
    while ( 1 )
    {
      v54 = *(_QWORD *)(v53 + 32);
      LOBYTE(i) = v54 != v52 + 48 && v54 != 0;
      if ( !(_BYTE)i )
        break;
      v53 = v54 - 24;
      if ( v5 == v54 - 24 )
        return (unsigned int)sub_25736B0(a1, a2, 1, (__int64)v73, 0, v74);
      LOBYTE(i) = a3 != 0 && v4 != v53;
      if ( (_BYTE)i )
      {
        if ( *(_BYTE *)(a3 + 28) )
        {
          v55 = *(_QWORD **)(a3 + 8);
          v56 = &v55[*(unsigned int *)(a3 + 20)];
          if ( v55 != v56 )
          {
            while ( *v55 != v53 )
            {
              if ( v56 == ++v55 )
                goto LABEL_113;
            }
            break;
          }
        }
        else
        {
          v77 = a3 != 0 && v4 != v53;
          v79 = a3;
          v57 = sub_C8CA60(a3, v54 - 24);
          a3 = v79;
          i = v77;
          if ( v57 )
            break;
        }
      }
LABEL_113:
      v52 = *(_QWORD *)(v54 + 16);
    }
    v75 = i;
    v5 = v73[1];
    v9 = v73[2];
    v8 = *(_QWORD *)(v81 + 56);
    if ( v8 )
    {
LABEL_3:
      v10 = v8 - 24;
      if ( v10 != v5 )
      {
        LOBYTE(a3) = v9 != 0;
        while ( 1 )
        {
          if ( v4 != v10 && v9 )
          {
            if ( *(_BYTE *)(v9 + 28) )
            {
              v11 = *(_QWORD **)(v9 + 8);
              v12 = &v11[*(unsigned int *)(v9 + 20)];
              if ( v11 != v12 )
              {
                while ( *v11 != v10 )
                {
                  if ( v12 == ++v11 )
                    goto LABEL_16;
                }
LABEL_12:
                v13 = 1;
                return (unsigned int)sub_25736B0(a1, a2, 0, (__int64)v73, v13, v74);
              }
            }
            else if ( sub_C8CA60(v9, v10) )
            {
              goto LABEL_12;
            }
          }
LABEL_16:
          i = *(_QWORD *)(v10 + 32);
          if ( i == *(_QWORD *)(v10 + 40) + 48LL || !i )
            break;
          v10 = i - 24;
          if ( i - 24 == v5 )
            goto LABEL_21;
        }
        if ( !v5 )
        {
LABEL_21:
          v9 = v73[2];
          goto LABEL_22;
        }
        goto LABEL_111;
      }
LABEL_22:
      v16 = *(_QWORD *)(v7 + 72);
      v93 = 0;
      v94 = &v98;
      v95 = 16;
      v96 = 0;
      v97 = 1;
      if ( v9 )
      {
        v17 = *(_QWORD *)(v9 + 8);
        v18 = *(_BYTE *)(v9 + 28) ? v17 + 8LL * *(unsigned int *)(v9 + 20) : v17 + 8LL * *(unsigned int *)(v9 + 16);
        v90 = *(_QWORD **)(v9 + 8);
        v91 = v18;
        sub_254BBF0((__int64)&v90);
        v92[0] = v9;
        v92[1] = *(_QWORD *)v9;
        a3 = *(_BYTE *)(v9 + 28) ? *(unsigned int *)(v9 + 20) : *(unsigned int *)(v9 + 16);
        v19 = *(_QWORD *)(v9 + 8) + 8 * a3;
        for ( i = (__int64)v90; (_QWORD *)v19 != v90; i = (__int64)v90 )
        {
          while ( 1 )
          {
            v20 = *(_QWORD *)i;
            if ( v16 == sub_B43CB0(*(_QWORD *)i) )
            {
              v50 = sub_AE6EC0((__int64)&v93, *(_QWORD *)(v20 + 40));
              if ( v97 )
                v51 = (unsigned __int64)&v94[8 * HIDWORD(v95)];
              else
                v51 = (unsigned __int64)&v94[8 * (unsigned int)v95];
              v99 = (unsigned __int64)v50;
              v100 = (char *)v51;
              sub_254BBF0((__int64)&v99);
            }
            i = v91;
            v21 = v90 + 1;
            v90 = v21;
            if ( v21 != (_QWORD *)v91 )
              break;
LABEL_32:
            if ( v19 == v91 )
              goto LABEL_33;
          }
          while ( 1 )
          {
            a3 = *v21 + 2LL;
            if ( a3 > 1 )
              break;
            v90 = ++v21;
            if ( v21 == (_QWORD *)v91 )
              goto LABEL_32;
          }
        }
      }
LABEL_33:
      if ( (unsigned __int8)sub_B19060((__int64)&v93, v7, a3, i) )
      {
        v22 = sub_986580(v7);
        v23 = v73[2];
        v24 = *v73;
        v25 = v22;
        if ( !*v73 )
        {
          if ( !v22 )
            goto LABEL_46;
LABEL_43:
          v28 = v74;
          v29 = 1;
LABEL_44:
          v14 = sub_25736B0(a1, a2, 0, (__int64)v73, v29, v28);
LABEL_83:
          if ( !v97 )
            _libc_free((unsigned __int64)v94);
          return v14;
        }
        if ( v22 != v24 )
        {
          while ( 1 )
          {
            if ( v4 != v24 && v23 )
            {
              if ( *(_BYTE *)(v23 + 28) )
              {
                v26 = *(_QWORD **)(v23 + 8);
                v27 = &v26[*(unsigned int *)(v23 + 20)];
                if ( v26 != v27 )
                {
                  while ( *v26 != v24 )
                  {
                    if ( v27 == ++v26 )
                      goto LABEL_69;
                  }
                  goto LABEL_43;
                }
              }
              else if ( sub_C8CA60(v23, v24) )
              {
                goto LABEL_43;
              }
            }
LABEL_69:
            v44 = *(_QWORD *)(v24 + 32);
            if ( v44 == *(_QWORD *)(v24 + 40) + 48LL || !v44 )
              break;
            v24 = v44 - 24;
            if ( v25 == v44 - 24 )
              goto LABEL_46;
          }
          if ( !v25 )
            goto LABEL_46;
          goto LABEL_43;
        }
      }
LABEL_46:
      v30 = sub_251BBC0(a2, *(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80), a1, 1, 0, 1);
      v31 = v30;
      if ( !v30
        || !(*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v30 + 160LL))(v30, v81, v71) )
      {
        v99 = 0;
        v100 = &v104;
        v90 = v92;
        v91 = 0x1000000001LL;
        v32 = 1;
        v101 = 16;
        v102 = 0;
        v103 = 1;
        v92[0] = v7;
        v86 = 0;
        v87 = 0;
        v88 = 0;
        v89 = 0;
        while ( v32 )
        {
          v33 = (__int64 *)v90[v32 - 1];
          LODWORD(v91) = v32 - 1;
          sub_AE6EC0((__int64)&v99, (__int64)v33);
          v76 = v34;
          if ( v34 )
          {
            v35 = sub_986580((__int64)v33);
            v36 = v35;
            if ( v35 )
            {
              v78 = sub_B46E30(v35);
              if ( v78 )
              {
                for ( j = 0; j != v78; ++j )
                {
                  v40 = sub_B46EC0(v36, j);
                  if ( v31
                    && (*(unsigned __int8 (__fastcall **)(__int64, __int64 *, __int64))(*(_QWORD *)v31 + 168LL))(
                         v31,
                         v33,
                         v40) )
                  {
                    v85 = v40;
                    v84 = v33;
                    if ( !(unsigned __int8)sub_255FB50((__int64)&v86, (__int64 *)&v84, &v82) )
                    {
                      v47 = v89;
                      v48 = v82;
                      ++v86;
                      v49 = v88 + 1;
                      v83 = v82;
                      if ( 4 * ((int)v88 + 1) >= (unsigned int)(3 * v89) )
                      {
                        v47 = 2 * v89;
                      }
                      else if ( (int)v89 - HIDWORD(v88) - v49 > (unsigned int)v89 >> 3 )
                      {
                        goto LABEL_88;
                      }
                      sub_256DD30((__int64)&v86, v47);
                      sub_255FB50((__int64)&v86, (__int64 *)&v84, &v83);
                      v48 = v83;
                      v49 = v88 + 1;
LABEL_88:
                      LODWORD(v88) = v49;
                      if ( *v48 != -4096 || v48[1] != -4096 )
                        --HIDWORD(v88);
                      *v48 = (__int64)v84;
                      v48[1] = v85;
                    }
                  }
                  else
                  {
                    if ( v81 == v40
                      || (v41 = *(_QWORD *)(a1 + 264)) != 0
                      && HIDWORD(v95) == v96
                      && (unsigned __int8)sub_B19720(v41, (__int64)v33, v81) )
                    {
                      v14 = sub_25736B0(a1, a2, 1, (__int64)v73, v75, v74);
                      goto LABEL_79;
                    }
                    if ( v97 )
                    {
                      v42 = v94;
                      v43 = &v94[8 * HIDWORD(v95)];
                      if ( v94 == v43 )
                        goto LABEL_74;
                      while ( v40 != *(_QWORD *)v42 )
                      {
                        v42 += 8;
                        if ( v43 == v42 )
                          goto LABEL_74;
                      }
                    }
                    else if ( !sub_C8CA60((__int64)&v93, v40) )
                    {
LABEL_74:
                      v45 = (unsigned int)v91;
                      v46 = (unsigned int)v91 + 1LL;
                      if ( v46 > HIDWORD(v91) )
                      {
                        sub_C8D5F0((__int64)&v90, v92, v46, 8u, v38, v39);
                        v45 = (unsigned int)v91;
                      }
                      v90[v45] = v40;
                      LODWORD(v91) = v91 + 1;
                      continue;
                    }
                    v75 = v76;
                  }
                }
              }
            }
          }
          v32 = v91;
        }
        v59 = v87;
        v60 = &v87[2 * (unsigned int)v89];
        if ( !(_DWORD)v88 || v87 == v60 )
          goto LABEL_119;
        while ( *v59 == -4096 )
        {
          if ( v59[1] != -4096 )
            goto LABEL_131;
LABEL_156:
          v59 += 2;
          if ( v60 == v59 )
            goto LABEL_119;
        }
        if ( *v59 == -8192 && v59[1] == -8192 )
          goto LABEL_156;
LABEL_131:
        if ( v60 == v59 )
        {
LABEL_119:
          v14 = sub_25736B0(a1, a2, 0, (__int64)v73, v75, v74);
LABEL_79:
          sub_C7D6A0((__int64)v87, 16LL * (unsigned int)v89, 8);
          if ( v90 != v92 )
            _libc_free((unsigned __int64)v90);
          if ( !v103 )
            _libc_free((unsigned __int64)v100);
          goto LABEL_83;
        }
        v65 = a1 + 232;
LABEL_135:
        if ( (unsigned __int8)sub_255FB50(v65, v59, &v83) )
          goto LABEL_136;
        v67 = *(_DWORD *)(a1 + 256);
        v68 = *(_DWORD *)(a1 + 248);
        v69 = v83;
        ++*(_QWORD *)(a1 + 232);
        v70 = v68 + 1;
        v84 = v69;
        if ( 4 * v70 >= 3 * v67 )
        {
          v67 *= 2;
        }
        else if ( v67 - *(_DWORD *)(a1 + 252) - v70 > v67 >> 3 )
        {
          goto LABEL_143;
        }
        sub_256DD30(v65, v67);
        sub_255FB50(v65, v59, &v84);
        v69 = v84;
        v70 = *(_DWORD *)(a1 + 248) + 1;
LABEL_143:
        *(_DWORD *)(a1 + 248) = v70;
        if ( *v69 != -4096 || v69[1] != -4096 )
          --*(_DWORD *)(a1 + 252);
        *v69 = *v59;
        v69[1] = v59[1];
LABEL_136:
        v66 = v59 + 2;
        if ( v60 == v59 + 2 )
          goto LABEL_119;
        while ( 1 )
        {
          v59 = v66;
          if ( *v66 == -4096 )
          {
            if ( v66[1] != -4096 )
              goto LABEL_134;
          }
          else if ( *v66 != -8192 || v66[1] != -8192 )
          {
LABEL_134:
            if ( v60 == v66 )
              goto LABEL_119;
            goto LABEL_135;
          }
          v66 += 2;
          if ( v60 == v66 )
            goto LABEL_119;
        }
      }
      v58 = a1 + 200;
      if ( (unsigned __int8)sub_F9EAB0(a1 + 200, &v81, &v90) )
      {
LABEL_115:
        v28 = v74;
        v29 = v75;
        goto LABEL_44;
      }
      v61 = v90;
      v62 = *(_DWORD *)(a1 + 216);
      v63 = *(_DWORD *)(a1 + 224);
      v99 = (unsigned __int64)v90;
      ++*(_QWORD *)(a1 + 200);
      v64 = v62 + 1;
      if ( 4 * v64 >= 3 * v63 )
      {
        v63 *= 2;
      }
      else if ( v63 - *(_DWORD *)(a1 + 220) - v64 > v63 >> 3 )
      {
LABEL_123:
        *(_DWORD *)(a1 + 216) = v64;
        if ( *v61 != -4096 )
          --*(_DWORD *)(a1 + 220);
        *v61 = v81;
        goto LABEL_115;
      }
      sub_E3B4A0(v58, v63);
      sub_F9EAB0(v58, &v81, &v99);
      v61 = (_QWORD *)v99;
      v64 = *(_DWORD *)(a1 + 216) + 1;
      goto LABEL_123;
    }
    if ( !v5 )
      goto LABEL_22;
  }
  else
  {
    v8 = *(_QWORD *)(v6 + 56);
    v75 = 0;
    v9 = *(_QWORD *)(a3 + 16);
    if ( v8 )
      goto LABEL_3;
  }
LABEL_111:
  v13 = v75;
  return (unsigned int)sub_25736B0(a1, a2, 0, (__int64)v73, v13, v74);
}
