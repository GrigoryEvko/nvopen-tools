// Function: sub_1E03840
// Address: 0x1e03840
//
__int64 __fastcall sub_1E03840(__int64 a1, unsigned int a2, int a3, char a4)
{
  int v6; // ebx
  int *v7; // rax
  int v8; // edx
  int v9; // eax
  __int64 *v10; // r12
  __int64 result; // rax
  __int64 v12; // rbx
  __int64 *v13; // r12
  __int64 v14; // r14
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned int v17; // r8d
  __int64 *v18; // rdx
  __int64 v19; // rdi
  unsigned int v20; // r8d
  __int64 v21; // rdi
  unsigned int v22; // esi
  __int64 *v23; // rax
  __int64 v24; // rcx
  unsigned __int64 v25; // r13
  int v26; // edx
  int v27; // r10d
  __int64 *v28; // rbx
  __int64 *v29; // rbx
  __int64 v30; // r14
  __int64 v31; // r13
  __int64 v32; // r8
  int v33; // r9d
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rsi
  unsigned int v37; // ecx
  __int64 *v38; // rdx
  __int64 v39; // rax
  __int64 *v40; // rbx
  char v41; // r9
  __int64 *v42; // rax
  __int64 v43; // rdx
  int v44; // esi
  _QWORD *v45; // rax
  unsigned int v46; // ecx
  __int64 *v47; // rdx
  _QWORD *v48; // r8
  int v49; // r13d
  unsigned int v50; // r12d
  __int64 *v51; // rbx
  __int64 v52; // r14
  __int64 v53; // r13
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rdi
  unsigned int v57; // ecx
  __int64 *v58; // rdx
  __int64 v59; // r9
  int v60; // esi
  __int64 v61; // rcx
  __int64 v62; // rdx
  __int64 *v63; // rax
  _QWORD *v64; // r8
  unsigned int v65; // eax
  _DWORD *v66; // rax
  int v67; // edx
  _DWORD *v68; // rax
  int v69; // edx
  int v70; // edx
  __int64 v71; // rdi
  int v72; // r8d
  int v73; // r9d
  __int64 *v74; // r13
  __int64 *i; // rbx
  __int64 v76; // r12
  int v77; // r10d
  __int64 *v78; // rdx
  int v79; // eax
  int v80; // ecx
  __int64 v81; // rsi
  int v82; // esi
  int v83; // edx
  int v84; // edx
  int v85; // esi
  int v86; // r11d
  __int64 *v87; // rdi
  int v88; // ecx
  __int64 *v89; // rdx
  __int64 *v90; // r14
  __int64 v91; // rax
  __int64 *v92; // r13
  __int64 v93; // r12
  __int64 *v94; // r15
  int v95; // ebx
  __int64 *v96; // r11
  int v97; // r11d
  __int64 *v98; // rdi
  int v99; // edx
  int v100; // ebx
  __int64 *v101; // rbx
  int v102; // [rsp+Ch] [rbp-144h]
  __int64 *v103; // [rsp+30h] [rbp-120h]
  __int64 v104; // [rsp+40h] [rbp-110h]
  __int64 v105; // [rsp+40h] [rbp-110h]
  _QWORD *v107; // [rsp+50h] [rbp-100h] BYREF
  _QWORD *v108; // [rsp+58h] [rbp-F8h] BYREF
  _QWORD *v109; // [rsp+60h] [rbp-F0h] BYREF
  __int64 *v110; // [rsp+68h] [rbp-E8h] BYREF
  __int64 v111[4]; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v112; // [rsp+90h] [rbp-C0h] BYREF
  __int64 *v113; // [rsp+98h] [rbp-B8h]
  __int64 v114; // [rsp+A0h] [rbp-B0h]
  unsigned int v115; // [rsp+A8h] [rbp-A8h]
  __int64 v116; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v117; // [rsp+B8h] [rbp-98h]
  __int64 v118; // [rsp+C0h] [rbp-90h]
  unsigned int v119; // [rsp+C8h] [rbp-88h]
  __int64 *v120; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v121; // [rsp+D8h] [rbp-78h]
  _BYTE v122[112]; // [rsp+E0h] [rbp-70h] BYREF

  if ( !a3 )
    a3 = *(_DWORD *)(a1 + 24);
  v6 = sub_1BF96D0(**(_QWORD **)a1, a2, a3);
  v7 = (int *)sub_16D40F0((__int64)qword_4FBB390);
  if ( v7 )
    v8 = *v7;
  else
    v8 = qword_4FBB390[2];
  v9 = 7;
  if ( v8 >= 0 )
  {
    v66 = sub_16D40F0((__int64)qword_4FBB390);
    v67 = v66 ? *v66 : LODWORD(qword_4FBB390[2]);
    v9 = 7;
    if ( v67 <= 10 )
    {
      v68 = sub_16D40F0((__int64)qword_4FBB390);
      v69 = v68 ? *v68 : LODWORD(qword_4FBB390[2]);
      v9 = 7;
      if ( (unsigned int)(v69 + 4) <= 0x12 )
      {
        v70 = v69 - 5;
        v6 += v70 * v6 / 10;
        v9 = 7 * v70 / 10 + 7;
      }
    }
  }
  v10 = *(__int64 **)a1;
  *(_DWORD *)(a1 + 44) = v9;
  result = a1 + 112;
  *(_DWORD *)(a1 + 40) = v6;
  v12 = v10[41];
  v13 = v10 + 40;
  v104 = a1 + 112;
  if ( (__int64 *)v12 != v13 )
  {
    while ( 1 )
    {
      v14 = *(_QWORD *)(a1 + 16);
      sub_1E06620(v14);
      v15 = *(_QWORD *)(v14 + 1312);
      result = *(unsigned int *)(v15 + 48);
      if ( !(_DWORD)result )
        goto LABEL_8;
      v16 = *(_QWORD *)(v15 + 32);
      v17 = (result - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v18 = (__int64 *)(v16 + 16LL * v17);
      v19 = *v18;
      if ( v12 != *v18 )
      {
        v26 = 1;
        while ( v19 != -8 )
        {
          v27 = v26 + 1;
          v17 = (result - 1) & (v26 + v17);
          v18 = (__int64 *)(v16 + 16LL * v17);
          v19 = *v18;
          if ( *v18 == v12 )
            goto LABEL_11;
          v26 = v27;
        }
        goto LABEL_8;
      }
LABEL_11:
      result = v16 + 16 * result;
      if ( v18 != (__int64 *)result && v18[1] )
      {
        v20 = *(_DWORD *)(a1 + 136);
        v116 = v12;
        if ( !v20 )
        {
          ++*(_QWORD *)(a1 + 112);
          goto LABEL_89;
        }
        v21 = *(_QWORD *)(a1 + 120);
        v22 = (v20 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v23 = (__int64 *)(v21 + 16LL * v22);
        v24 = *v23;
        if ( *v23 != v12 )
        {
          v77 = 1;
          v78 = 0;
          while ( v24 != -8 )
          {
            if ( v24 != -16 || v78 )
              v23 = v78;
            v22 = (v20 - 1) & (v77 + v22);
            v96 = (__int64 *)(v21 + 16LL * v22);
            v24 = *v96;
            if ( *v96 == v12 )
            {
              result = v96[1];
              goto LABEL_16;
            }
            ++v77;
            v78 = v23;
            v23 = (__int64 *)(v21 + 16LL * v22);
          }
          if ( !v78 )
            v78 = v23;
          v79 = *(_DWORD *)(a1 + 128);
          ++*(_QWORD *)(a1 + 112);
          v80 = v79 + 1;
          if ( 4 * (v79 + 1) < 3 * v20 )
          {
            v81 = v12;
            if ( v20 - *(_DWORD *)(a1 + 132) - v80 > v20 >> 3 )
              goto LABEL_85;
            v82 = v20;
            goto LABEL_90;
          }
LABEL_89:
          v82 = 2 * v20;
LABEL_90:
          sub_1DFD130(v104, v82);
          sub_1DF9290(v104, &v116, &v120);
          v78 = v120;
          v81 = v116;
          v80 = *(_DWORD *)(a1 + 128) + 1;
LABEL_85:
          *(_DWORD *)(a1 + 128) = v80;
          if ( *v78 != -8 )
            --*(_DWORD *)(a1 + 132);
          *v78 = v81;
          result = 0;
          v78[1] = 0;
          goto LABEL_16;
        }
        result = v23[1];
LABEL_16:
        *(_QWORD *)(result + 16) = *(_QWORD *)(a1 + 40);
        v12 = *(_QWORD *)(v12 + 8);
        if ( v13 == (__int64 *)v12 )
          break;
      }
      else
      {
LABEL_8:
        v12 = *(_QWORD *)(v12 + 8);
        if ( v13 == (__int64 *)v12 )
          break;
      }
    }
  }
  v25 = a2 - 90;
  if ( (unsigned int)v25 <= 0x1E )
  {
    result = 1073742849;
    if ( _bittest64(&result, v25) )
    {
      if ( a4 )
      {
        result = sub_21EA9F0(a1);
        if ( (_BYTE)result )
        {
          v28 = *(__int64 **)a1;
          v116 = 0;
          v120 = (__int64 *)v122;
          v29 = v28 + 40;
          v121 = 0x800000000LL;
          v117 = 0;
          v118 = 0;
          v119 = 0;
          v30 = v29[1];
          v111[0] = (__int64)&v112;
          v112 = 0;
          v113 = 0;
          v114 = 0;
          v115 = 0;
          v111[1] = a1;
          v111[2] = (__int64)&v116;
          if ( (__int64 *)v30 == v29 )
          {
            v71 = 0;
            if ( !*(_BYTE *)(a1 + 48) )
            {
LABEL_73:
              j___libc_free_0(v71);
              return j___libc_free_0(v113);
            }
            v107 = (_QWORD *)v30;
            v39 = 0;
          }
          else
          {
            do
            {
              v31 = *(_QWORD *)(a1 + 16);
              sub_1E06620(v31);
              v34 = *(_QWORD *)(v31 + 1312);
              v35 = *(unsigned int *)(v34 + 48);
              if ( (_DWORD)v35 )
              {
                v36 = *(_QWORD *)(v34 + 32);
                v37 = (v35 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
                v38 = (__int64 *)(v36 + 16LL * v37);
                v32 = *v38;
                if ( *v38 == v30 )
                {
LABEL_29:
                  if ( v38 != (__int64 *)(v36 + 16 * v35) && v38[1] )
                    sub_1E03570(v111, v30);
                }
                else
                {
                  v83 = 1;
                  while ( v32 != -8 )
                  {
                    v33 = v83 + 1;
                    v37 = (v35 - 1) & (v83 + v37);
                    v38 = (__int64 *)(v36 + 16LL * v37);
                    v32 = *v38;
                    if ( *v38 == v30 )
                      goto LABEL_29;
                    v83 = v33;
                  }
                }
              }
              v30 = *(_QWORD *)(v30 + 8);
            }
            while ( (__int64 *)v30 != v29 );
            if ( !*(_BYTE *)(a1 + 48) )
            {
LABEL_70:
              if ( v120 != (__int64 *)v122 )
                _libc_free((unsigned __int64)v120);
              v71 = v117;
              goto LABEL_73;
            }
            v39 = (unsigned int)v121;
            v107 = *(_QWORD **)(*(_QWORD *)a1 + 328LL);
            if ( HIDWORD(v121) <= (unsigned int)v121 )
            {
              sub_16CD150((__int64)&v120, v122, 0, 8, v32, v33);
              v39 = (unsigned int)v121;
            }
          }
          v120[v39] = (__int64)v107;
          LODWORD(v121) = v121 + 1;
          v109 = v107;
          v40 = (__int64 *)(v117 + 16LL * v119);
          v41 = sub_1DF9680((__int64)&v116, (__int64 *)&v109, &v110);
          v42 = v110;
          if ( !v41 )
            v42 = (__int64 *)(v117 + 16LL * v119);
          if ( v42 == v40 )
          {
            v100 = *(_DWORD *)(a1 + 40);
            *((_DWORD *)sub_1E03450((__int64)&v116, (__int64 *)&v107) + 2) = v100;
          }
          LODWORD(v43) = v121;
          if ( (_DWORD)v121 )
          {
            while ( 1 )
            {
              v44 = v115;
              v45 = (_QWORD *)v120[(unsigned int)v43 - 1];
              LODWORD(v121) = v43 - 1;
              v108 = v45;
              if ( !v115 )
                break;
              v46 = (v115 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
              v47 = &v113[2 * v46];
              v48 = (_QWORD *)*v47;
              if ( v45 == (_QWORD *)*v47 )
              {
                v49 = *((_DWORD *)v47 + 2);
                goto LABEL_44;
              }
              v86 = 1;
              v87 = 0;
              while ( v48 != (_QWORD *)-8LL )
              {
                if ( v87 || v48 != (_QWORD *)-16LL )
                  v47 = v87;
                v46 = (v115 - 1) & (v86 + v46);
                v101 = &v113[2 * v46];
                v48 = (_QWORD *)*v101;
                if ( v45 == (_QWORD *)*v101 )
                {
                  v49 = *((_DWORD *)v101 + 2);
                  goto LABEL_44;
                }
                ++v86;
                v87 = v47;
                v47 = &v113[2 * v46];
              }
              if ( !v87 )
                v87 = v47;
              ++v112;
              v88 = v114 + 1;
              if ( 4 * ((int)v114 + 1) >= 3 * v115 )
                goto LABEL_112;
              if ( v115 - HIDWORD(v114) - v88 <= v115 >> 3 )
                goto LABEL_113;
LABEL_108:
              LODWORD(v114) = v88;
              if ( *v87 != -8 )
                --HIDWORD(v114);
              *v87 = (__int64)v45;
              v49 = 0;
              v45 = v108;
              *((_DWORD *)v87 + 2) = 0;
LABEL_44:
              v109 = v45;
              if ( !(unsigned __int8)sub_1DF9680((__int64)&v116, (__int64 *)&v109, &v110)
                || v110 == (__int64 *)(v117 + 16LL * v119) )
              {
                v50 = -1;
                v51 = (__int64 *)v108[8];
                v103 = (__int64 *)v108[9];
                if ( v103 == v51 )
                  goto LABEL_67;
                v102 = v49;
                while ( 2 )
                {
                  v52 = *(_QWORD *)(a1 + 16);
                  v53 = *v51;
                  v109 = (_QWORD *)*v51;
                  sub_1E06620(v52);
                  v54 = *(_QWORD *)(v52 + 1312);
                  v55 = *(unsigned int *)(v54 + 48);
                  if ( (_DWORD)v55 )
                  {
                    v56 = *(_QWORD *)(v54 + 32);
                    v57 = (v55 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
                    v58 = (__int64 *)(v56 + 16LL * v57);
                    v59 = *v58;
                    if ( v53 == *v58 )
                    {
LABEL_50:
                      if ( v58 == (__int64 *)(v56 + 16 * v55) || !v58[1] )
                        goto LABEL_47;
                      v60 = v115;
                      if ( v115 )
                      {
                        v61 = (__int64)v109;
                        LODWORD(v62) = (v115 - 1) & (((unsigned int)v109 >> 9) ^ ((unsigned int)v109 >> 4));
                        v63 = &v113[2 * (unsigned int)v62];
                        v64 = (_QWORD *)*v63;
                        if ( (_QWORD *)*v63 == v109 )
                        {
LABEL_54:
                          v65 = *((_DWORD *)v63 + 2);
                          if ( v50 > v65 )
                            v50 = v65;
                          goto LABEL_47;
                        }
                        v97 = 1;
                        v98 = 0;
                        while ( v64 != (_QWORD *)-8LL )
                        {
                          if ( !v98 && v64 == (_QWORD *)-16LL )
                            v98 = v63;
                          v62 = (v115 - 1) & ((_DWORD)v62 + v97);
                          v63 = &v113[2 * v62];
                          v64 = (_QWORD *)*v63;
                          if ( v109 == (_QWORD *)*v63 )
                            goto LABEL_54;
                          ++v97;
                        }
                        if ( !v98 )
                          v98 = v63;
                        ++v112;
                        v99 = v114 + 1;
                        if ( 4 * ((int)v114 + 1) < 3 * v115 )
                        {
                          if ( v115 - HIDWORD(v114) - v99 > v115 >> 3 )
                            goto LABEL_144;
                          goto LABEL_149;
                        }
                      }
                      else
                      {
                        ++v112;
                      }
                      v60 = 2 * v115;
LABEL_149:
                      sub_1E03290((__int64)&v112, v60);
                      sub_1DF9680((__int64)&v112, (__int64 *)&v109, &v110);
                      v98 = v110;
                      v61 = (__int64)v109;
                      v99 = v114 + 1;
LABEL_144:
                      LODWORD(v114) = v99;
                      if ( *v98 != -8 )
                        --HIDWORD(v114);
                      *v98 = v61;
                      v50 = 0;
                      *((_DWORD *)v98 + 2) = 0;
                    }
                    else
                    {
                      v84 = 1;
                      while ( v59 != -8 )
                      {
                        v85 = v84 + 1;
                        v57 = (v55 - 1) & (v84 + v57);
                        v58 = (__int64 *)(v56 + 16LL * v57);
                        v59 = *v58;
                        if ( v53 == *v58 )
                          goto LABEL_50;
                        v84 = v85;
                      }
                    }
                  }
LABEL_47:
                  if ( v103 == ++v51 )
                  {
                    v49 = v102;
                    goto LABEL_67;
                  }
                  continue;
                }
              }
              v50 = *((_DWORD *)v110 + 2);
LABEL_67:
              LODWORD(v43) = v121;
              if ( v49 != v50 )
              {
                *((_DWORD *)sub_1E03450((__int64)&v112, (__int64 *)&v108) + 2) = v50;
                v43 = (unsigned int)v121;
                v74 = (__int64 *)v108[12];
                for ( i = (__int64 *)v108[11]; v74 != i; LODWORD(v121) = v121 + 1 )
                {
                  v76 = *i;
                  if ( HIDWORD(v121) <= (unsigned int)v43 )
                  {
                    sub_16CD150((__int64)&v120, v122, 0, 8, v72, v73);
                    v43 = (unsigned int)v121;
                  }
                  ++i;
                  v120[v43] = v76;
                  v43 = (unsigned int)(v121 + 1);
                }
              }
              if ( !(_DWORD)v43 )
                goto LABEL_69;
            }
            ++v112;
LABEL_112:
            v44 = 2 * v115;
LABEL_113:
            sub_1E03290((__int64)&v112, v44);
            sub_1DF9680((__int64)&v112, (__int64 *)&v108, &v110);
            v87 = v110;
            v45 = v108;
            v88 = v114 + 1;
            goto LABEL_108;
          }
LABEL_69:
          if ( (_DWORD)v114 )
          {
            v89 = v113;
            v90 = &v113[2 * v115];
            if ( v113 != v90 )
            {
              while ( 1 )
              {
                v91 = *v89;
                v92 = v89;
                if ( *v89 != -8 && v91 != -16 )
                  break;
                v89 += 2;
                if ( v90 == v89 )
                  goto LABEL_70;
              }
              if ( v90 != v89 )
              {
                v105 = a1;
                v93 = a1 + 112;
                v94 = &v113[2 * v115];
                do
                {
                  v95 = *((_DWORD *)v92 + 2);
                  v109 = (_QWORD *)v91;
                  if ( (unsigned __int8)sub_1DF9290(v93, (__int64 *)&v109, &v110)
                    && v110 != (__int64 *)(*(_QWORD *)(v105 + 120) + 16LL * *(unsigned int *)(v105 + 136)) )
                  {
                    *(_DWORD *)(v110[1] + 16) = v95;
                  }
                  v92 += 2;
                  if ( v92 == v94 )
                    break;
                  while ( 1 )
                  {
                    v91 = *v92;
                    if ( *v92 != -8 && v91 != -16 )
                      break;
                    v92 += 2;
                    if ( v94 == v92 )
                      goto LABEL_70;
                  }
                }
                while ( v94 != v92 );
              }
            }
          }
          goto LABEL_70;
        }
      }
    }
  }
  return result;
}
