// Function: sub_34C4890
// Address: 0x34c4890
//
__int64 __fastcall sub_34C4890(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // r15
  __int64 v8; // r12
  unsigned __int64 v10; // rbx
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // r14
  __int64 v14; // r12
  __int64 v15; // r8
  __int64 v16; // rdi
  __int64 v17; // rcx
  _QWORD *v18; // rax
  __int64 v19; // r14
  __int64 v20; // r15
  unsigned __int64 v21; // rbx
  __int64 v22; // r13
  unsigned __int64 v23; // r12
  __int64 v24; // rsi
  __int64 v25; // rax
  int v26; // ecx
  __int64 v27; // rsi
  int v28; // ecx
  __int64 *v29; // rax
  __int64 v30; // rdi
  __int64 *v31; // r11
  __int64 v32; // r9
  __int64 v33; // rdi
  unsigned __int64 v34; // r13
  unsigned __int64 v35; // r10
  __int64 *v36; // r12
  __int64 v37; // rbx
  __int64 v38; // r8
  __int64 v39; // rsi
  __int64 v40; // rcx
  _QWORD *v41; // rax
  char *v42; // rax
  __int64 v43; // rax
  int v44; // esi
  __int64 v45; // rcx
  int v46; // esi
  __int64 *v47; // rax
  __int64 v48; // rdi
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 (*v51)(); // rax
  __int64 v52; // rcx
  __int64 v53; // r9
  __int64 v54; // rdx
  unsigned __int64 v55; // rdi
  int v56; // eax
  unsigned __int64 v57; // rax
  int v58; // edx
  __int64 v59; // rdi
  char v60; // al
  int v61; // eax
  char v62; // di
  unsigned __int64 v63; // r12
  __int64 v64; // rsi
  _QWORD *v65; // rax
  char v66; // di
  __int64 v67; // rsi
  _QWORD *v68; // rax
  __int64 v69; // rdi
  __int64 (*v70)(); // rax
  int v71; // r8d
  int v72; // eax
  int v73; // r8d
  __int64 v74; // [rsp+20h] [rbp-260h]
  unsigned __int8 v75; // [rsp+2Fh] [rbp-251h]
  __int64 v76; // [rsp+30h] [rbp-250h]
  __int64 v77; // [rsp+40h] [rbp-240h]
  __int64 *v79; // [rsp+48h] [rbp-238h]
  __int64 v80; // [rsp+50h] [rbp-230h] BYREF
  __int64 v81; // [rsp+58h] [rbp-228h] BYREF
  unsigned __int8 *v82; // [rsp+60h] [rbp-220h] BYREF
  unsigned __int8 *v83; // [rsp+68h] [rbp-218h] BYREF
  int v84; // [rsp+70h] [rbp-210h] BYREF
  __int64 v85; // [rsp+78h] [rbp-208h]
  __int64 v86[2]; // [rsp+80h] [rbp-200h] BYREF
  __int64 v87; // [rsp+90h] [rbp-1F0h] BYREF
  char *v88; // [rsp+98h] [rbp-1E8h]
  __int64 v89; // [rsp+A0h] [rbp-1E0h]
  int v90; // [rsp+A8h] [rbp-1D8h]
  char v91; // [rsp+ACh] [rbp-1D4h]
  char v92; // [rsp+B0h] [rbp-1D0h] BYREF
  unsigned __int8 *v93; // [rsp+F0h] [rbp-190h] BYREF
  __int64 v94; // [rsp+F8h] [rbp-188h]
  _BYTE v95[160]; // [rsp+100h] [rbp-180h] BYREF
  __int64 *v96; // [rsp+1A0h] [rbp-E0h] BYREF
  __int64 v97; // [rsp+1A8h] [rbp-D8h]
  __int64 v98[26]; // [rsp+1B0h] [rbp-D0h] BYREF

  v75 = *(_BYTE *)(a1 + 129);
  if ( v75 )
  {
    v7 = *(_QWORD *)a1;
    v8 = *(_QWORD *)(a1 + 8);
    if ( *(_QWORD *)a1 != v8 )
    {
      v10 = *(_QWORD *)a1;
      do
      {
        v11 = *(_QWORD *)(v10 + 16);
        if ( v11 )
          sub_B91220(v10 + 16, v11);
        v10 += 24LL;
      }
      while ( v8 != v10 );
      *(_QWORD *)(a1 + 8) = v7;
      v8 = *(_QWORD *)a1;
    }
    v12 = (unsigned int)qword_503AC28;
    v13 = *(_QWORD *)(a2 + 328);
    v77 = a2 + 320;
    if ( v13 != a2 + 320 )
    {
      a6 = v8;
      v14 = a2 + 320;
      while ( 1 )
      {
        while ( 1 )
        {
          v15 = v12;
          v16 = v7 - a6;
          v17 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v7 - a6) >> 3);
          if ( v12 == v17 )
          {
            v8 = a6;
            goto LABEL_131;
          }
          if ( !*(_BYTE *)(a1 + 52) )
            break;
          v18 = *(_QWORD **)(a1 + 32);
          a3 = (__int64)&v18[*(unsigned int *)(a1 + 44)];
          if ( v18 == (_QWORD *)a3 )
            goto LABEL_100;
          while ( *v18 != v13 )
          {
            if ( (_QWORD *)a3 == ++v18 )
              goto LABEL_100;
          }
          v13 = *(_QWORD *)(v13 + 8);
          if ( v13 == v14 )
          {
LABEL_18:
            v8 = a6;
            goto LABEL_19;
          }
        }
        if ( !sub_C8CA60(a1 + 24, v13) )
        {
LABEL_100:
          a3 = *(unsigned int *)(v13 + 120);
          if ( !(_DWORD)a3 )
          {
            sub_2E32880((__int64 *)&v93, v13);
            v57 = sub_2E31A10(v13, 0);
            v58 = 0;
            if ( v57 != v13 + 48 )
              v58 = sub_34BE380(v57);
            LODWORD(v96) = v58;
            v97 = v13;
            v98[0] = (__int64)v93;
            if ( v93 )
            {
              sub_B976B0((__int64)&v93, v93, (__int64)v98);
              v93 = 0;
            }
            sub_34C3C80((unsigned __int64 *)a1, (__int64)&v96);
            if ( v98[0] )
              sub_B91220((__int64)v98, v98[0]);
            if ( v93 )
              sub_B91220((__int64)&v93, (__int64)v93);
          }
        }
        v7 = *(_QWORD *)(a1 + 8);
        a6 = *(_QWORD *)a1;
        v12 = (unsigned int)qword_503AC28;
        v13 = *(_QWORD *)(v13 + 8);
        v16 = v7 - *(_QWORD *)a1;
        v15 = (unsigned int)qword_503AC28;
        v17 = 0xAAAAAAAAAAAAAAABLL * (v16 >> 3);
        if ( v13 == v14 )
          goto LABEL_18;
      }
    }
    v15 = (unsigned int)qword_503AC28;
    v16 = v7 - v8;
    v17 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v7 - v8) >> 3);
LABEL_19:
    if ( v17 != v15 )
    {
      v75 = 0;
      if ( (unsigned __int64)v16 <= 0x18 )
      {
LABEL_21:
        v19 = *(_QWORD *)(*(_QWORD *)(a2 + 328) + 8LL);
        if ( v77 == v19 )
          return v75;
        v20 = a1;
        while ( *(_DWORD *)(v19 + 72) <= 1u )
        {
LABEL_51:
          v19 = *(_QWORD *)(v19 + 8);
          if ( v77 == v19 )
            return v75;
        }
        v21 = *(_QWORD *)v20;
        v87 = 0;
        v89 = 8;
        v22 = *(_QWORD *)(v20 + 8);
        v88 = &v92;
        v90 = 0;
        v91 = 1;
        v74 = *(_QWORD *)v19;
        if ( v21 != v22 )
        {
          v23 = v21;
          do
          {
            v24 = *(_QWORD *)(v23 + 16);
            if ( v24 )
              sub_B91220(v23 + 16, v24);
            v23 += 24LL;
          }
          while ( v22 != v23 );
          *(_QWORD *)(v20 + 8) = v21;
        }
        if ( !*(_BYTE *)(v20 + 128) )
          goto LABEL_36;
        v25 = *(_QWORD *)(v20 + 160);
        if ( !v25 )
          goto LABEL_36;
        v26 = *(_DWORD *)(v25 + 24);
        v27 = *(_QWORD *)(v25 + 8);
        if ( v26 )
        {
          v28 = v26 - 1;
          a3 = v28 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
          v29 = (__int64 *)(v27 + 16 * a3);
          v30 = *v29;
          if ( *v29 == v19 )
          {
LABEL_34:
            v76 = v29[1];
            if ( v76 && **(_QWORD **)(v76 + 32) == v19 )
            {
LABEL_49:
              if ( !v91 )
                _libc_free((unsigned __int64)v88);
              goto LABEL_51;
            }
LABEL_36:
            v31 = *(__int64 **)(v19 + 64);
            v32 = *(_QWORD *)v20;
            v33 = (unsigned int)qword_503AC28;
            v34 = *(_QWORD *)(v20 + 8);
            v79 = &v31[*(unsigned int *)(v19 + 72)];
            v35 = *(_QWORD *)v20;
            if ( v31 != v79 )
            {
              v36 = *(__int64 **)(v19 + 64);
              while ( 1 )
              {
                v37 = *v36;
                v38 = v33;
                v39 = v34 - v32;
                v40 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v34 - v32) >> 3);
                if ( v33 == v40 )
                  goto LABEL_116;
                if ( *(_BYTE *)(v20 + 52) )
                {
                  v41 = *(_QWORD **)(v20 + 32);
                  a3 = (__int64)&v41[*(unsigned int *)(v20 + 44)];
                  if ( v41 == (_QWORD *)a3 )
                    goto LABEL_54;
                  while ( v37 != *v41 )
                  {
                    if ( (_QWORD *)a3 == ++v41 )
                      goto LABEL_54;
                  }
                  if ( v79 == ++v36 )
                    goto LABEL_45;
                }
                else
                {
                  if ( sub_C8CA60(v20 + 24, *v36) )
                    goto LABEL_60;
LABEL_54:
                  if ( v37 == v19 )
                    goto LABEL_60;
                  if ( !v91 )
                    goto LABEL_62;
                  v42 = v88;
                  v40 = HIDWORD(v89);
                  a3 = (__int64)&v88[8 * HIDWORD(v89)];
                  if ( v88 != (char *)a3 )
                  {
                    while ( v37 != *(_QWORD *)v42 )
                    {
                      v42 += 8;
                      if ( (char *)a3 == v42 )
                        goto LABEL_97;
                    }
                    goto LABEL_60;
                  }
LABEL_97:
                  if ( HIDWORD(v89) < (unsigned int)v89 )
                  {
                    ++HIDWORD(v89);
                    *(_QWORD *)a3 = v37;
                    ++v87;
                  }
                  else
                  {
LABEL_62:
                    sub_C8CC70((__int64)&v87, v37, a3, v40, v38, v32);
                    if ( !(_BYTE)a3 )
                      goto LABEL_60;
                  }
                  if ( (unsigned __int8)sub_2E31A70(v37) || (unsigned __int8)sub_2E31AC0(v37) )
                    goto LABEL_60;
                  if ( !*(_BYTE *)(v20 + 128) )
                    goto LABEL_71;
                  v43 = *(_QWORD *)(v20 + 160);
                  if ( !v43 )
                    goto LABEL_71;
                  v44 = *(_DWORD *)(v43 + 24);
                  v45 = *(_QWORD *)(v43 + 8);
                  if ( v44 )
                  {
                    v46 = v44 - 1;
                    a3 = v46 & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
                    v47 = (__int64 *)(v45 + 16 * a3);
                    v48 = *v47;
                    if ( v37 == *v47 )
                    {
LABEL_69:
                      v49 = v47[1];
                      goto LABEL_70;
                    }
                    v72 = 1;
                    while ( v48 != -4096 )
                    {
                      v73 = v72 + 1;
                      a3 = v46 & (unsigned int)(v72 + a3);
                      v47 = (__int64 *)(v45 + 16LL * (unsigned int)a3);
                      v48 = *v47;
                      if ( v37 == *v47 )
                        goto LABEL_69;
                      v72 = v73;
                    }
                  }
                  v49 = 0;
LABEL_70:
                  if ( v76 != v49 )
                    goto LABEL_60;
LABEL_71:
                  v50 = *(_QWORD *)(v20 + 136);
                  v80 = 0;
                  v81 = 0;
                  v93 = v95;
                  v94 = 0x400000000LL;
                  v51 = *(__int64 (**)())(*(_QWORD *)v50 + 344LL);
                  if ( v51 != sub_2DB1AE0 )
                  {
                    if ( !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, unsigned __int8 **, __int64))v51)(
                            v50,
                            v37,
                            &v80,
                            &v81,
                            &v93,
                            1) )
                    {
                      v96 = v98;
                      v97 = 0x400000000LL;
                      if ( !(_DWORD)v94 )
                        goto LABEL_74;
                      sub_34BE650((__int64)&v96, (__int64)&v93, a3, v52, (unsigned int)v94, v53);
                      if ( !(_DWORD)v94 || v80 != v19 )
                        goto LABEL_74;
                      v69 = *(_QWORD *)(v20 + 136);
                      v70 = *(__int64 (**)())(*(_QWORD *)v69 + 880LL);
                      if ( v70 != sub_2DB1B20 && !((unsigned __int8 (__fastcall *)(__int64, __int64 **))v70)(v69, &v96) )
                      {
                        if ( !v81 && v77 != *(_QWORD *)(v37 + 8) )
                          v81 = *(_QWORD *)(v37 + 8);
LABEL_74:
                        sub_2E32880((__int64 *)&v82, v37);
                        if ( v80 && (!(_DWORD)v94 || v81) )
                        {
                          (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v20 + 136) + 360LL))(
                            *(_QWORD *)(v20 + 136),
                            v37,
                            0);
                          if ( (_DWORD)v94 )
                          {
                            v54 = v80;
                            if ( v80 == v19 )
                              v54 = v81;
                            (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, __int64 *, _QWORD, unsigned __int8 **, _QWORD))(**(_QWORD **)(v20 + 136) + 368LL))(
                              *(_QWORD *)(v20 + 136),
                              v37,
                              v54,
                              0,
                              v96,
                              (unsigned int)v97,
                              &v82,
                              0);
                          }
                        }
                        v83 = v82;
                        if ( v82 )
                          sub_B96E90((__int64)&v83, (__int64)v82, 1);
                        v55 = sub_2E31A10(v37, 0);
                        if ( v55 == v37 + 48 )
                          v56 = 0;
                        else
                          v56 = sub_34BE380(v55);
                        v84 = v56;
                        v85 = v37;
                        v86[0] = (__int64)v83;
                        if ( v83 )
                        {
                          sub_B976B0((__int64)&v83, v83, (__int64)v86);
                          v83 = 0;
                        }
                        sub_34C3C80((unsigned __int64 *)v20, (__int64)&v84);
                        if ( v86[0] )
                          sub_B91220((__int64)v86, v86[0]);
                        if ( v83 )
                          sub_B91220((__int64)&v83, (__int64)v83);
                        if ( v82 )
                          sub_B91220((__int64)&v82, (__int64)v82);
                      }
                      if ( v96 != v98 )
                        _libc_free((unsigned __int64)v96);
                    }
                    if ( v93 != v95 )
                      _libc_free((unsigned __int64)v93);
                  }
LABEL_60:
                  v34 = *(_QWORD *)(v20 + 8);
                  v32 = *(_QWORD *)v20;
                  ++v36;
                  v33 = (unsigned int)qword_503AC28;
                  v35 = *(_QWORD *)v20;
                  v39 = v34 - *(_QWORD *)v20;
                  v38 = (unsigned int)qword_503AC28;
                  v40 = 0xAAAAAAAAAAAAAAABLL * (v39 >> 3);
                  if ( v79 == v36 )
                    goto LABEL_45;
                }
              }
            }
            v38 = (unsigned int)qword_503AC28;
            v39 = v34 - v32;
            v40 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v34 - v32) >> 3);
LABEL_45:
            if ( v40 != v38 )
              goto LABEL_46;
LABEL_116:
            if ( v34 != v32 )
            {
              v62 = *(_BYTE *)(v20 + 52);
              v63 = v35;
              while ( 1 )
              {
                while ( 1 )
                {
                  v64 = *(_QWORD *)(v63 + 8);
                  if ( v62 )
                    break;
LABEL_125:
                  v63 += 24LL;
                  sub_C8CC70(v20 + 24, v64, a3, v40, v38, v32);
                  v62 = *(_BYTE *)(v20 + 52);
                  if ( v63 == v34 )
                    goto LABEL_124;
                }
                v65 = *(_QWORD **)(v20 + 32);
                v40 = *(unsigned int *)(v20 + 44);
                a3 = (__int64)&v65[v40];
                if ( v65 == (_QWORD *)a3 )
                {
LABEL_127:
                  if ( (unsigned int)v40 >= *(_DWORD *)(v20 + 40) )
                    goto LABEL_125;
                  v40 = (unsigned int)(v40 + 1);
                  v63 += 24LL;
                  *(_DWORD *)(v20 + 44) = v40;
                  *(_QWORD *)a3 = v64;
                  v62 = *(_BYTE *)(v20 + 52);
                  ++*(_QWORD *)(v20 + 24);
                  if ( v63 == v34 )
                    goto LABEL_124;
                }
                else
                {
                  while ( v64 != *v65 )
                  {
                    if ( (_QWORD *)a3 == ++v65 )
                      goto LABEL_127;
                  }
                  v63 += 24LL;
                  if ( v63 == v34 )
                  {
LABEL_124:
                    v35 = *(_QWORD *)v20;
                    v39 = *(_QWORD *)(v20 + 8) - *(_QWORD *)v20;
                    goto LABEL_46;
                  }
                }
              }
            }
            v35 = v34;
            v39 = 0;
LABEL_46:
            if ( (unsigned __int64)v39 > 0x18 )
            {
              v60 = sub_34C4490((__int64 *)v20, v19, v74 & 0xFFFFFFFFFFFFFFF8LL, *(_DWORD *)(v20 + 132));
              v35 = *(_QWORD *)v20;
              v75 |= v60;
              v39 = *(_QWORD *)(v20 + 8) - *(_QWORD *)v20;
            }
            if ( v39 == 24 )
            {
              v59 = *(_QWORD *)(v35 + 8);
              if ( v59 != (*(_QWORD *)v19 & 0xFFFFFFFFFFFFFFF8LL) )
                sub_34BEAF0(v59, v19, *(_QWORD *)(v20 + 136), (__int64 *)(v35 + 16));
            }
            goto LABEL_49;
          }
          v61 = 1;
          while ( v30 != -4096 )
          {
            v71 = v61 + 1;
            a3 = v28 & (unsigned int)(v61 + a3);
            v29 = (__int64 *)(v27 + 16LL * (unsigned int)a3);
            v30 = *v29;
            if ( *v29 == v19 )
              goto LABEL_34;
            v61 = v71;
          }
        }
        v76 = 0;
        goto LABEL_36;
      }
LABEL_140:
      v75 = sub_34C4490((__int64 *)a1, 0, 0, *(_DWORD *)(a1 + 132));
      goto LABEL_21;
    }
LABEL_131:
    if ( v7 == v8 )
    {
LABEL_139:
      v75 = 0;
      if ( *(_QWORD *)(a1 + 8) - *(_QWORD *)a1 <= 0x18u )
        goto LABEL_21;
      goto LABEL_140;
    }
    v66 = *(_BYTE *)(a1 + 52);
    while ( 1 )
    {
      v67 = *(_QWORD *)(v8 + 8);
      if ( !v66 )
        goto LABEL_141;
      v68 = *(_QWORD **)(a1 + 32);
      v17 = *(unsigned int *)(a1 + 44);
      a3 = (__int64)&v68[v17];
      if ( v68 != (_QWORD *)a3 )
      {
        while ( v67 != *v68 )
        {
          if ( (_QWORD *)a3 == ++v68 )
            goto LABEL_142;
        }
        goto LABEL_138;
      }
LABEL_142:
      if ( (unsigned int)v17 < *(_DWORD *)(a1 + 40) )
      {
        v17 = (unsigned int)(v17 + 1);
        *(_DWORD *)(a1 + 44) = v17;
        *(_QWORD *)a3 = v67;
        v66 = *(_BYTE *)(a1 + 52);
        ++*(_QWORD *)(a1 + 24);
      }
      else
      {
LABEL_141:
        sub_C8CC70(a1 + 24, v67, a3, v17, v15, a6);
        v66 = *(_BYTE *)(a1 + 52);
      }
LABEL_138:
      v8 += 24;
      if ( v8 == v7 )
        goto LABEL_139;
    }
  }
  return v75;
}
