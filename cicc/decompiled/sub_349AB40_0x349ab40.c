// Function: sub_349AB40
// Address: 0x349ab40
//
__int64 __fastcall sub_349AB40(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // r14
  __int64 v8; // r12
  __int64 v9; // r8
  __int64 v10; // rax
  _BYTE *v11; // r13
  unsigned __int64 v12; // r12
  __int64 v13; // r14
  unsigned __int64 v14; // r15
  unsigned __int64 *v15; // r14
  unsigned __int64 *v16; // r13
  unsigned __int64 *v17; // r12
  __int64 v18; // r12
  __int64 v19; // rcx
  unsigned int v20; // eax
  int v21; // eax
  __int64 v22; // rax
  __int64 v23; // r13
  unsigned __int64 v24; // rax
  __int64 v25; // rdx
  unsigned int v26; // eax
  __int64 v27; // rdx
  __int64 (__fastcall *v28)(__int64, __int64, __int64 *, __int64); // rax
  int v29; // eax
  __int64 (__fastcall *v30)(__int64, __int64, unsigned int); // rax
  unsigned int v31; // esi
  int v32; // edx
  __int16 v33; // ax
  __int64 v34; // rdx
  unsigned __int64 v35; // r13
  unsigned __int64 v36; // r12
  __int64 v37; // rbx
  unsigned __int64 v38; // r15
  unsigned __int64 *v39; // rbx
  unsigned __int64 v40; // r12
  unsigned __int64 *v41; // rbx
  _DWORD *v42; // rbx
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rax
  unsigned __int64 *v45; // r9
  __int64 v46; // r13
  __int64 *v47; // rbx
  bool v48; // cl
  __int64 v49; // r12
  __int64 v50; // rax
  unsigned __int64 v51; // r15
  __int64 v52; // rdx
  __int64 v53; // rdx
  __int16 v54; // ax
  unsigned __int16 v55; // cx
  __int64 v56; // rdi
  __int64 (__fastcall *v57)(__int64, __int64, unsigned int); // rax
  __int64 v58; // rdx
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  __int16 v63; // ax
  unsigned int v64; // eax
  __int64 *v65; // r13
  __int64 v66; // rdx
  __int64 v67; // r14
  unsigned int v68; // r15d
  __int64 v69; // r8
  __int64 v70; // r9
  unsigned __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // r13
  __int64 v74; // rcx
  __int64 v75; // rdx
  __int64 (__fastcall *v76)(__int64, __int64, __int64 *, __int64); // rax
  __int64 (__fastcall *v77)(__int64, __int64, unsigned int); // rax
  __int64 v78; // rdx
  int v79; // edx
  __int16 v80; // ax
  __int64 v82; // rdx
  __int64 v83; // rdi
  __int64 (__fastcall *v84)(__int64, __int64, unsigned int); // rax
  __int64 v85; // rdx
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r9
  __int16 v90; // ax
  __int64 v91; // rax
  unsigned int v92; // eax
  __int64 *v93; // r13
  __int64 v94; // rdx
  __int64 v95; // r14
  unsigned int v96; // r15d
  __int64 v97; // r9
  unsigned int v98; // r14d
  unsigned __int16 v99; // cx
  unsigned __int16 v100; // ax
  bool v101; // di
  int v102; // edi
  __int64 v103; // rax
  __int64 v104; // rdi
  int v105; // r12d
  unsigned __int64 v106; // rax
  __int64 v107; // r15
  __int64 v108; // rbx
  __int64 v109; // rsi
  __int64 v110; // rax
  int v111; // eax
  _DWORD *v112; // r12
  int v114; // [rsp+10h] [rbp-450h]
  char v115; // [rsp+10h] [rbp-450h]
  unsigned int v116; // [rsp+14h] [rbp-44Ch]
  __int64 *v118; // [rsp+20h] [rbp-440h]
  unsigned int v119; // [rsp+28h] [rbp-438h]
  unsigned int v120; // [rsp+2Ch] [rbp-434h]
  bool v121; // [rsp+30h] [rbp-430h]
  char v122; // [rsp+30h] [rbp-430h]
  int v123; // [rsp+30h] [rbp-430h]
  bool v124; // [rsp+38h] [rbp-428h]
  unsigned int v125; // [rsp+38h] [rbp-428h]
  __int64 *v127; // [rsp+40h] [rbp-420h]
  unsigned __int64 *v128; // [rsp+40h] [rbp-420h]
  int v129; // [rsp+40h] [rbp-420h]
  __int64 v131; // [rsp+48h] [rbp-418h]
  unsigned int v132; // [rsp+50h] [rbp-410h]
  __int64 v133; // [rsp+50h] [rbp-410h]
  __int64 v135; // [rsp+60h] [rbp-400h] BYREF
  __int64 v136; // [rsp+68h] [rbp-3F8h]
  unsigned __int64 *v137; // [rsp+70h] [rbp-3F0h] BYREF
  __int64 v138; // [rsp+78h] [rbp-3E8h]
  _BYTE v139[32]; // [rsp+80h] [rbp-3E0h] BYREF
  _BYTE *v140; // [rsp+A0h] [rbp-3C0h] BYREF
  __int64 v141; // [rsp+A8h] [rbp-3B8h]
  _BYTE v142[112]; // [rsp+B0h] [rbp-3B0h] BYREF
  __int64 *v143; // [rsp+120h] [rbp-340h] BYREF
  unsigned int v144; // [rsp+128h] [rbp-338h]
  _BYTE v145[816]; // [rsp+130h] [rbp-330h] BYREF

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  sub_B428A0((__int64 *)&v143, *(_BYTE **)(*(_QWORD *)(a5 - 32) + 56LL), *(_QWORD *)(*(_QWORD *)(a5 - 32) + 64LL));
  v5 = v143;
  v6 = 24LL * v144;
  v7 = (unsigned __int64)&v143[v6];
  if ( v143 != &v143[v6] )
  {
    v119 = 0;
    v116 = 0;
    v120 = 0;
    v132 = 0;
    v118 = &v143[v6];
    while ( 1 )
    {
      v8 = *(_QWORD *)(a1 + 8);
      if ( v8 == *(_QWORD *)(a1 + 16) )
      {
        sub_349A380((unsigned __int64 *)a1, *(int **)(a1 + 8), v5);
        v18 = *(_QWORD *)(a1 + 8);
      }
      else
      {
        v135 = *v5;
        v136 = v5[1];
        v137 = (unsigned __int64 *)v139;
        v138 = 0x100000000LL;
        v9 = *((unsigned int *)v5 + 6);
        if ( (_DWORD)v9 )
          sub_34996D0((__int64)&v137, (__int64)(v5 + 2));
        v140 = v142;
        v141 = 0x200000000LL;
        if ( *((_DWORD *)v5 + 18) )
          sub_3499E80((__int64)&v140, (__int64)(v5 + 8));
        if ( v8 )
        {
          *(_QWORD *)v8 = v135;
          *(_QWORD *)(v8 + 8) = v136;
          *(_QWORD *)(v8 + 16) = v8 + 32;
          *(_QWORD *)(v8 + 24) = 0x100000000LL;
          if ( (_DWORD)v138 )
            sub_34996D0(v8 + 16, (__int64)&v137);
          *(_QWORD *)(v8 + 64) = v8 + 80;
          *(_QWORD *)(v8 + 72) = 0x200000000LL;
          v10 = (unsigned int)v141;
          if ( (_DWORD)v141 )
          {
            sub_3499E80(v8 + 64, (__int64)&v140);
            v10 = (unsigned int)v141;
          }
          *(_QWORD *)(v8 + 200) = 0;
          *(_QWORD *)(v8 + 192) = v8 + 208;
          *(_BYTE *)(v8 + 208) = 0;
          *(_DWORD *)(v8 + 224) = 6;
          *(_QWORD *)(v8 + 232) = 0;
          *(_WORD *)(v8 + 240) = 1;
        }
        else
        {
          v10 = (unsigned int)v141;
        }
        v11 = v140;
        v12 = (unsigned __int64)&v140[56 * v10];
        if ( v140 != (_BYTE *)v12 )
        {
          do
          {
            v13 = *(unsigned int *)(v12 - 40);
            v14 = *(_QWORD *)(v12 - 48);
            v12 -= 56LL;
            v15 = (unsigned __int64 *)(v14 + 32 * v13);
            if ( (unsigned __int64 *)v14 != v15 )
            {
              do
              {
                v15 -= 4;
                if ( (unsigned __int64 *)*v15 != v15 + 2 )
                  j_j___libc_free_0(*v15);
              }
              while ( (unsigned __int64 *)v14 != v15 );
              v14 = *(_QWORD *)(v12 + 8);
            }
            if ( v14 != v12 + 24 )
              _libc_free(v14);
          }
          while ( v11 != (_BYTE *)v12 );
          v12 = (unsigned __int64)v140;
        }
        if ( (_BYTE *)v12 != v142 )
          _libc_free(v12);
        v16 = v137;
        v17 = &v137[4 * (unsigned int)v138];
        if ( v137 != v17 )
        {
          do
          {
            v17 -= 4;
            if ( (unsigned __int64 *)*v17 != v17 + 2 )
              j_j___libc_free_0(*v17);
          }
          while ( v16 != v17 );
          v17 = v137;
        }
        if ( v17 != (unsigned __int64 *)v139 )
          _libc_free((unsigned __int64)v17);
        v18 = *(_QWORD *)(a1 + 8) + 248LL;
        *(_QWORD *)(a1 + 8) = v18;
      }
      v19 = v132;
      v20 = *(_DWORD *)(v18 - 176);
      *(_WORD *)(v18 - 8) = 1;
      if ( v132 >= v20 )
        v20 = v132;
      v132 = v20;
      v21 = *(_DWORD *)(v18 - 248);
      if ( v21 == 1 )
        break;
      if ( v21 == 3 )
      {
        v71 = v119++ - (unsigned __int64)*(unsigned int *)(a5 + 88);
        *(_QWORD *)(v18 - 16) = *(_QWORD *)(a5 + 32 * v71 - 32);
        goto LABEL_61;
      }
      if ( v21 )
      {
        v22 = *(_QWORD *)(v18 - 16);
      }
      else
      {
        v19 = a5;
        v22 = *(_QWORD *)(a5 + 32 * (v120 - (unsigned __int64)(*(_DWORD *)(a5 + 4) & 0x7FFFFFF)));
        *(_QWORD *)(v18 - 16) = v22;
      }
LABEL_40:
      if ( v22 )
      {
        if ( *(_BYTE *)(v18 - 238) )
          goto LABEL_135;
        v23 = *(_QWORD *)(v22 + 8);
        goto LABEL_43;
      }
LABEL_61:
      v5 += 24;
      if ( v118 == v5 )
      {
        v127 = v143;
        v7 = (unsigned __int64)&v143[24 * v144];
        v124 = v132 != 0;
        if ( v143 != (__int64 *)v7 )
        {
          do
          {
            v34 = *(unsigned int *)(v7 - 120);
            v35 = *(_QWORD *)(v7 - 128);
            v7 -= 192LL;
            v36 = v35 + 56 * v34;
            if ( v35 != v36 )
            {
              do
              {
                v37 = *(unsigned int *)(v36 - 40);
                v38 = *(_QWORD *)(v36 - 48);
                v36 -= 56LL;
                v39 = (unsigned __int64 *)(v38 + 32 * v37);
                if ( (unsigned __int64 *)v38 != v39 )
                {
                  do
                  {
                    v39 -= 4;
                    if ( (unsigned __int64 *)*v39 != v39 + 2 )
                      j_j___libc_free_0(*v39);
                  }
                  while ( (unsigned __int64 *)v38 != v39 );
                  v38 = *(_QWORD *)(v36 + 8);
                }
                if ( v38 != v36 + 24 )
                  _libc_free(v38);
              }
              while ( v35 != v36 );
              v35 = *(_QWORD *)(v7 + 64);
            }
            if ( v35 != v7 + 80 )
              _libc_free(v35);
            v40 = *(_QWORD *)(v7 + 16);
            v41 = (unsigned __int64 *)(v40 + 32LL * *(unsigned int *)(v7 + 24));
            if ( (unsigned __int64 *)v40 != v41 )
            {
              do
              {
                v41 -= 4;
                if ( (unsigned __int64 *)*v41 != v41 + 2 )
                  j_j___libc_free_0(*v41);
              }
              while ( (unsigned __int64 *)v40 != v41 );
              v40 = *(_QWORD *)(v7 + 16);
            }
            if ( v40 != v7 + 32 )
              _libc_free(v40);
          }
          while ( v127 != (__int64 *)v7 );
          v7 = (unsigned __int64)v143;
        }
        if ( (_BYTE *)v7 != v145 )
          goto LABEL_85;
        goto LABEL_86;
      }
    }
    if ( *(_BYTE *)(v18 - 238) )
    {
      v72 = *(_QWORD *)(a5 + 32 * (v120 - (unsigned __int64)(*(_DWORD *)(a5 + 4) & 0x7FFFFFF)));
      *(_QWORD *)(v18 - 16) = v72;
      if ( !v72 )
        goto LABEL_61;
LABEL_135:
      v23 = sub_A74920((_QWORD *)(a5 + 72), v120);
LABEL_43:
      v24 = *(unsigned __int8 *)(v23 + 8);
      if ( (_BYTE)v24 == 15 )
      {
        if ( *(_DWORD *)(v23 + 12) != 1 )
          goto LABEL_45;
        v23 = **(_QWORD **)(v23 + 16);
        v24 = *(unsigned __int8 *)(v23 + 8);
      }
      if ( (unsigned __int8)v24 <= 3u || (_BYTE)v24 == 5 || (v24 & 0xF5) == 4 )
        goto LABEL_49;
      if ( (unsigned __int8)v24 <= 0x14u )
      {
        v82 = 1442816;
        if ( !_bittest64(&v82, v24) )
        {
LABEL_45:
          if ( (v24 & 0xFB) == 0xA )
            goto LABEL_46;
          LOBYTE(v19) = (_DWORD)v24 == 18 || (unsigned int)(v24 - 15) <= 1;
          if ( (_BYTE)v19 || (_DWORD)v24 == 20 )
          {
            if ( (unsigned __int8)sub_BCEBA0(v23, 0) )
              goto LABEL_46;
            goto LABEL_49;
          }
        }
LABEL_112:
        v27 = *a2;
        v28 = *(__int64 (__fastcall **)(__int64, __int64, __int64 *, __int64))(*a2 + 720);
        if ( v28 == sub_2FE5500 )
        {
          v29 = *(unsigned __int8 *)(v23 + 8);
          goto LABEL_114;
        }
      }
      else
      {
        if ( (v24 & 0xFB) != 0xA )
          goto LABEL_112;
LABEL_46:
        v135 = sub_9208B0(a3, v23);
        v136 = v25;
        v26 = sub_CA1930(&v135);
        if ( v26 > 0x40 )
        {
          if ( v26 == 128 )
            goto LABEL_138;
        }
        else if ( v26 > 7 )
        {
          v19 = 0x100000001000101LL;
          if ( _bittest64(&v19, v26 - 8) )
            goto LABEL_138;
        }
        else
        {
          if ( v26 != 1 )
            goto LABEL_49;
LABEL_138:
          v23 = sub_BCCE00(*(_QWORD **)v23, v26);
        }
LABEL_49:
        v27 = *a2;
        v28 = *(__int64 (__fastcall **)(__int64, __int64, __int64 *, __int64))(*a2 + 720);
        if ( v28 == sub_2FE5500 )
        {
          v29 = *(unsigned __int8 *)(v23 + 8);
          if ( (_BYTE)v29 == 14 )
          {
            v30 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(v27 + 32);
            v31 = *(_DWORD *)(v23 + 8) >> 8;
            if ( v30 == sub_2D42F30 )
            {
              v32 = sub_AE2980(a3, v31)[1];
              v33 = 2;
              if ( v32 != 1 )
              {
                v33 = 3;
                if ( v32 != 2 )
                {
                  v33 = 4;
                  if ( v32 != 4 )
                  {
                    v33 = 5;
                    if ( v32 != 8 )
                    {
                      v33 = 6;
                      if ( v32 != 16 )
                      {
                        v33 = 7;
                        if ( v32 != 32 )
                        {
                          v33 = 8;
                          if ( v32 != 64 )
                            v33 = 8 * (v32 == 128) + 1;
                        }
                      }
                    }
                  }
                }
              }
            }
            else
            {
              v33 = v30((__int64)a2, a3, v31);
              if ( !v33 )
                v33 = 1;
            }
LABEL_60:
            ++v120;
            *(_WORD *)(v18 - 8) = v33;
            goto LABEL_61;
          }
LABEL_114:
          if ( (unsigned int)(v29 - 17) > 1 )
          {
            v33 = sub_30097B0(v23, 1, v27, v19, v9);
            if ( !v33 )
              v33 = 1;
            goto LABEL_60;
          }
          v56 = *(_QWORD *)(v23 + 24);
          if ( *(_BYTE *)(v56 + 8) == 14 )
          {
            v57 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(v27 + 32);
            v58 = *(_DWORD *)(v56 + 8) >> 8;
            if ( v57 == sub_2D42F30 )
            {
              v59 = (unsigned int)sub_AE2980(a3, v58)[1];
              v63 = 2;
              if ( (_DWORD)v59 != 1 )
              {
                v63 = 3;
                if ( (_DWORD)v59 != 2 )
                {
                  v63 = 4;
                  if ( (_DWORD)v59 != 4 )
                  {
                    v63 = 5;
                    if ( (_DWORD)v59 != 8 )
                    {
                      v63 = 6;
                      if ( (_DWORD)v59 != 16 )
                      {
                        v63 = 7;
                        if ( (_DWORD)v59 != 32 )
                        {
                          v63 = 8;
                          if ( (_DWORD)v59 != 64 )
                            v63 = 9 * ((_DWORD)v59 == 128);
                        }
                      }
                    }
                  }
                }
              }
            }
            else
            {
              v63 = v57((__int64)a2, a3, v58);
            }
            LOWORD(v135) = v63;
            v136 = 0;
            v56 = sub_3007410((__int64)&v135, *(__int64 **)v23, v59, v60, v61, v62);
          }
          v114 = *(_DWORD *)(v23 + 32);
          v122 = *(_BYTE *)(v23 + 8);
          v64 = sub_30097B0(v56, 0, v27, v19, v9);
          v65 = *(__int64 **)v23;
          v67 = v66;
          v68 = v64;
          LODWORD(v135) = v114;
          BYTE4(v135) = v122 == 18;
          if ( v122 == 18 )
            v33 = sub_2D43AD0(v64, v114);
          else
            v33 = sub_2D43050(v64, v114);
          if ( !v33 )
            v33 = sub_3009450(v65, v68, v67, v135, v69, v70);
LABEL_155:
          if ( !v33 )
            v33 = 1;
          goto LABEL_60;
        }
      }
      v33 = v28((__int64)a2, a3, (__int64 *)v23, 1);
      goto LABEL_155;
    }
    v73 = *(_QWORD *)(a5 + 8);
    v74 = *(unsigned __int8 *)(v73 + 8);
    v75 = *a2;
    v76 = *(__int64 (__fastcall **)(__int64, __int64, __int64 *, __int64))(*a2 + 720);
    if ( (_BYTE)v74 == 15 )
    {
      v73 = *(_QWORD *)(*(_QWORD *)(v73 + 16) + 8LL * v116);
      if ( v76 == sub_2FE5500 )
      {
        v74 = *(unsigned __int8 *)(v73 + 8);
        if ( (_BYTE)v74 == 14 )
          goto LABEL_144;
        goto LABEL_172;
      }
    }
    else if ( v76 == sub_2FE5500 )
    {
      if ( (_BYTE)v74 == 14 )
      {
LABEL_144:
        v77 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(v75 + 32);
        v78 = *(_DWORD *)(v73 + 8) >> 8;
        if ( v77 == sub_2D42F30 )
        {
          v79 = sub_AE2980(a3, v78)[1];
          v80 = 2;
          if ( v79 != 1 )
          {
            v80 = 3;
            if ( v79 != 2 )
            {
              v80 = 4;
              if ( v79 != 4 )
              {
                v80 = 5;
                if ( v79 != 8 )
                {
                  v80 = 6;
                  if ( v79 != 16 )
                  {
                    v80 = 7;
                    if ( v79 != 32 )
                    {
                      v80 = 8;
                      if ( v79 != 64 )
                        v80 = 9 * (v79 == 128);
                    }
                  }
                }
              }
            }
          }
        }
        else
        {
          v80 = v77((__int64)a2, a3, v78);
        }
        goto LABEL_167;
      }
LABEL_172:
      if ( (unsigned int)(unsigned __int8)v74 - 17 > 1 )
      {
        v80 = sub_30097B0(v73, 0, v75, v74, v9);
      }
      else
      {
        v83 = *(_QWORD *)(v73 + 24);
        if ( *(_BYTE *)(v83 + 8) == 14 )
        {
          v84 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(v75 + 32);
          v85 = *(_DWORD *)(v83 + 8) >> 8;
          if ( v84 == sub_2D42F30 )
          {
            v86 = (unsigned int)sub_AE2980(a3, v85)[1];
            v90 = 2;
            if ( (_DWORD)v86 != 1 )
            {
              v90 = 3;
              if ( (_DWORD)v86 != 2 )
              {
                v90 = 4;
                if ( (_DWORD)v86 != 4 )
                {
                  v90 = 5;
                  if ( (_DWORD)v86 != 8 )
                  {
                    v90 = 6;
                    if ( (_DWORD)v86 != 16 )
                    {
                      v90 = 7;
                      if ( (_DWORD)v86 != 32 )
                      {
                        v90 = 8;
                        if ( (_DWORD)v86 != 64 )
                          v90 = 9 * ((_DWORD)v86 == 128);
                      }
                    }
                  }
                }
              }
            }
          }
          else
          {
            v90 = v84((__int64)a2, a3, v85);
          }
          LOWORD(v135) = v90;
          v136 = 0;
          v91 = sub_3007410((__int64)&v135, *(__int64 **)v73, v86, v87, v88, v89);
          v74 = *(unsigned __int8 *)(v73 + 8);
          v83 = v91;
        }
        v115 = v74;
        v123 = *(_DWORD *)(v73 + 32);
        v92 = sub_30097B0(v83, 0, v75, v74, v9);
        v93 = *(__int64 **)v73;
        v95 = v94;
        v96 = v92;
        LODWORD(v135) = v123;
        BYTE4(v135) = v115 == 18;
        if ( v115 == 18 )
          v80 = sub_2D43AD0(v92, v123);
        else
          v80 = sub_2D43050(v92, v123);
        if ( !v80 )
          v80 = sub_3009450(v93, v96, v95, v135, v9, v97);
      }
      goto LABEL_167;
    }
    v80 = v76((__int64)a2, a3, (__int64 *)v73, 0);
LABEL_167:
    *(_WORD *)(v18 - 8) = v80;
    v22 = *(_QWORD *)(v18 - 16);
    ++v116;
    goto LABEL_40;
  }
  if ( v143 == (__int64 *)v145 )
    goto LABEL_224;
  v132 = 0;
  v124 = 0;
LABEL_85:
  _libc_free(v7);
LABEL_86:
  v42 = *(_DWORD **)(a1 + 8);
  v43 = *(_QWORD *)a1;
  v121 = v124 && v42 != *(_DWORD **)a1;
  if ( !v121 )
    goto LABEL_87;
  v129 = -1;
  v125 = 0;
  v98 = 0;
  do
  {
    v106 = 0xEF7BDEF7BDEF7BDFLL * ((__int64)((__int64)v42 - v43) >> 3);
    if ( (_DWORD)v106 )
    {
      v107 = 0;
      v105 = 0;
      v108 = 248LL * (unsigned int)v106;
      while ( 1 )
      {
        v109 = v43 + v107;
        if ( *(_DWORD *)(v43 + v107) == 2 )
          goto LABEL_213;
        v110 = *(int *)(v109 + 4);
        if ( (_DWORD)v110 != -1 )
        {
          v99 = *(_WORD *)(v109 + 240);
          v100 = *(_WORD *)(v43 + 248 * v110 + 240);
          if ( v99 != v100 )
            break;
        }
LABEL_211:
        v111 = (*(__int64 (__fastcall **)(__int64 *, __int64, _QWORD))(*a2 + 2464))(a2, v109, v98);
        if ( v111 == -1 )
        {
          v105 = -1;
          v42 = *(_DWORD **)(a1 + 8);
          v43 = *(_QWORD *)a1;
          goto LABEL_204;
        }
        v105 += v111;
        v43 = *(_QWORD *)a1;
LABEL_213:
        v107 += 248;
        if ( v108 == v107 )
        {
          v42 = *(_DWORD **)(a1 + 8);
          goto LABEL_204;
        }
      }
      v101 = (unsigned __int16)(v100 - 17) <= 0x6Cu || (unsigned __int16)(v100 - 2) <= 7u;
      if ( (unsigned __int16)(v99 - 2) <= 7u
        || (unsigned __int16)(v99 - 17) <= 0x6Cu
        || (unsigned __int16)(v99 - 176) <= 0x1Fu )
      {
        if ( v101 )
          goto LABEL_198;
        v101 = v121;
      }
      else if ( v101 )
      {
        goto LABEL_203;
      }
      if ( (unsigned __int16)(v100 - 176) <= 0x1Fu != v101 )
        goto LABEL_203;
LABEL_198:
      v102 = v100;
      if ( v100 <= 1u
        || (unsigned __int16)(v100 - 504) <= 7u
        || (v103 = 16LL * (v100 - 1) + 71615648, v104 = *(_QWORD *)&byte_444C4A0[16 * v102 - 16], v99 <= 1u)
        || (unsigned __int16)(v99 - 504) <= 7u )
      {
        BUG();
      }
      if ( *(_QWORD *)&byte_444C4A0[16 * v99 - 16] != v104 || byte_444C4A0[16 * v99 - 8] != *(_BYTE *)(v103 + 8) )
      {
LABEL_203:
        v105 = -1;
        v42 = *(_DWORD **)(a1 + 8);
        goto LABEL_204;
      }
      goto LABEL_211;
    }
    v105 = 0;
LABEL_204:
    if ( v105 > v129 )
    {
      v129 = v105;
      v125 = v98;
    }
    ++v98;
  }
  while ( v98 < v132 );
  if ( (_DWORD *)v43 != v42 )
  {
    v112 = (_DWORD *)v43;
    do
    {
      if ( *v112 != 2 )
        sub_B3C4B0((__int64)v112, v125);
      v112 += 62;
    }
    while ( v42 != v112 );
LABEL_224:
    v42 = *(_DWORD **)(a1 + 8);
    v43 = *(_QWORD *)a1;
LABEL_87:
    v44 = 0xEF7BDEF7BDEF7BDFLL * ((__int64)((__int64)v42 - v43) >> 3);
    if ( (_DWORD)v44 )
    {
      v45 = (unsigned __int64 *)a1;
      v46 = 0;
      v133 = 248LL * (unsigned int)(v44 - 1);
      v47 = a2;
      while ( 1 )
      {
        v49 = v43 + v46;
        v50 = *(int *)(v43 + v46 + 4);
        if ( (_DWORD)v50 == -1 )
          goto LABEL_95;
        v51 = v43 + 248 * v50;
        if ( *(_WORD *)(v49 + 240) == *(_WORD *)(v51 + 240) )
          goto LABEL_95;
        v128 = v45;
        (*(void (__fastcall **)(__int64 *, __int64, _QWORD, _QWORD))(*v47 + 2496))(
          v47,
          a4,
          *(_QWORD *)(v49 + 192),
          *(_QWORD *)(v49 + 200));
        v131 = v52;
        (*(void (__fastcall **)(__int64 *, __int64, _QWORD, _QWORD, _QWORD))(*v47 + 2496))(
          v47,
          a4,
          *(_QWORD *)(v51 + 192),
          *(_QWORD *)(v51 + 200),
          *(unsigned __int16 *)(v51 + 240));
        v45 = v128;
        v54 = *(_WORD *)(v51 + 240);
        v55 = v54 - 2;
        if ( (unsigned __int16)(*(_WORD *)(v49 + 240) - 2) > 0xE2u )
        {
          if ( v55 <= 7u
            || (unsigned __int16)(v54 - 17) <= 0x6Cu
            || (unsigned __int16)(v54 - 176) <= 0x1Fu
            || (v48 = (unsigned __int16)(v54 - 126) <= 0x31u || (unsigned __int16)(v54 - 10) <= 6u) )
          {
LABEL_234:
            sub_C64ED0("Unsupported asm: input constraint with a matching output constraint of incompatible type!", 1u);
          }
        }
        else
        {
          if ( v55 <= 7u
            || (unsigned __int16)(v54 - 17) <= 0x6Cu
            || (unsigned __int16)(v54 - 176) <= 0x1Fu
            || (unsigned __int16)(v54 - 126) <= 0x31u
            || (unsigned __int16)(v54 - 10) <= 6u )
          {
            goto LABEL_94;
          }
          v48 = 1;
        }
        if ( (unsigned __int16)(v54 - 208) <= 0x14u != v48 )
          goto LABEL_234;
LABEL_94:
        if ( v131 != v53 )
          goto LABEL_234;
LABEL_95:
        if ( v46 == v133 )
          return a1;
        v43 = *v45;
        v46 += 248;
      }
    }
  }
  return a1;
}
