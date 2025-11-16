// Function: sub_1F372D0
// Address: 0x1f372d0
//
__int64 __fastcall sub_1F372D0(__int64 *a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 *v7; // r14
  __int64 v8; // r13
  unsigned int v9; // r13d
  int v11; // r15d
  unsigned int v12; // ebx
  __int64 *v13; // rdx
  int v14; // eax
  int v15; // r12d
  unsigned int v16; // eax
  int v17; // esi
  int v18; // edx
  int v19; // r10d
  __int64 *v20; // r8
  int v21; // r10d
  __int64 *v22; // r8
  __int64 *v23; // rsi
  unsigned __int64 *v24; // rax
  __int64 v25; // rbx
  int v26; // r8d
  int v27; // r9d
  __int64 v28; // rax
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // rdi
  __int64 (*v33)(); // rax
  __int64 *v34; // r12
  __int64 *v35; // r13
  __int64 v36; // r14
  int v37; // eax
  _QWORD *v38; // rdx
  __int64 *v39; // rbx
  __int64 *v40; // r13
  __int64 v41; // r15
  _QWORD *v42; // rdx
  __int64 v43; // r12
  __int64 v44; // rcx
  _QWORD *v45; // rsi
  __int64 v46; // rdi
  __int64 v47; // rcx
  _QWORD *v48; // rcx
  unsigned int v49; // r14d
  int v50; // esi
  __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // rbx
  __int64 (*v54)(); // rax
  unsigned __int64 *v55; // rbx
  unsigned __int64 v56; // rdx
  __int64 v57; // rax
  int v58; // r8d
  int v59; // r9d
  __int64 v60; // r13
  _BYTE *v61; // r12
  int *v62; // r9
  _BYTE *v63; // r10
  __int64 *v64; // r9
  __int64 v65; // rax
  __int64 v66; // rbx
  __int64 v67; // rax
  __int64 v68; // rsi
  __int64 v69; // [rsp+20h] [rbp-250h]
  __int64 *v71; // [rsp+28h] [rbp-248h]
  __int64 *v72; // [rsp+30h] [rbp-240h]
  unsigned __int8 v74; // [rsp+38h] [rbp-238h]
  _BYTE *v75; // [rsp+38h] [rbp-238h]
  unsigned __int64 v76; // [rsp+40h] [rbp-230h]
  __int64 *v78; // [rsp+48h] [rbp-228h]
  __int64 v79; // [rsp+50h] [rbp-220h]
  __int64 *v80; // [rsp+58h] [rbp-218h]
  __int64 v81; // [rsp+58h] [rbp-218h]
  __int64 *v82; // [rsp+58h] [rbp-218h]
  int *v83; // [rsp+58h] [rbp-218h]
  __int64 v84; // [rsp+60h] [rbp-210h] BYREF
  __int64 v85; // [rsp+68h] [rbp-208h] BYREF
  __int64 v86; // [rsp+70h] [rbp-200h] BYREF
  __int64 v87; // [rsp+78h] [rbp-1F8h]
  __int64 v88; // [rsp+80h] [rbp-1F0h]
  __int64 v89; // [rsp+88h] [rbp-1E8h]
  __int64 v90; // [rsp+90h] [rbp-1E0h] BYREF
  __int64 v91; // [rsp+98h] [rbp-1D8h]
  __int64 v92; // [rsp+A0h] [rbp-1D0h]
  int v93; // [rsp+A8h] [rbp-1C8h]
  _BYTE *v94; // [rsp+B0h] [rbp-1C0h] BYREF
  __int64 v95; // [rsp+B8h] [rbp-1B8h]
  _BYTE v96[48]; // [rsp+C0h] [rbp-1B0h] BYREF
  __int64 v97; // [rsp+F0h] [rbp-180h] BYREF
  __int64 v98; // [rsp+F8h] [rbp-178h]
  __int64 v99; // [rsp+100h] [rbp-170h] BYREF
  __int64 *v100; // [rsp+140h] [rbp-130h] BYREF
  __int64 v101; // [rsp+148h] [rbp-128h]
  _BYTE v102[64]; // [rsp+150h] [rbp-120h] BYREF
  _BYTE *v103; // [rsp+190h] [rbp-E0h] BYREF
  __int64 v104; // [rsp+198h] [rbp-D8h]
  _BYTE v105[208]; // [rsp+1A0h] [rbp-D0h] BYREF

  v6 = a3;
  v7 = a1;
  v8 = *(_QWORD *)(a3 + 32);
  v76 = a4;
  v69 = a6;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v79 = a3 + 24;
  if ( v8 != a3 + 24 )
  {
    while ( 1 )
    {
      if ( **(_WORD **)(v8 + 16) != 45 && **(_WORD **)(v8 + 16) )
      {
LABEL_4:
        v7 = a1;
        v6 = a3;
        goto LABEL_5;
      }
      v11 = *(_DWORD *)(v8 + 40);
      v12 = 1;
      if ( v11 != 1 )
        break;
LABEL_21:
      if ( (*(_BYTE *)v8 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v8 + 46) & 8) != 0 )
          v8 = *(_QWORD *)(v8 + 8);
      }
      v8 = *(_QWORD *)(v8 + 8);
      if ( v8 == v79 )
        goto LABEL_4;
    }
    while ( 1 )
    {
      v15 = *(_DWORD *)(*(_QWORD *)(v8 + 32) + 40LL * v12 + 8);
      if ( !(_DWORD)v89 )
        break;
      a4 = ((_DWORD)v89 - 1) & (unsigned int)(37 * v15);
      v13 = (__int64 *)(v87 + 4 * a4);
      v14 = *(_DWORD *)v13;
      if ( *(_DWORD *)v13 != v15 )
      {
        v21 = 1;
        a6 = 0;
        while ( v14 != -1 )
        {
          if ( a6 || v14 != -2 )
            v13 = (__int64 *)a6;
          a6 = (unsigned int)(v21 + 1);
          a4 = ((_DWORD)v89 - 1) & (unsigned int)(v21 + a4);
          v14 = *(_DWORD *)(v87 + 4LL * (unsigned int)a4);
          if ( v15 == v14 )
            goto LABEL_11;
          ++v21;
          a6 = (__int64)v13;
          v13 = (__int64 *)(v87 + 4LL * (unsigned int)a4);
        }
        if ( !a6 )
          a6 = (__int64)v13;
        ++v86;
        v18 = v88 + 1;
        if ( 4 * ((int)v88 + 1) < (unsigned int)(3 * v89) )
        {
          a4 = (unsigned int)v89 >> 3;
          if ( (int)v89 - HIDWORD(v88) - v18 <= (unsigned int)a4 )
          {
            sub_136B240((__int64)&v86, v89);
            if ( !(_DWORD)v89 )
            {
LABEL_181:
              LODWORD(v88) = v88 + 1;
              BUG();
            }
            v20 = 0;
            v49 = (v89 - 1) & (37 * v15);
            a6 = v87 + 4LL * v49;
            v18 = v88 + 1;
            v50 = 1;
            a4 = *(unsigned int *)a6;
            if ( (_DWORD)a4 != v15 )
            {
              while ( (_DWORD)a4 != -1 )
              {
                if ( (_DWORD)a4 == -2 && !v20 )
                  v20 = (__int64 *)a6;
                v49 = (v89 - 1) & (v50 + v49);
                a6 = v87 + 4LL * v49;
                a4 = *(unsigned int *)a6;
                if ( v15 == (_DWORD)a4 )
                  goto LABEL_33;
                ++v50;
              }
LABEL_18:
              if ( v20 )
                a6 = (__int64)v20;
            }
          }
LABEL_33:
          LODWORD(v88) = v18;
          if ( *(_DWORD *)a6 != -1 )
            --HIDWORD(v88);
          *(_DWORD *)a6 = v15;
          goto LABEL_11;
        }
LABEL_14:
        sub_136B240((__int64)&v86, 2 * v89);
        if ( !(_DWORD)v89 )
          goto LABEL_181;
        a4 = (unsigned int)(v89 - 1);
        v16 = a4 & (37 * v15);
        a6 = v87 + 4LL * v16;
        v17 = *(_DWORD *)a6;
        v18 = v88 + 1;
        if ( v15 != *(_DWORD *)a6 )
        {
          v19 = 1;
          v20 = 0;
          while ( v17 != -1 )
          {
            if ( !v20 && v17 == -2 )
              v20 = (__int64 *)a6;
            v16 = a4 & (v19 + v16);
            a6 = v87 + 4LL * v16;
            v17 = *(_DWORD *)a6;
            if ( v15 == *(_DWORD *)a6 )
              goto LABEL_33;
            ++v19;
          }
          goto LABEL_18;
        }
        goto LABEL_33;
      }
LABEL_11:
      v12 += 2;
      if ( v11 == v12 )
        goto LABEL_21;
    }
    ++v86;
    goto LABEL_14;
  }
LABEL_5:
  if ( a2 )
  {
    v9 = sub_1F34980(v7, (_QWORD *)v6, a5);
    goto LABEL_7;
  }
  v22 = *(__int64 **)(v6 + 72);
  v23 = *(__int64 **)(v6 + 64);
  v97 = 0;
  v24 = (unsigned __int64 *)&v99;
  v98 = 1;
  do
    *v24++ = -8;
  while ( v24 != (unsigned __int64 *)&v100 );
  v100 = (__int64 *)v102;
  v101 = 0x800000000LL;
  sub_1F36F80((__int64)&v97, v23, v22, a4, (__int64)v22, (__int64 *)a6);
  v72 = &v100[(unsigned int)v101];
  if ( v100 != v72 )
  {
    v80 = v100;
    v9 = 0;
    while ( 1 )
    {
      v25 = *v80;
      v74 = sub_1F350E0(v7, v6, *v80);
      if ( v74 )
      {
        if ( v76 )
        {
          if ( v25 != v76 )
            goto LABEL_44;
        }
        else
        {
          if ( !sub_1DD69A0(v25, v6) )
          {
LABEL_44:
            v28 = *(unsigned int *)(a5 + 8);
            if ( (unsigned int)v28 >= *(_DWORD *)(a5 + 12) )
              goto LABEL_93;
            goto LABEL_45;
          }
          if ( !sub_1DD6C00((__int64 *)v25) )
          {
            v28 = *(unsigned int *)(a5 + 8);
            if ( (unsigned int)v28 >= *(_DWORD *)(a5 + 12) )
            {
LABEL_93:
              sub_16CD150(a5, (const void *)(a5 + 16), 0, 8, v26, v27);
              v28 = *(unsigned int *)(a5 + 8);
            }
LABEL_45:
            *(_QWORD *)(*(_QWORD *)a5 + 8 * v28) = v25;
            ++*(_DWORD *)(a5 + 8);
            (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)*v7 + 280LL))(*v7, v25, 0);
            v90 = 0;
            v91 = 0;
            v29 = *(_QWORD *)(v6 + 32);
            v94 = v96;
            v92 = 0;
            v93 = 0;
            v95 = 0x400000000LL;
            if ( v29 != v79 )
            {
              while ( 1 )
              {
                if ( !v29 )
                  BUG();
                v30 = v29;
                if ( (*(_BYTE *)v29 & 4) == 0 && (*(_BYTE *)(v29 + 46) & 8) != 0 )
                {
                  do
                    v30 = *(_QWORD *)(v30 + 8);
                  while ( (*(_BYTE *)(v30 + 46) & 8) != 0 );
                }
                v31 = *(_QWORD *)(v30 + 8);
                if ( **(_WORD **)(v29 + 16) == 45 || !**(_WORD **)(v29 + 16) )
                {
                  sub_1F36790((__int64)v7, v29, v6, v25, (__int64)&v90, (int *)&v94, (__int64)&v86, 1);
                  if ( v31 == v79 )
                    break;
                }
                else
                {
                  sub_1F36040((__int64)v7, v29, v6, (__int64 *)v25, (__int64)&v90, (__int64)&v86);
                  if ( v31 == v79 )
                    break;
                }
                v29 = v31;
              }
            }
            sub_1F351B0((__int64)v7, v25, (__int64)&v94, v69);
            v32 = *v7;
            v84 = 0;
            v103 = v105;
            v85 = 0;
            v104 = 0x400000000LL;
            v33 = *(__int64 (**)())(*(_QWORD *)v32 + 264LL);
            if ( v33 != sub_1D820E0 )
              ((void (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v33)(
                v32,
                v25,
                &v84,
                &v85,
                &v103,
                0);
            v34 = v7;
            sub_1DD9130(v25, *(__int64 **)(v25 + 88), 0);
            v35 = *(__int64 **)(v6 + 88);
            v78 = *(__int64 **)(v6 + 96);
            if ( v35 != v78 )
            {
              do
              {
                v36 = *v35++;
                v37 = sub_1DF1780(v34[2], (_QWORD *)v6, v36);
                sub_1DD8FE0(v25, v36, v37);
              }
              while ( v78 != v35 );
              v7 = v34;
            }
            if ( v103 != v105 )
              _libc_free((unsigned __int64)v103);
            if ( v94 != v96 )
              _libc_free((unsigned __int64)v94);
            j___libc_free_0(v91);
            v9 = v74;
          }
        }
      }
      if ( v72 == ++v80 )
        goto LABEL_66;
    }
  }
  v9 = 0;
LABEL_66:
  if ( !v76 )
    v76 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
  v84 = 0;
  v103 = v105;
  v104 = 0x400000000LL;
  v85 = 0;
  v38 = *(_QWORD **)(v76 + 88);
  if ( (unsigned int)((__int64)(*(_QWORD *)(v76 + 96) - (_QWORD)v38) >> 3) == 1 && *v38 == v6 )
  {
    v54 = *(__int64 (**)())(*(_QWORD *)*v7 + 264LL);
    if ( v54 != sub_1D820E0 )
    {
      if ( ((unsigned __int8 (__fastcall *)(__int64, unsigned __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v54)(
             *v7,
             v76,
             &v84,
             &v85,
             &v103,
             0)
        || (_DWORD)v104
        || v84 && v84 != v6
        || (unsigned int)((__int64)(*(_QWORD *)(v6 + 72) - *(_QWORD *)(v6 + 64)) >> 3) != 1
        || *(_BYTE *)(v6 + 181) )
      {
        if ( !*((_BYTE *)v7 + 48) || !(_BYTE)v9 )
        {
LABEL_84:
          if ( v103 != v105 )
            _libc_free((unsigned __int64)v103);
          goto LABEL_86;
        }
      }
      else
      {
        (*(void (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)*v7 + 280LL))(*v7, v76, 0);
        if ( *((_BYTE *)v7 + 48) )
        {
          v60 = *(_QWORD *)(v6 + 32);
          v61 = v96;
          v90 = 0;
          v91 = 0;
          v62 = (int *)&v94;
          v92 = 0;
          v93 = 0;
          v94 = v96;
          v95 = 0x400000000LL;
          while ( v60 != v79 )
          {
            if ( **(_WORD **)(v60 + 16) != 45 && **(_WORD **)(v60 + 16) )
            {
              v63 = v96;
              v64 = &v86;
              while ( 1 )
              {
                v65 = v60;
                if ( (*(_BYTE *)v60 & 4) == 0 && (*(_BYTE *)(v60 + 46) & 8) != 0 )
                {
                  do
                    v65 = *(_QWORD *)(v65 + 8);
                  while ( (*(_BYTE *)(v65 + 46) & 8) != 0 );
                }
                v75 = v63;
                v66 = *(_QWORD *)(v65 + 8);
                v82 = v64;
                sub_1F36040((__int64)v7, v60, v6, (__int64 *)v76, (__int64)&v90, (__int64)v64);
                sub_1E16240(v60);
                v64 = v82;
                v63 = v75;
                if ( v79 == v66 )
                  break;
                if ( !v66 )
                  BUG();
                v60 = v66;
              }
              v61 = v75;
              break;
            }
            v67 = v60;
            if ( (*(_BYTE *)v60 & 4) == 0 && (*(_BYTE *)(v60 + 46) & 8) != 0 )
            {
              do
                v67 = *(_QWORD *)(v67 + 8);
              while ( (*(_BYTE *)(v67 + 46) & 8) != 0 );
            }
            v68 = v60;
            v83 = v62;
            v60 = *(_QWORD *)(v67 + 8);
            sub_1F36790((__int64)v7, v68, v6, v76, (__int64)&v90, v62, (__int64)&v86, 1);
            v62 = v83;
          }
          sub_1F351B0((__int64)v7, v76, (__int64)&v94, v69);
          if ( v94 != v61 )
            _libc_free((unsigned __int64)v94);
          j___libc_free_0(v91);
        }
        else
        {
          (*(void (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)*v7 + 280LL))(*v7, v76, 0);
          v55 = *(unsigned __int64 **)(v6 + 32);
          if ( v55 != (unsigned __int64 *)v79 && v76 + 24 != v79 )
          {
            if ( v6 != v76 )
              sub_1DD5C00((__int64 *)(v76 + 16), v6 + 16, *(_QWORD *)(v6 + 32), v79);
            if ( (unsigned __int64 *)v79 != v55 )
            {
              v56 = *(_QWORD *)(v6 + 24) & 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)((*v55 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v79;
              *(_QWORD *)(v6 + 24) = *(_QWORD *)(v6 + 24) & 7LL | *v55 & 0xFFFFFFFFFFFFFFF8LL;
              v57 = *(_QWORD *)(v76 + 24);
              *(_QWORD *)(v56 + 8) = v76 + 24;
              *v55 = v57 & 0xFFFFFFFFFFFFFFF8LL | *v55 & 7;
              *(_QWORD *)((v57 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v55;
              *(_QWORD *)(v76 + 24) = v56 | *(_QWORD *)(v76 + 24) & 7LL;
            }
          }
        }
        sub_1DD9130(v76, *(__int64 **)(v76 + 88), 0);
        sub_1DD91F0(v76, (_QWORD *)v6);
        if ( *(_DWORD *)(a5 + 8) >= *(_DWORD *)(a5 + 12) )
          sub_16CD150(a5, (const void *)(a5 + 16), 0, 8, v58, v59);
        *(_QWORD *)(*(_QWORD *)a5 + 8LL * (unsigned int)(*(_DWORD *)(a5 + 8))++) = v76;
        if ( !*((_BYTE *)v7 + 48) )
          goto LABEL_83;
      }
      goto LABEL_71;
    }
  }
  if ( *((_BYTE *)v7 + 48) && (_BYTE)v9 )
  {
LABEL_71:
    v39 = v100;
    v40 = &v100[(unsigned int)v101];
    if ( v40 == v100 )
    {
LABEL_83:
      v9 = 1;
      goto LABEL_84;
    }
    v81 = v6;
    v41 = a5;
    while ( 1 )
    {
      v42 = *(_QWORD **)v41;
      v43 = *v39;
      v44 = 8LL * *(unsigned int *)(v41 + 8);
      v45 = (_QWORD *)(*(_QWORD *)v41 + v44);
      v46 = v44 >> 3;
      v47 = v44 >> 5;
      if ( v47 )
      {
        v48 = &v42[4 * v47];
        while ( v43 != *v42 )
        {
          if ( v43 == v42[1] )
          {
            ++v42;
            goto LABEL_80;
          }
          if ( v43 == v42[2] )
          {
            v42 += 2;
            goto LABEL_80;
          }
          if ( v43 == v42[3] )
          {
            v42 += 3;
            goto LABEL_80;
          }
          v42 += 4;
          if ( v48 == v42 )
          {
            v46 = v45 - v42;
            goto LABEL_103;
          }
        }
        goto LABEL_80;
      }
LABEL_103:
      if ( v46 == 2 )
        goto LABEL_133;
      if ( v46 == 3 )
        break;
      if ( v46 != 1 )
        goto LABEL_81;
LABEL_106:
      if ( v43 != *v42 )
      {
LABEL_81:
        if ( (unsigned int)((__int64)(*(_QWORD *)(v43 + 96) - *(_QWORD *)(v43 + 88)) >> 3) == 1 )
        {
          v90 = 0;
          v91 = 0;
          v94 = v96;
          v95 = 0x400000000LL;
          v92 = 0;
          v93 = 0;
          v51 = *(_QWORD *)(v81 + 32);
          if ( v51 != v79 )
          {
            v71 = v39;
            while ( **(_WORD **)(v51 + 16) == 45 || !**(_WORD **)(v51 + 16) )
            {
              v52 = v51;
              if ( (*(_BYTE *)v51 & 4) == 0 && (*(_BYTE *)(v51 + 46) & 8) != 0 )
              {
                do
                  v52 = *(_QWORD *)(v52 + 8);
                while ( (*(_BYTE *)(v52 + 46) & 8) != 0 );
              }
              v53 = *(_QWORD *)(v52 + 8);
              sub_1F36790((__int64)v7, v51, v81, v43, (__int64)&v90, (int *)&v94, (__int64)&v86, 0);
              if ( v79 == v53 )
                break;
              v51 = v53;
            }
            v39 = v71;
          }
          sub_1F351B0((__int64)v7, v43, (__int64)&v94, v69);
          if ( v94 != v96 )
            _libc_free((unsigned __int64)v94);
          j___libc_free_0(v91);
        }
        goto LABEL_82;
      }
LABEL_80:
      if ( v45 == v42 )
        goto LABEL_81;
LABEL_82:
      if ( v40 == ++v39 )
        goto LABEL_83;
    }
    if ( v43 == *v42 )
      goto LABEL_80;
    ++v42;
LABEL_133:
    if ( v43 == *v42 )
      goto LABEL_80;
    ++v42;
    goto LABEL_106;
  }
LABEL_86:
  if ( v100 != (__int64 *)v102 )
    _libc_free((unsigned __int64)v100);
  if ( (v98 & 1) == 0 )
    j___libc_free_0(v99);
LABEL_7:
  j___libc_free_0(v87);
  return v9;
}
