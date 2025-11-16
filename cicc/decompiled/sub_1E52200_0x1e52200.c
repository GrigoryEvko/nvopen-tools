// Function: sub_1E52200
// Address: 0x1e52200
//
__int64 __fastcall sub_1E52200(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rax
  _QWORD *v3; // rax
  __int64 **v4; // rax
  __int64 v5; // rdi
  __int16 v6; // ax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // r12
  unsigned __int64 v13; // rdi
  __int64 v14; // rbx
  __int64 v15; // r12
  unsigned __int64 v16; // rdi
  __int64 v18; // rax
  __int16 v19; // dx
  __int64 v20; // rax
  __int16 v21; // dx
  bool v22; // al
  unsigned __int64 v23; // rdx
  int v24; // r8d
  int v25; // r9d
  __int64 v26; // rax
  int v27; // r8d
  unsigned int v28; // edx
  __int64 *v29; // rax
  __int64 v30; // r10
  __int64 v31; // rax
  __int64 *v32; // r14
  __int64 v33; // rax
  _QWORD *v34; // rdi
  unsigned int v35; // eax
  __int64 v36; // rdx
  __int64 v37; // rdx
  _BYTE *v38; // r14
  _BYTE *v39; // r15
  _QWORD *v40; // rax
  unsigned __int64 v41; // r12
  _QWORD *v42; // r13
  __int64 v43; // rax
  unsigned __int64 *v44; // rax
  char v45; // bl
  __int64 v46; // rbx
  __int64 v47; // rdi
  __int64 (*v48)(); // rax
  __int64 *v49; // rdx
  unsigned __int64 v50; // rax
  _QWORD *v51; // rsi
  unsigned __int64 v52; // rcx
  __int64 v53; // rdi
  __int64 v54; // rax
  unsigned __int64 v55; // r8
  __int64 v56; // rdi
  __int64 v57; // rsi
  __int64 v58; // rcx
  __int64 v59; // rdx
  unsigned __int64 v60; // rax
  unsigned __int64 v61; // rdx
  int v62; // r8d
  int v63; // r9d
  __int64 v64; // rax
  unsigned __int64 *v65; // r12
  __int64 *v66; // r14
  __int64 v67; // rax
  int v68; // r8d
  int v69; // r9d
  __int64 v70; // rdx
  unsigned int v71; // ecx
  _QWORD *v72; // rdi
  unsigned int v73; // eax
  __int64 v74; // rax
  unsigned __int64 v75; // r12
  unsigned int v76; // eax
  _QWORD *v77; // rax
  _QWORD *j; // rdx
  unsigned __int64 *v79; // rsi
  unsigned __int64 *v80; // rcx
  _QWORD *v81; // rdx
  __int64 v82; // rdi
  __int64 (*v83)(); // rax
  int i; // eax
  __int64 v85; // rax
  __int16 v86; // dx
  char v87; // al
  _QWORD *v88; // rax
  __int64 v89; // [rsp+10h] [rbp-1F0h]
  __int64 v90; // [rsp+18h] [rbp-1E8h]
  _BYTE *v91; // [rsp+20h] [rbp-1E0h]
  _QWORD *v92; // [rsp+28h] [rbp-1D8h]
  __int64 *v93; // [rsp+38h] [rbp-1C8h]
  __int64 v96; // [rsp+58h] [rbp-1A8h]
  __int64 v97; // [rsp+68h] [rbp-198h]
  __int64 *v98; // [rsp+70h] [rbp-190h]
  __int64 v99; // [rsp+70h] [rbp-190h]
  __int64 v100; // [rsp+78h] [rbp-188h]
  int v101; // [rsp+88h] [rbp-178h] BYREF
  int v102; // [rsp+8Ch] [rbp-174h] BYREF
  int v103; // [rsp+90h] [rbp-170h] BYREF
  int v104; // [rsp+98h] [rbp-168h] BYREF
  _BYTE *v105; // [rsp+A0h] [rbp-160h] BYREF
  __int64 v106; // [rsp+A8h] [rbp-158h]
  _BYTE v107[32]; // [rsp+B0h] [rbp-150h] BYREF
  __int64 v108; // [rsp+D0h] [rbp-130h] BYREF
  _QWORD *v109; // [rsp+D8h] [rbp-128h]
  __int64 v110; // [rsp+E0h] [rbp-120h]
  unsigned int v111; // [rsp+E8h] [rbp-118h]
  __int64 v112; // [rsp+F0h] [rbp-110h]
  __int64 v113; // [rsp+F8h] [rbp-108h]
  __int64 v114; // [rsp+100h] [rbp-100h]
  _QWORD *v115; // [rsp+110h] [rbp-F0h] BYREF
  __int64 v116; // [rsp+118h] [rbp-E8h]
  _QWORD v117[8]; // [rsp+120h] [rbp-E0h] BYREF
  unsigned __int64 *v118; // [rsp+160h] [rbp-A0h] BYREF
  __int64 v119; // [rsp+168h] [rbp-98h]
  _BYTE *v120; // [rsp+170h] [rbp-90h] BYREF
  __int64 v121; // [rsp+178h] [rbp-88h]
  __int64 v122; // [rsp+180h] [rbp-80h]
  _BYTE v123[120]; // [rsp+188h] [rbp-78h] BYREF

  v2 = (__int64 *)a1[4];
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v3 = (_QWORD *)sub_15E0530(*v2);
  v4 = (__int64 **)sub_1643270(v3);
  v90 = sub_1599EF0(v4);
  v89 = a1[7];
  v100 = a1[6];
  if ( v100 != v89 )
  {
    while ( 1 )
    {
      v5 = *(_QWORD *)(v100 + 8);
      v6 = *(_WORD *)(v5 + 46);
      v97 = v5;
      if ( (v6 & 4) != 0 || (v6 & 8) == 0 )
      {
        if ( (*(_QWORD *)(*(_QWORD *)(v5 + 16) + 8LL) & 0x10LL) != 0 )
          break;
      }
      else if ( sub_1E15D00(v5, 0x10u, 1) )
      {
        break;
      }
      if ( sub_1E17880(v5) )
        break;
      if ( (unsigned __int8)sub_1E178F0(v5) )
      {
        v85 = *(_QWORD *)(v5 + 16);
        if ( *(_WORD *)v85 != 1 || (*(_BYTE *)(*(_QWORD *)(v5 + 32) + 64LL) & 8) == 0 )
        {
          v86 = *(_WORD *)(v5 + 46);
          if ( (v86 & 4) != 0 || (v86 & 8) == 0 )
            v87 = WORD1(*(_QWORD *)(v85 + 8)) & 1;
          else
            v87 = sub_1E15D00(v5, 0x10000u, 1);
          if ( !v87 )
            break;
        }
        if ( !(unsigned __int8)sub_1E176D0(v5, a2) )
          break;
      }
      v18 = *(_QWORD *)(v5 + 16);
      if ( *(_WORD *)v18 == 1 && (*(_BYTE *)(*(_QWORD *)(v5 + 32) + 64LL) & 8) != 0 )
      {
LABEL_90:
        v119 = 0x400000000LL;
        v118 = (unsigned __int64 *)&v120;
        v61 = sub_1E0A0C0(a1[4]);
        if ( *(_BYTE *)(v5 + 49) == 1 )
          sub_1E41E60(*(__int64 **)(v5 + 56), (__int64)&v118, v61);
        v64 = (unsigned int)v119;
        if ( !(_DWORD)v119 )
        {
          if ( !HIDWORD(v119) )
          {
            sub_16CD150((__int64)&v118, &v120, 0, 8, v62, v63);
            v64 = (unsigned int)v119;
          }
          v118[v64] = v90;
          v64 = (unsigned int)(v119 + 1);
          LODWORD(v119) = v119 + 1;
        }
        v65 = &v118[v64];
        if ( v118 != v65 )
        {
          v66 = (__int64 *)v118;
          do
          {
            v115 = (_QWORD *)*v66;
            v67 = sub_1E51FD0((__int64)&v108, (__int64 *)&v115);
            v70 = *(unsigned int *)(v67 + 8);
            if ( (unsigned int)v70 >= *(_DWORD *)(v67 + 12) )
            {
              v99 = v67;
              sub_16CD150(v67, (const void *)(v67 + 16), 0, 8, v68, v69);
              v67 = v99;
              v70 = *(unsigned int *)(v99 + 8);
            }
            ++v66;
            *(_QWORD *)(*(_QWORD *)v67 + 8 * v70) = v100;
            ++*(_DWORD *)(v67 + 8);
          }
          while ( v65 != (unsigned __int64 *)v66 );
          v65 = v118;
        }
        if ( v65 != (unsigned __int64 *)&v120 )
          _libc_free((unsigned __int64)v65);
        goto LABEL_16;
      }
      v19 = *(_WORD *)(v5 + 46);
      if ( (v19 & 4) != 0 || (v19 & 8) == 0 )
      {
        if ( (*(_QWORD *)(v18 + 8) & 0x10000LL) != 0 )
          goto LABEL_90;
      }
      else if ( sub_1E15D00(v5, 0x10000u, 1) )
      {
        goto LABEL_90;
      }
      v20 = *(_QWORD *)(v5 + 16);
      if ( *(_WORD *)v20 == 1 && (*(_BYTE *)(*(_QWORD *)(v5 + 32) + 64LL) & 0x10) != 0
        || ((v21 = *(_WORD *)(v5 + 46), (v21 & 4) != 0) || (v21 & 8) == 0
          ? (v22 = (*(_QWORD *)(v20 + 8) & 0x20000LL) != 0)
          : (v22 = sub_1E15D00(v5, 0x20000u, 1)),
            v22) )
      {
        v105 = v107;
        v106 = 0x400000000LL;
        v23 = sub_1E0A0C0(a1[4]);
        if ( *(_BYTE *)(v5 + 49) == 1 )
          sub_1E41E60(*(__int64 **)(v5 + 56), (__int64)&v105, v23);
        v26 = (unsigned int)v106;
        if ( !(_DWORD)v106 )
        {
          if ( !HIDWORD(v106) )
          {
            sub_16CD150((__int64)&v105, v107, 0, 8, v24, v25);
            v26 = (unsigned int)v106;
          }
          *(_QWORD *)&v105[8 * v26] = v90;
          v26 = (unsigned int)(v106 + 1);
          LODWORD(v106) = v106 + 1;
        }
        v91 = &v105[8 * v26];
        if ( v105 == v91 )
        {
LABEL_87:
          if ( v91 != v107 )
            _libc_free((unsigned __int64)v91);
          goto LABEL_16;
        }
        v92 = v105;
        while ( 1 )
        {
          if ( !v111 )
            goto LABEL_85;
          v27 = v111 - 1;
          v28 = (v111 - 1) & (((unsigned int)*v92 >> 9) ^ ((unsigned int)*v92 >> 4));
          v29 = &v109[2 * v28];
          v30 = *v29;
          if ( *v92 != *v29 )
          {
            for ( i = 1; ; i = v25 )
            {
              if ( v30 == -8 )
                goto LABEL_85;
              v25 = i + 1;
              v28 = v27 & (i + v28);
              v29 = &v109[2 * v28];
              v30 = *v29;
              if ( *v92 == *v29 )
                break;
            }
          }
          if ( v29 != &v109[2 * v111] )
          {
            v31 = v112 + 56LL * *((unsigned int *)v29 + 2);
            if ( v113 != v31 )
            {
              v32 = *(__int64 **)(v31 + 8);
              v93 = &v32[*(unsigned int *)(v31 + 16)];
              if ( v32 != v93 )
                break;
            }
          }
LABEL_85:
          if ( v91 == (_BYTE *)++v92 )
          {
            v91 = v105;
            goto LABEL_87;
          }
        }
        v98 = *(__int64 **)(v31 + 8);
        while ( 2 )
        {
          v33 = *v98;
          v119 = (__int64)v123;
          v120 = v123;
          v34 = v117;
          v118 = 0;
          v121 = 8;
          LODWORD(v122) = 0;
          v115 = v117;
          v96 = v33;
          v117[0] = v33;
          v116 = 0x800000001LL;
          v35 = 1;
          while ( 1 )
          {
            v36 = v35--;
            v37 = v34[v36 - 1];
            LODWORD(v116) = v35;
            v38 = *(_BYTE **)(v37 + 112);
            v39 = &v38[16 * *(unsigned int *)(v37 + 120)];
            if ( v38 != v39 )
              break;
LABEL_66:
            if ( !v35 )
            {
              v45 = 0;
              goto LABEL_68;
            }
          }
LABEL_54:
          if ( ((*v38 ^ 6) & 6) != 0 )
            goto LABEL_53;
          v40 = (_QWORD *)v119;
          v41 = *(_QWORD *)v38 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v120 == (_BYTE *)v119 )
          {
            v42 = (_QWORD *)(v119 + 8LL * HIDWORD(v121));
            if ( (_QWORD *)v119 == v42 )
            {
              v81 = (_QWORD *)v119;
            }
            else
            {
              do
              {
                if ( v41 == *v40 )
                  break;
                ++v40;
              }
              while ( v42 != v40 );
              v81 = (_QWORD *)(v119 + 8LL * HIDWORD(v121));
            }
          }
          else
          {
            v42 = &v120[8 * (unsigned int)v121];
            v40 = sub_16CC9F0((__int64)&v118, v41);
            if ( v41 == *v40 )
            {
              if ( v120 == (_BYTE *)v119 )
                v81 = &v120[8 * HIDWORD(v121)];
              else
                v81 = &v120[8 * (unsigned int)v121];
            }
            else
            {
              if ( v120 != (_BYTE *)v119 )
              {
                v40 = &v120[8 * (unsigned int)v121];
LABEL_59:
                if ( v42 != v40 )
                  goto LABEL_53;
                if ( v100 == v41 )
                {
                  v34 = v115;
                  v45 = 1;
LABEL_68:
                  if ( v34 != v117 )
                    _libc_free((unsigned __int64)v34);
                  if ( v120 != (_BYTE *)v119 )
                    _libc_free((unsigned __int64)v120);
                  if ( !v45 )
                  {
                    v46 = *(_QWORD *)(v96 + 8);
                    v47 = a1[2];
                    v48 = *(__int64 (**)())(*(_QWORD *)v47 + 592LL);
                    if ( v48 != sub_1D9BA90 )
                    {
                      if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, int *, int *, _QWORD))v48)(
                             v47,
                             *(_QWORD *)(v96 + 8),
                             &v101,
                             &v103,
                             a1[3]) )
                      {
                        v82 = a1[2];
                        v83 = *(__int64 (**)())(*(_QWORD *)v82 + 592LL);
                        if ( v83 != sub_1D9BA90
                          && ((unsigned __int8 (__fastcall *)(__int64, __int64, int *, int *, _QWORD))v83)(
                               v82,
                               v97,
                               &v102,
                               &v104,
                               a1[3])
                          && v101 == v102
                          && v103 < v104 )
                        {
                          goto LABEL_83;
                        }
                      }
                    }
                    if ( !a2 )
                      goto LABEL_83;
                    v49 = **(__int64 ***)(v46 + 56);
                    if ( (*v49 & 4) != 0 )
                      goto LABEL_83;
                    v50 = *v49 & 0xFFFFFFFFFFFFFFF8LL;
                    if ( !v50 )
                      goto LABEL_83;
                    v51 = **(_QWORD ***)(v97 + 56);
                    if ( (*v51 & 4) != 0 )
                      goto LABEL_83;
                    v52 = *v51 & 0xFFFFFFFFFFFFFFF8LL;
                    if ( !v52 || v52 == v50 && v51[1] >= v49[1] )
                      goto LABEL_83;
                    v53 = v51[6];
                    v54 = v51[7];
                    v55 = v51[5];
                    v118 = (unsigned __int64 *)(*v51 & 0xFFFFFFFFFFFFFFF8LL);
                    v122 = v54;
                    v119 = -1;
                    v120 = (_BYTE *)v55;
                    v121 = v53;
                    v56 = v49[5];
                    v57 = v49[6];
                    v58 = v49[7];
                    v59 = *v49;
                    v117[0] = v56;
                    v117[1] = v57;
                    v117[2] = v58;
                    v116 = -1;
                    v60 = v59 & 0xFFFFFFFFFFFFFFF8LL;
                    if ( (v59 & 4) != 0 )
                      v60 = 0;
                    v115 = (_QWORD *)v60;
                    if ( (unsigned __int8)sub_134CB50(a2, (__int64)&v115, (__int64)&v118) )
                    {
LABEL_83:
                      v119 = 0x100000000LL;
                      v118 = (unsigned __int64 *)(v96 | 6);
                      sub_1F01A00(v100, &v118, 1);
                    }
                  }
                  if ( v93 == ++v98 )
                    goto LABEL_85;
                  continue;
                }
                v43 = (unsigned int)v116;
                if ( (unsigned int)v116 >= HIDWORD(v116) )
                {
                  sub_16CD150((__int64)&v115, v117, 0, 8, v27, v25);
                  v43 = (unsigned int)v116;
                }
                v115[v43] = v41;
                v44 = (unsigned __int64 *)v119;
                LODWORD(v116) = v116 + 1;
                if ( v120 == (_BYTE *)v119 )
                {
                  v79 = (unsigned __int64 *)(v119 + 8LL * HIDWORD(v121));
                  if ( (unsigned __int64 *)v119 != v79 )
                  {
                    v80 = 0;
                    while ( v41 != *v44 )
                    {
                      if ( *v44 == -2 )
                        v80 = v44;
                      if ( v79 == ++v44 )
                      {
                        if ( !v80 )
                          goto LABEL_136;
                        *v80 = v41;
                        LODWORD(v122) = v122 - 1;
                        v118 = (unsigned __int64 *)((char *)v118 + 1);
                        break;
                      }
                    }
LABEL_53:
                    v38 += 16;
                    if ( v39 == v38 )
                      goto LABEL_65;
                    goto LABEL_54;
                  }
LABEL_136:
                  if ( HIDWORD(v121) < (unsigned int)v121 )
                  {
                    ++HIDWORD(v121);
                    *v79 = v41;
                    v118 = (unsigned __int64 *)((char *)v118 + 1);
                    goto LABEL_53;
                  }
                }
                v38 += 16;
                sub_16CCBA0((__int64)&v118, v41);
                if ( v39 == v38 )
                {
LABEL_65:
                  v35 = v116;
                  v34 = v115;
                  goto LABEL_66;
                }
                goto LABEL_54;
              }
              v40 = &v120[8 * HIDWORD(v121)];
              v81 = v40;
            }
          }
          break;
        }
        for ( ; v81 != v40; ++v40 )
        {
          if ( *v40 < 0xFFFFFFFFFFFFFFFELL )
            break;
        }
        goto LABEL_59;
      }
LABEL_16:
      v100 += 272;
      if ( v89 == v100 )
        goto LABEL_17;
    }
    ++v108;
    if ( (_DWORD)v110 )
    {
      v71 = 4 * v110;
      v7 = v111;
      if ( (unsigned int)(4 * v110) < 0x40 )
        v71 = 64;
      if ( v71 >= v111 )
      {
LABEL_8:
        v8 = v109;
        v9 = &v109[2 * v7];
        if ( v109 != v9 )
        {
          do
          {
            *v8 = -8;
            v8 += 2;
          }
          while ( v9 != v8 );
        }
        goto LABEL_10;
      }
      v72 = v109;
      if ( (_DWORD)v110 == 1 )
      {
        v75 = 86;
      }
      else
      {
        _BitScanReverse(&v73, v110 - 1);
        v74 = (unsigned int)(1 << (33 - (v73 ^ 0x1F)));
        if ( (int)v74 < 64 )
          v74 = 64;
        if ( (_DWORD)v74 == v111 )
        {
          v110 = 0;
          v88 = &v109[2 * v74];
          do
          {
            if ( v72 )
              *v72 = -8;
            v72 += 2;
          }
          while ( v88 != v72 );
          goto LABEL_11;
        }
        v75 = 4 * (int)v74 / 3u + 1;
      }
      j___libc_free_0(v109);
      v76 = sub_1454B60(v75);
      v111 = v76;
      if ( v76 )
      {
        v77 = (_QWORD *)sub_22077B0(16LL * v76);
        v110 = 0;
        v109 = v77;
        for ( j = &v77[2 * v111]; j != v77; v77 += 2 )
        {
          if ( v77 )
            *v77 = -8;
        }
        goto LABEL_11;
      }
    }
    else
    {
      if ( !HIDWORD(v110) )
        goto LABEL_11;
      v7 = v111;
      if ( v111 <= 0x40 )
        goto LABEL_8;
      j___libc_free_0(v109);
      v111 = 0;
    }
    v109 = 0;
LABEL_10:
    v110 = 0;
LABEL_11:
    v10 = v112;
    v11 = v113;
    v12 = v112;
    if ( v112 != v113 )
    {
      do
      {
        v13 = *(_QWORD *)(v12 + 8);
        if ( v13 != v12 + 24 )
          _libc_free(v13);
        v12 += 56;
      }
      while ( v11 != v12 );
      v113 = v10;
    }
    goto LABEL_16;
  }
LABEL_17:
  v14 = v113;
  v15 = v112;
  if ( v113 != v112 )
  {
    do
    {
      v16 = *(_QWORD *)(v15 + 8);
      if ( v16 != v15 + 24 )
        _libc_free(v16);
      v15 += 56;
    }
    while ( v14 != v15 );
    v15 = v112;
  }
  if ( v15 )
    j_j___libc_free_0(v15, v114 - v15);
  return j___libc_free_0(v109);
}
