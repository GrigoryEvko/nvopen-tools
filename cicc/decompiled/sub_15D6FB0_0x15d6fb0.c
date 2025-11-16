// Function: sub_15D6FB0
// Address: 0x15d6fb0
//
void __fastcall sub_15D6FB0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 *v8; // rax
  unsigned __int64 *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r12
  _QWORD *v13; // rax
  _BYTE *v14; // rdi
  __int64 v15; // rax
  unsigned int v16; // r8d
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // rcx
  _BYTE *v20; // rsi
  unsigned int v21; // eax
  _BYTE *v22; // rdx
  __int64 v23; // rbx
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // rdx
  int v28; // eax
  unsigned int v29; // eax
  __int64 *v30; // r12
  __int64 *v31; // rbx
  __int64 **v32; // r13
  __int64 **v33; // rax
  __int64 v34; // r14
  _QWORD *v35; // r8
  int v36; // edx
  unsigned int v37; // eax
  __int64 v38; // rdi
  int v39; // r9d
  __int64 v40; // r13
  __int64 v41; // rax
  _QWORD *v42; // rax
  _BYTE *v43; // rsi
  __int64 v44; // rax
  unsigned int v45; // edi
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rcx
  _BYTE *v49; // rax
  _BYTE *v50; // rdx
  __int64 v51; // rcx
  _QWORD *v52; // rdi
  int v53; // esi
  int v54; // r8d
  unsigned int v55; // eax
  __int64 v56; // rdx
  char v57; // al
  __int64 v58; // rcx
  unsigned int v59; // eax
  unsigned int v60; // esi
  __int64 v61; // rax
  unsigned int v62; // eax
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 **v65; // rsi
  __int64 *v66; // rbx
  __int64 *v67; // r12
  __int64 v68; // rdi
  __int64 *v69; // rbx
  __int64 *v70; // r12
  __int64 v71; // rdi
  unsigned __int64 v72; // rdi
  _BYTE *v73; // rax
  unsigned int v74; // r10d
  __int64 v75; // r13
  __int64 v76; // rax
  __int64 v77; // r9
  __int64 v78; // rcx
  __int64 v79; // r12
  __int64 v80; // r11
  __int64 i; // rcx
  __int64 v82; // rax
  _BYTE *v83; // rdi
  int v84; // r8d
  _BYTE *v85; // rcx
  __int64 v86; // rcx
  __int64 v87; // rcx
  __int64 v88; // rcx
  _BYTE *v89; // rax
  _BYTE *v90; // rdi
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // [rsp+30h] [rbp-420h]
  int v95; // [rsp+88h] [rbp-3C8h]
  unsigned int v96; // [rsp+8Ch] [rbp-3C4h]
  __int64 *v97; // [rsp+90h] [rbp-3C0h]
  __int64 *v98; // [rsp+98h] [rbp-3B8h] BYREF
  __int64 v99; // [rsp+A8h] [rbp-3A8h] BYREF
  __int64 v100; // [rsp+B0h] [rbp-3A0h] BYREF
  unsigned int v101; // [rsp+B8h] [rbp-398h] BYREF
  _QWORD v102[6]; // [rsp+C0h] [rbp-390h] BYREF
  _QWORD *v103; // [rsp+F0h] [rbp-360h] BYREF
  __int64 v104; // [rsp+F8h] [rbp-358h]
  _QWORD v105[8]; // [rsp+100h] [rbp-350h] BYREF
  __int64 *v106; // [rsp+140h] [rbp-310h] BYREF
  unsigned int v107; // [rsp+148h] [rbp-308h] BYREF
  char v108; // [rsp+150h] [rbp-300h] BYREF
  __int64 v109; // [rsp+190h] [rbp-2C0h] BYREF
  __int64 **v110; // [rsp+198h] [rbp-2B8h]
  __int64 **v111; // [rsp+1A0h] [rbp-2B0h]
  __int64 v112; // [rsp+1A8h] [rbp-2A8h]
  int v113; // [rsp+1B0h] [rbp-2A0h]
  _BYTE v114[72]; // [rsp+1B8h] [rbp-298h] BYREF
  _BYTE *v115; // [rsp+200h] [rbp-250h] BYREF
  __int64 v116; // [rsp+208h] [rbp-248h]
  _BYTE v117[136]; // [rsp+210h] [rbp-240h] BYREF
  __int64 v118; // [rsp+298h] [rbp-1B8h] BYREF
  __int64 v119; // [rsp+2A0h] [rbp-1B0h]
  _QWORD *v120; // [rsp+2A8h] [rbp-1A8h] BYREF
  int v121; // [rsp+2B0h] [rbp-1A0h]
  __int64 v122; // [rsp+2E8h] [rbp-168h] BYREF
  __int64 v123; // [rsp+2F0h] [rbp-160h]
  _QWORD *v124; // [rsp+2F8h] [rbp-158h] BYREF
  unsigned int v125; // [rsp+300h] [rbp-150h]
  __int64 *v126; // [rsp+378h] [rbp-D8h] BYREF
  __int64 v127; // [rsp+380h] [rbp-D0h]
  _BYTE v128[64]; // [rsp+388h] [rbp-C8h] BYREF
  __int64 *v129; // [rsp+3C8h] [rbp-88h] BYREF
  __int64 v130; // [rsp+3D0h] [rbp-80h]
  _BYTE v131[120]; // [rsp+3D8h] [rbp-78h] BYREF

  v6 = a3;
  v98 = a4;
  if ( a3 )
  {
    if ( *a4 )
      v6 = sub_15CC590(a1, a3, *a4);
    else
      v6 = 0;
  }
  v7 = sub_15CC510(a1, v6);
  v93 = v7;
  if ( a4[1] == v7 || (__int64 *)v7 == a4 )
    return;
  v118 = 0;
  v115 = v117;
  v116 = 0x800000000LL;
  v8 = (__int64 *)&v120;
  v119 = 1;
  do
    *v8++ = -8;
  while ( v8 != &v122 );
  v9 = (unsigned __int64 *)&v124;
  v122 = 0;
  v123 = 1;
  do
  {
    *v9 = -8;
    v9 += 2;
  }
  while ( v9 != (unsigned __int64 *)&v126 );
  v126 = (__int64 *)v128;
  v127 = 0x800000000LL;
  v130 = 0x800000000LL;
  v129 = (__int64 *)v131;
  sub_15D6A20((__int64)&v109, (__int64)&v118, (__int64 *)&v98);
  v10 = (__int64)v98;
  v11 = (unsigned int)v116;
  v12 = *((unsigned int *)v98 + 4);
  if ( (unsigned int)v116 >= HIDWORD(v116) )
  {
    sub_16CD150(&v115, v117, 0, 16);
    v11 = (unsigned int)v116;
  }
  v13 = &v115[16 * v11];
  *v13 = v12;
  v14 = v115;
  v13[1] = v10;
  LODWORD(v116) = v116 + 1;
  v15 = 16LL * (unsigned int)v116;
  v16 = *(_DWORD *)&v14[v15 - 16];
  v17 = *(_QWORD *)&v14[v15 - 8];
  v18 = (v15 >> 4) - 1;
  v19 = ((v15 >> 4) - 2) / 2;
  if ( v18 > 0 )
  {
    while ( 1 )
    {
      v20 = &v14[16 * v19];
      v92 = (__int64)&v14[16 * v18];
      if ( *(_DWORD *)v20 <= v16 )
        break;
      *(_DWORD *)v92 = *(_DWORD *)v20;
      *(_QWORD *)(v92 + 8) = *((_QWORD *)v20 + 1);
      v18 = v19;
      if ( v19 <= 0 )
      {
        v92 = (__int64)&v14[16 * v19];
        break;
      }
      v19 = (v19 - 1) / 2;
    }
  }
  else
  {
    v92 = (__int64)&v14[v15 - 16];
  }
  *(_DWORD *)v92 = v16;
  *(_QWORD *)(v92 + 8) = v17;
  v21 = v116;
  if ( (_DWORD)v116 )
  {
    while ( 1 )
    {
      v22 = v115;
      v23 = *((_QWORD *)v115 + 1);
      v96 = *(_DWORD *)(v23 + 16);
      v24 = v21;
      v25 = 16LL * v21;
      if ( v24 != 1 )
        break;
LABEL_19:
      v106 = (__int64 *)v23;
      v107 = v96;
      LODWORD(v116) = v116 - 1;
      sub_15D6E30((__int64)&v109, (__int64)&v122, (__int64 *)&v106, &v107);
      v26 = (unsigned int)v127;
      if ( (unsigned int)v127 >= HIDWORD(v127) )
      {
        sub_16CD150(&v126, v128, 0, 8);
        v26 = (unsigned int)v127;
      }
      v126[v26] = v23;
      v27 = v105;
      LODWORD(v127) = v127 + 1;
      v28 = *(_DWORD *)(v93 + 16);
      v105[0] = v23;
      v109 = 0;
      v95 = v28;
      v112 = 8;
      v103 = v105;
      v104 = 0x800000001LL;
      v113 = 0;
      v110 = (__int64 **)v114;
      v111 = (__int64 **)v114;
      v29 = 1;
      while ( 1 )
      {
        v30 = (__int64 *)v27[v29 - 1];
        LODWORD(v104) = v29 - 1;
        sub_15CF6C0((__int64)&v106, *v30, a2);
        v31 = v106;
        v97 = &v106[v107];
        if ( v106 != v97 )
        {
          while ( 1 )
          {
            v99 = sub_15CC510(a1, *v31);
            v34 = *(unsigned int *)(v99 + 16);
            v33 = v110;
            if ( v111 == v110 )
            {
              v32 = &v110[HIDWORD(v112)];
              if ( v110 == v32 )
              {
                v65 = v110;
              }
              else
              {
                do
                {
                  if ( v30 == *v33 )
                    break;
                  ++v33;
                }
                while ( v32 != v33 );
                v65 = &v110[HIDWORD(v112)];
              }
              goto LABEL_35;
            }
            v32 = &v111[(unsigned int)v112];
            v33 = (__int64 **)sub_16CC9F0(&v109, v30);
            if ( v30 == *v33 )
              break;
            if ( v111 == v110 )
            {
              v33 = &v111[HIDWORD(v112)];
              v65 = v33;
              goto LABEL_35;
            }
            v33 = &v111[(unsigned int)v112];
LABEL_27:
            if ( v32 != v33 )
              goto LABEL_28;
            if ( v96 < (unsigned int)v34 )
            {
              v51 = v99;
              if ( (v123 & 1) != 0 )
              {
                v52 = &v124;
                v53 = 7;
              }
              else
              {
                v52 = v124;
                if ( !v125 )
                  goto LABEL_73;
                v53 = v125 - 1;
              }
              v54 = 1;
              v55 = v53 & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
              v56 = v52[2 * v55];
              if ( v99 != v56 )
              {
                while ( v56 != -8 )
                {
                  v55 = v53 & (v54 + v55);
                  v56 = v52[2 * v55];
                  if ( v99 == v56 )
                    goto LABEL_64;
                  ++v54;
                }
                goto LABEL_73;
              }
LABEL_64:
              v57 = sub_15D0950((__int64)&v122, &v99, v102);
              v58 = v102[0];
              if ( v57 )
              {
                v62 = *(_DWORD *)(v102[0] + 8LL);
                goto LABEL_71;
              }
              ++v122;
              v59 = ((unsigned int)v123 >> 1) + 1;
              if ( (v123 & 1) != 0 )
              {
                v60 = 8;
                if ( 4 * v59 >= 0x18 )
                {
LABEL_90:
                  v60 *= 2;
LABEL_91:
                  sub_15D6B70((__int64)&v122, v60);
                  sub_15D0950((__int64)&v122, &v99, v102);
                  v58 = v102[0];
                  v59 = ((unsigned int)v123 >> 1) + 1;
                  goto LABEL_68;
                }
              }
              else
              {
                v60 = v125;
                if ( 4 * v59 >= 3 * v125 )
                  goto LABEL_90;
              }
              if ( v60 - (v59 + HIDWORD(v123)) <= v60 >> 3 )
                goto LABEL_91;
LABEL_68:
              LODWORD(v123) = v123 & 1 | (2 * v59);
              if ( *(_QWORD *)v58 != -8 )
                --HIDWORD(v123);
              v61 = v99;
              *(_DWORD *)(v58 + 8) = 0;
              *(_QWORD *)v58 = v61;
              v62 = 0;
LABEL_71:
              if ( v96 > v62 )
              {
                v51 = v99;
LABEL_73:
                v100 = v51;
                v101 = v96;
                sub_15D6E30((__int64)v102, (__int64)&v122, &v100, &v101);
                v63 = (unsigned int)v130;
                if ( (unsigned int)v130 >= HIDWORD(v130) )
                {
                  sub_16CD150(&v129, v131, 0, 8);
                  v63 = (unsigned int)v130;
                }
                v129[v63] = v99;
                v64 = (unsigned int)v104;
                LODWORD(v130) = v130 + 1;
                if ( (unsigned int)v104 >= HIDWORD(v104) )
                {
                  sub_16CD150(&v103, v105, 0, 8);
                  v64 = (unsigned int)v104;
                }
                v103[v64] = v99;
                LODWORD(v104) = v104 + 1;
              }
LABEL_28:
              if ( v97 == ++v31 )
                goto LABEL_54;
            }
            else
            {
              if ( (unsigned int)v34 <= v95 + 1 )
                goto LABEL_28;
              if ( (v119 & 1) != 0 )
              {
                v35 = &v120;
                v36 = 7;
              }
              else
              {
                v35 = v120;
                if ( !v121 )
                  goto LABEL_45;
                v36 = v121 - 1;
              }
              v37 = v36 & (((unsigned int)v99 >> 9) ^ ((unsigned int)v99 >> 4));
              v38 = v35[v37];
              if ( v99 == v38 )
                goto LABEL_28;
              v39 = 1;
              while ( v38 != -8 )
              {
                v37 = v36 & (v39 + v37);
                v38 = v35[v37];
                if ( v99 == v38 )
                  goto LABEL_28;
                ++v39;
              }
LABEL_45:
              sub_15D6A20((__int64)v102, (__int64)&v118, &v99);
              v40 = v99;
              v41 = (unsigned int)v116;
              if ( (unsigned int)v116 >= HIDWORD(v116) )
              {
                sub_16CD150(&v115, v117, 0, 16);
                v41 = (unsigned int)v116;
              }
              v42 = &v115[16 * v41];
              *v42 = v34;
              v43 = v115;
              v42[1] = v40;
              LODWORD(v116) = v116 + 1;
              v44 = 16LL * (unsigned int)v116;
              v45 = *(_DWORD *)&v43[v44 - 16];
              v46 = *(_QWORD *)&v43[v44 - 8];
              v47 = (v44 >> 4) - 1;
              v48 = ((v44 >> 4) - 2) / 2;
              if ( v47 > 0 )
              {
                while ( 1 )
                {
                  v49 = &v43[16 * v47];
                  v50 = &v43[16 * v48];
                  if ( *(_DWORD *)v50 <= v45 )
                  {
                    *(_DWORD *)v49 = v45;
                    *((_QWORD *)v49 + 1) = v46;
                    goto LABEL_53;
                  }
                  *(_DWORD *)v49 = *(_DWORD *)v50;
                  *((_QWORD *)v49 + 1) = *((_QWORD *)v50 + 1);
                  v47 = v48;
                  if ( v48 <= 0 )
                    break;
                  v48 = (v48 - 1) / 2;
                }
                *(_DWORD *)v50 = v45;
                *((_QWORD *)v50 + 1) = v46;
              }
              else
              {
                v91 = (__int64)&v43[v44 - 16];
                *(_DWORD *)v91 = v45;
                *(_QWORD *)(v91 + 8) = v46;
              }
LABEL_53:
              if ( v97 == ++v31 )
              {
LABEL_54:
                v97 = v106;
                goto LABEL_55;
              }
            }
          }
          if ( v111 == v110 )
            v65 = &v111[HIDWORD(v112)];
          else
            v65 = &v111[(unsigned int)v112];
LABEL_35:
          while ( v65 != v33 && (unsigned __int64)*v33 >= 0xFFFFFFFFFFFFFFFELL )
            ++v33;
          goto LABEL_27;
        }
LABEL_55:
        if ( v97 != (__int64 *)&v108 )
          _libc_free((unsigned __int64)v97);
        sub_1412190((__int64)&v109, (__int64)v30);
        v29 = v104;
        if ( !(_DWORD)v104 )
          break;
        v27 = v103;
      }
      if ( v111 != v110 )
        _libc_free((unsigned __int64)v111);
      if ( v103 != v105 )
        _libc_free((unsigned __int64)v103);
      v21 = v116;
      if ( !(_DWORD)v116 )
        goto LABEL_97;
    }
    v73 = &v115[v25];
    v74 = *((_DWORD *)v73 - 4);
    v75 = *((_QWORD *)v73 - 1);
    v73 -= 16;
    *(_DWORD *)v73 = *(_DWORD *)v115;
    *((_QWORD *)v73 + 1) = *((_QWORD *)v22 + 1);
    v76 = v73 - v22;
    v77 = v76 >> 4;
    v78 = (v76 >> 4) - 1;
    v79 = (v76 >> 4) & 1;
    v80 = v78 / 2;
    if ( v76 <= 32 )
    {
      v89 = v22;
      if ( v79 || (unsigned __int64)v78 > 2 )
      {
LABEL_124:
        *(_DWORD *)v89 = v74;
        *((_QWORD *)v89 + 1) = v75;
        goto LABEL_19;
      }
      v82 = 0;
      v83 = v22;
    }
    else
    {
      for ( i = 0; ; i = v82 )
      {
        v82 = 2 * (i + 1);
        v83 = &v22[32 * i + 32];
        v84 = *(_DWORD *)v83;
        if ( *(_DWORD *)v83 > *((_DWORD *)v83 - 4) )
        {
          --v82;
          v83 = &v22[16 * v82];
          v84 = *(_DWORD *)v83;
        }
        v85 = &v22[16 * i];
        *(_DWORD *)v85 = v84;
        *((_QWORD *)v85 + 1) = *((_QWORD *)v83 + 1);
        if ( v82 >= v80 )
          break;
      }
      if ( v79 )
      {
LABEL_120:
        v86 = (v82 - 1) >> 1;
LABEL_123:
        while ( 1 )
        {
          v89 = &v22[16 * v82];
          v90 = &v22[16 * v86];
          if ( *(_DWORD *)v90 <= v74 )
            goto LABEL_124;
          *(_DWORD *)v89 = *(_DWORD *)v90;
          *((_QWORD *)v89 + 1) = *((_QWORD *)v90 + 1);
          v82 = v86;
          if ( !v86 )
          {
            *(_DWORD *)v90 = v74;
            *((_QWORD *)v90 + 1) = v75;
            goto LABEL_19;
          }
          v86 = (v86 - 1) / 2;
        }
      }
      v86 = (v82 - 1) >> 1;
      if ( v82 != (v77 - 2) / 2 )
        goto LABEL_123;
    }
    v87 = v82 + 1;
    v82 = 2 * (v82 + 1) - 1;
    v88 = (__int64)&v22[32 * v87 - 16];
    *(_DWORD *)v83 = *(_DWORD *)v88;
    *((_QWORD *)v83 + 1) = *(_QWORD *)(v88 + 8);
    goto LABEL_120;
  }
LABEL_97:
  v66 = v126;
  v67 = &v126[(unsigned int)v127];
  if ( v126 != v67 )
  {
    do
    {
      v68 = *v66++;
      sub_15CE4D0(v68, v93);
    }
    while ( v67 != v66 );
  }
  v69 = v129;
  v70 = &v129[(unsigned int)v130];
  if ( v129 != v70 )
  {
    do
    {
      v71 = *v69++;
      sub_15CC3F0(v71);
    }
    while ( v70 != v69 );
    v70 = v129;
  }
  if ( v70 != (__int64 *)v131 )
    _libc_free((unsigned __int64)v70);
  if ( v126 != (__int64 *)v128 )
    _libc_free((unsigned __int64)v126);
  if ( (v123 & 1) != 0 )
  {
    if ( (v119 & 1) != 0 )
      goto LABEL_108;
  }
  else
  {
    j___libc_free_0(v124);
    if ( (v119 & 1) != 0 )
    {
LABEL_108:
      v72 = (unsigned __int64)v115;
      if ( v115 == v117 )
        return;
      goto LABEL_109;
    }
  }
  j___libc_free_0(v120);
  v72 = (unsigned __int64)v115;
  if ( v115 != v117 )
LABEL_109:
    _libc_free(v72);
}
