// Function: sub_15D8000
// Address: 0x15d8000
//
void __fastcall sub_15D8000(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4)
{
  __int64 *v4; // r8
  _QWORD *v7; // r12
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 *v10; // rax
  unsigned __int64 *v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // r12
  _QWORD *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  unsigned int v18; // r8d
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rax
  unsigned int v23; // eax
  __int64 v24; // rdx
  __int64 v25; // rbx
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rax
  _QWORD *v29; // rdx
  int v30; // eax
  unsigned int v31; // eax
  __int64 *v32; // r12
  __int64 *v33; // rbx
  __int64 **v34; // r13
  __int64 **v35; // rax
  __int64 v36; // r14
  __int64 v37; // rsi
  __int64 *v38; // rbx
  __int64 *v39; // r12
  __int64 v40; // rdi
  __int64 *v41; // rbx
  __int64 *v42; // r12
  __int64 v43; // rdi
  unsigned __int64 v44; // rdi
  _QWORD *v45; // r8
  int v46; // edx
  unsigned int v47; // eax
  __int64 v48; // rdi
  int v49; // r9d
  __int64 v50; // r13
  __int64 v51; // rax
  _QWORD *v52; // rax
  __int64 v53; // rsi
  __int64 v54; // rax
  unsigned int v55; // edi
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 v58; // rcx
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rcx
  _QWORD *v62; // rdi
  int v63; // esi
  int v64; // r8d
  unsigned int v65; // eax
  __int64 v66; // rdx
  char v67; // al
  __int64 v68; // rcx
  unsigned int v69; // eax
  unsigned int v70; // esi
  __int64 v71; // rax
  unsigned int v72; // eax
  __int64 v73; // rax
  __int64 v74; // rax
  __int64 **v75; // rsi
  _BYTE *v76; // rax
  unsigned int v77; // r10d
  __int64 v78; // r13
  __int64 v79; // rax
  __int64 v80; // r9
  __int64 v81; // rcx
  __int64 v82; // r12
  __int64 v83; // r11
  __int64 i; // rcx
  __int64 v85; // rax
  __int64 v86; // rdi
  int v87; // r8d
  __int64 v88; // rcx
  __int64 v89; // rcx
  __int64 v90; // rax
  __int64 v91; // rdi
  __int64 v92; // rcx
  __int64 v93; // rcx
  __int64 v94; // rax
  _QWORD *v95; // rdi
  _QWORD *v96; // rsi
  __int64 v97; // [rsp+30h] [rbp-420h]
  int v99; // [rsp+88h] [rbp-3C8h]
  unsigned int v100; // [rsp+8Ch] [rbp-3C4h]
  __int64 *v101; // [rsp+90h] [rbp-3C0h]
  __int64 *v102; // [rsp+98h] [rbp-3B8h] BYREF
  __int64 v103; // [rsp+A8h] [rbp-3A8h] BYREF
  __int64 v104; // [rsp+B0h] [rbp-3A0h] BYREF
  unsigned int v105; // [rsp+B8h] [rbp-398h] BYREF
  _QWORD v106[6]; // [rsp+C0h] [rbp-390h] BYREF
  _QWORD *v107; // [rsp+F0h] [rbp-360h] BYREF
  __int64 v108; // [rsp+F8h] [rbp-358h]
  _QWORD v109[8]; // [rsp+100h] [rbp-350h] BYREF
  __int64 *v110; // [rsp+140h] [rbp-310h] BYREF
  unsigned int v111; // [rsp+148h] [rbp-308h] BYREF
  char v112; // [rsp+150h] [rbp-300h] BYREF
  __int64 v113; // [rsp+190h] [rbp-2C0h] BYREF
  __int64 **v114; // [rsp+198h] [rbp-2B8h]
  __int64 **v115; // [rsp+1A0h] [rbp-2B0h]
  __int64 v116; // [rsp+1A8h] [rbp-2A8h]
  int v117; // [rsp+1B0h] [rbp-2A0h]
  _BYTE v118[72]; // [rsp+1B8h] [rbp-298h] BYREF
  _BYTE *v119; // [rsp+200h] [rbp-250h] BYREF
  __int64 v120; // [rsp+208h] [rbp-248h]
  _BYTE v121[136]; // [rsp+210h] [rbp-240h] BYREF
  __int64 v122; // [rsp+298h] [rbp-1B8h] BYREF
  __int64 v123; // [rsp+2A0h] [rbp-1B0h]
  _QWORD *v124; // [rsp+2A8h] [rbp-1A8h] BYREF
  int v125; // [rsp+2B0h] [rbp-1A0h]
  __int64 v126; // [rsp+2E8h] [rbp-168h] BYREF
  __int64 v127; // [rsp+2F0h] [rbp-160h]
  _QWORD *v128; // [rsp+2F8h] [rbp-158h] BYREF
  unsigned int v129; // [rsp+300h] [rbp-150h]
  __int64 *v130; // [rsp+378h] [rbp-D8h] BYREF
  __int64 v131; // [rsp+380h] [rbp-D0h]
  _BYTE v132[64]; // [rsp+388h] [rbp-C8h] BYREF
  __int64 *v133; // [rsp+3C8h] [rbp-88h] BYREF
  __int64 v134; // [rsp+3D0h] [rbp-80h]
  _BYTE v135[120]; // [rsp+3D8h] [rbp-78h] BYREF

  v4 = a3;
  v7 = (_QWORD *)a4[1];
  v102 = a4;
  if ( !*v7 )
  {
    v95 = *(_QWORD **)a1;
    v119 = (_BYTE *)*a4;
    v96 = &v95[*(unsigned int *)(a1 + 8)];
    if ( v96 != sub_15CBCA0(v95, (__int64)v96, (__int64 *)&v119) )
    {
      sub_15D5DA0(a1, a2);
      return;
    }
  }
  v8 = *v4;
  if ( *v4 )
  {
    if ( *a4 )
      v8 = sub_15CC9E0(a1, v8, *a4);
    else
      v8 = 0;
  }
  v9 = sub_15CC960(a1, v8);
  v97 = v9;
  if ( (__int64 *)v9 == a4 || (_QWORD *)v9 == v7 )
    return;
  v122 = 0;
  v119 = v121;
  v120 = 0x800000000LL;
  v10 = (__int64 *)&v124;
  v123 = 1;
  do
    *v10++ = -8;
  while ( v10 != &v126 );
  v11 = (unsigned __int64 *)&v128;
  v126 = 0;
  v127 = 1;
  do
  {
    *v11 = -8;
    v11 += 2;
  }
  while ( v11 != (unsigned __int64 *)&v130 );
  v130 = (__int64 *)v132;
  v131 = 0x800000000LL;
  v134 = 0x800000000LL;
  v133 = (__int64 *)v135;
  sub_15D6A20((__int64)&v113, (__int64)&v122, (__int64 *)&v102);
  v12 = (__int64)v102;
  v13 = (unsigned int)v120;
  v14 = *((unsigned int *)v102 + 4);
  if ( (unsigned int)v120 >= HIDWORD(v120) )
  {
    sub_16CD150(&v119, v121, 0, 16);
    v13 = (unsigned int)v120;
  }
  v15 = &v119[16 * v13];
  *v15 = v14;
  v15[1] = v12;
  v16 = (__int64)v119;
  LODWORD(v120) = v120 + 1;
  v17 = 16LL * (unsigned int)v120;
  v18 = *(_DWORD *)&v119[v17 - 16];
  v19 = *(_QWORD *)&v119[v17 - 8];
  v20 = (v17 >> 4) - 1;
  v21 = ((v17 >> 4) - 2) / 2;
  if ( v20 > 0 )
  {
    while ( 1 )
    {
      v37 = v16 + 16 * v21;
      v22 = v16 + 16 * v20;
      if ( *(_DWORD *)v37 <= v18 )
        break;
      *(_DWORD *)v22 = *(_DWORD *)v37;
      *(_QWORD *)(v22 + 8) = *(_QWORD *)(v37 + 8);
      v20 = v21;
      if ( v21 <= 0 )
      {
        v22 = v16 + 16 * v21;
        break;
      }
      v21 = (v21 - 1) / 2;
    }
  }
  else
  {
    v22 = (__int64)&v119[v17 - 16];
  }
  *(_DWORD *)v22 = v18;
  *(_QWORD *)(v22 + 8) = v19;
  v23 = v120;
  if ( (_DWORD)v120 )
  {
    while ( 1 )
    {
      v24 = (__int64)v119;
      v25 = *((_QWORD *)v119 + 1);
      v100 = *(_DWORD *)(v25 + 16);
      v26 = v23;
      v27 = 16LL * v23;
      if ( v26 != 1 )
        break;
LABEL_17:
      v110 = (__int64 *)v25;
      v111 = v100;
      LODWORD(v120) = v120 - 1;
      sub_15D6E30((__int64)&v113, (__int64)&v126, (__int64 *)&v110, &v111);
      v28 = (unsigned int)v131;
      if ( (unsigned int)v131 >= HIDWORD(v131) )
      {
        sub_16CD150(&v130, v132, 0, 8);
        v28 = (unsigned int)v131;
      }
      v130[v28] = v25;
      v29 = v109;
      LODWORD(v131) = v131 + 1;
      v30 = *(_DWORD *)(v97 + 16);
      v109[0] = v25;
      v113 = 0;
      v99 = v30;
      v116 = 8;
      v107 = v109;
      v108 = 0x800000001LL;
      v117 = 0;
      v114 = (__int64 **)v118;
      v115 = (__int64 **)v118;
      v31 = 1;
      while ( 1 )
      {
        v32 = (__int64 *)v29[v31 - 1];
        LODWORD(v108) = v31 - 1;
        sub_15CF0D0((__int64)&v110, *v32, a2);
        v33 = v110;
        v101 = &v110[v111];
        if ( v110 != v101 )
        {
          while ( 1 )
          {
            v103 = sub_15CC960(a1, *v33);
            v36 = *(unsigned int *)(v103 + 16);
            v35 = v114;
            if ( v115 == v114 )
            {
              v34 = &v114[HIDWORD(v116)];
              if ( v114 == v34 )
              {
                v75 = v114;
              }
              else
              {
                do
                {
                  if ( v32 == *v35 )
                    break;
                  ++v35;
                }
                while ( v34 != v35 );
                v75 = &v114[HIDWORD(v116)];
              }
              goto LABEL_33;
            }
            v34 = &v115[(unsigned int)v116];
            v35 = (__int64 **)sub_16CC9F0(&v113, v32);
            if ( v32 == *v35 )
              break;
            if ( v115 == v114 )
            {
              v35 = &v115[HIDWORD(v116)];
              v75 = v35;
              goto LABEL_33;
            }
            v35 = &v115[(unsigned int)v116];
LABEL_25:
            if ( v34 != v35 )
              goto LABEL_26;
            if ( v100 < (unsigned int)v36 )
            {
              v61 = v103;
              if ( (v127 & 1) != 0 )
              {
                v62 = &v128;
                v63 = 7;
              }
              else
              {
                v62 = v128;
                if ( !v129 )
                  goto LABEL_98;
                v63 = v129 - 1;
              }
              v64 = 1;
              v65 = v63 & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
              v66 = v62[2 * v65];
              if ( v103 != v66 )
              {
                while ( v66 != -8 )
                {
                  v65 = v63 & (v64 + v65);
                  v66 = v62[2 * v65];
                  if ( v103 == v66 )
                    goto LABEL_89;
                  ++v64;
                }
                goto LABEL_98;
              }
LABEL_89:
              v67 = sub_15D0950((__int64)&v126, &v103, v106);
              v68 = v106[0];
              if ( v67 )
              {
                v72 = *(_DWORD *)(v106[0] + 8LL);
                goto LABEL_96;
              }
              ++v126;
              v69 = ((unsigned int)v127 >> 1) + 1;
              if ( (v127 & 1) != 0 )
              {
                v70 = 8;
                if ( 4 * v69 >= 0x18 )
                {
LABEL_115:
                  v70 *= 2;
LABEL_116:
                  sub_15D6B70((__int64)&v126, v70);
                  sub_15D0950((__int64)&v126, &v103, v106);
                  v68 = v106[0];
                  v69 = ((unsigned int)v127 >> 1) + 1;
                  goto LABEL_93;
                }
              }
              else
              {
                v70 = v129;
                if ( 4 * v69 >= 3 * v129 )
                  goto LABEL_115;
              }
              if ( v70 - (v69 + HIDWORD(v127)) <= v70 >> 3 )
                goto LABEL_116;
LABEL_93:
              LODWORD(v127) = v127 & 1 | (2 * v69);
              if ( *(_QWORD *)v68 != -8 )
                --HIDWORD(v127);
              v71 = v103;
              *(_DWORD *)(v68 + 8) = 0;
              *(_QWORD *)v68 = v71;
              v72 = 0;
LABEL_96:
              if ( v100 > v72 )
              {
                v61 = v103;
LABEL_98:
                v104 = v61;
                v105 = v100;
                sub_15D6E30((__int64)v106, (__int64)&v126, &v104, &v105);
                v73 = (unsigned int)v134;
                if ( (unsigned int)v134 >= HIDWORD(v134) )
                {
                  sub_16CD150(&v133, v135, 0, 8);
                  v73 = (unsigned int)v134;
                }
                v133[v73] = v103;
                v74 = (unsigned int)v108;
                LODWORD(v134) = v134 + 1;
                if ( (unsigned int)v108 >= HIDWORD(v108) )
                {
                  sub_16CD150(&v107, v109, 0, 8);
                  v74 = (unsigned int)v108;
                }
                v107[v74] = v103;
                LODWORD(v108) = v108 + 1;
              }
LABEL_26:
              if ( v101 == ++v33 )
                goto LABEL_79;
            }
            else
            {
              if ( (unsigned int)v36 <= v99 + 1 )
                goto LABEL_26;
              if ( (v123 & 1) != 0 )
              {
                v45 = &v124;
                v46 = 7;
              }
              else
              {
                v45 = v124;
                if ( !v125 )
                  goto LABEL_70;
                v46 = v125 - 1;
              }
              v47 = v46 & (((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4));
              v48 = v45[v47];
              if ( v103 == v48 )
                goto LABEL_26;
              v49 = 1;
              while ( v48 != -8 )
              {
                v47 = v46 & (v49 + v47);
                v48 = v45[v47];
                if ( v103 == v48 )
                  goto LABEL_26;
                ++v49;
              }
LABEL_70:
              sub_15D6A20((__int64)v106, (__int64)&v122, &v103);
              v50 = v103;
              v51 = (unsigned int)v120;
              if ( (unsigned int)v120 >= HIDWORD(v120) )
              {
                sub_16CD150(&v119, v121, 0, 16);
                v51 = (unsigned int)v120;
              }
              v52 = &v119[16 * v51];
              *v52 = v36;
              v52[1] = v50;
              v53 = (__int64)v119;
              LODWORD(v120) = v120 + 1;
              v54 = 16LL * (unsigned int)v120;
              v55 = *(_DWORD *)&v119[v54 - 16];
              v56 = *(_QWORD *)&v119[v54 - 8];
              v57 = (v54 >> 4) - 1;
              v58 = ((v54 >> 4) - 2) / 2;
              if ( v57 > 0 )
              {
                while ( 1 )
                {
                  v59 = v53 + 16 * v57;
                  v60 = v53 + 16 * v58;
                  if ( *(_DWORD *)v60 <= v55 )
                  {
                    *(_DWORD *)v59 = v55;
                    *(_QWORD *)(v59 + 8) = v56;
                    goto LABEL_78;
                  }
                  *(_DWORD *)v59 = *(_DWORD *)v60;
                  *(_QWORD *)(v59 + 8) = *(_QWORD *)(v60 + 8);
                  v57 = v58;
                  if ( v58 <= 0 )
                    break;
                  v58 = (v58 - 1) / 2;
                }
                *(_DWORD *)v60 = v55;
                *(_QWORD *)(v60 + 8) = v56;
              }
              else
              {
                v94 = (__int64)&v119[v54 - 16];
                *(_DWORD *)v94 = v55;
                *(_QWORD *)(v94 + 8) = v56;
              }
LABEL_78:
              if ( v101 == ++v33 )
              {
LABEL_79:
                v101 = v110;
                goto LABEL_80;
              }
            }
          }
          if ( v115 == v114 )
            v75 = &v115[HIDWORD(v116)];
          else
            v75 = &v115[(unsigned int)v116];
LABEL_33:
          while ( v75 != v35 && (unsigned __int64)*v35 >= 0xFFFFFFFFFFFFFFFELL )
            ++v35;
          goto LABEL_25;
        }
LABEL_80:
        if ( v101 != (__int64 *)&v112 )
          _libc_free((unsigned __int64)v101);
        sub_1412190((__int64)&v113, (__int64)v32);
        v31 = v108;
        if ( !(_DWORD)v108 )
          break;
        v29 = v107;
      }
      if ( v115 != v114 )
        _libc_free((unsigned __int64)v115);
      if ( v107 != v109 )
        _libc_free((unsigned __int64)v107);
      v23 = v120;
      if ( !(_DWORD)v120 )
        goto LABEL_45;
    }
    v76 = &v119[v27];
    v77 = *((_DWORD *)v76 - 4);
    v78 = *((_QWORD *)v76 - 1);
    v76 -= 16;
    *(_DWORD *)v76 = *(_DWORD *)v119;
    *((_QWORD *)v76 + 1) = *(_QWORD *)(v24 + 8);
    v79 = (__int64)&v76[-v24];
    v80 = v79 >> 4;
    v81 = (v79 >> 4) - 1;
    v82 = (v79 >> 4) & 1;
    v83 = v81 / 2;
    if ( v79 <= 32 )
    {
      v90 = v24;
      if ( v82 || (unsigned __int64)v81 > 2 )
      {
LABEL_129:
        *(_DWORD *)v90 = v77;
        *(_QWORD *)(v90 + 8) = v78;
        goto LABEL_17;
      }
      v86 = v24;
      v85 = 0;
    }
    else
    {
      for ( i = 0; ; i = v85 )
      {
        v85 = 2 * (i + 1);
        v86 = v24 + 32 * (i + 1);
        v87 = *(_DWORD *)v86;
        if ( *(_DWORD *)v86 > *(_DWORD *)(v86 - 16) )
        {
          --v85;
          v86 = v24 + 16 * v85;
          v87 = *(_DWORD *)v86;
        }
        v88 = v24 + 16 * i;
        *(_DWORD *)v88 = v87;
        *(_QWORD *)(v88 + 8) = *(_QWORD *)(v86 + 8);
        if ( v85 >= v83 )
          break;
      }
      if ( v82 )
      {
LABEL_134:
        v89 = (v85 - 1) >> 1;
LABEL_128:
        while ( 1 )
        {
          v90 = v24 + 16 * v85;
          v91 = v24 + 16 * v89;
          if ( *(_DWORD *)v91 <= v77 )
            goto LABEL_129;
          *(_DWORD *)v90 = *(_DWORD *)v91;
          *(_QWORD *)(v90 + 8) = *(_QWORD *)(v91 + 8);
          v85 = v89;
          if ( !v89 )
          {
            *(_DWORD *)v91 = v77;
            *(_QWORD *)(v91 + 8) = v78;
            goto LABEL_17;
          }
          v89 = (v89 - 1) / 2;
        }
      }
      v89 = (v85 - 1) >> 1;
      if ( v85 != (v80 - 2) / 2 )
        goto LABEL_128;
    }
    v92 = v85 + 1;
    v85 = 2 * (v85 + 1) - 1;
    v93 = v24 + 32 * v92 - 16;
    *(_DWORD *)v86 = *(_DWORD *)v93;
    *(_QWORD *)(v86 + 8) = *(_QWORD *)(v93 + 8);
    goto LABEL_134;
  }
LABEL_45:
  v38 = v130;
  v39 = &v130[(unsigned int)v131];
  if ( v130 != v39 )
  {
    do
    {
      v40 = *v38++;
      sub_15CE4D0(v40, v97);
    }
    while ( v39 != v38 );
  }
  v41 = v133;
  v42 = &v133[(unsigned int)v134];
  if ( v133 != v42 )
  {
    do
    {
      v43 = *v41++;
      sub_15CC3F0(v43);
    }
    while ( v42 != v41 );
  }
  sub_15D6090(a1, a2);
  if ( v133 != (__int64 *)v135 )
    _libc_free((unsigned __int64)v133);
  if ( v130 != (__int64 *)v132 )
    _libc_free((unsigned __int64)v130);
  if ( (v127 & 1) != 0 )
  {
    if ( (v123 & 1) != 0 )
      goto LABEL_55;
  }
  else
  {
    j___libc_free_0(v128);
    if ( (v123 & 1) != 0 )
    {
LABEL_55:
      v44 = (unsigned __int64)v119;
      if ( v119 == v121 )
        return;
      goto LABEL_56;
    }
  }
  j___libc_free_0(v124);
  v44 = (unsigned __int64)v119;
  if ( v119 != v121 )
LABEL_56:
    _libc_free(v44);
}
