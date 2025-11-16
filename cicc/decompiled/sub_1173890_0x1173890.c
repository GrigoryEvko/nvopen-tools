// Function: sub_1173890
// Address: 0x1173890
//
unsigned __int8 *__fastcall sub_1173890(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned __int8 **v3; // rdi
  unsigned __int8 *v4; // r15
  __int64 *v5; // rax
  __int64 v6; // r14
  __int64 v7; // r12
  __int64 v8; // rsi
  __int64 v9; // rsi
  __int64 v10; // rax
  const char *v11; // r9
  __int64 *v12; // rdx
  __int64 *v13; // rcx
  __int64 v14; // rax
  const char *v15; // r12
  __int64 v16; // r14
  __int64 *v17; // rbx
  _BYTE *v18; // r15
  unsigned __int8 v19; // al
  const char **v20; // rdi
  const char *v21; // rdi
  __int64 v22; // rax
  const char *v23; // r13
  unsigned __int8 *v24; // r13
  __int64 *v25; // rax
  __int64 v26; // r13
  __int64 v28; // rdi
  __int64 v29; // rax
  unsigned __int64 *v30; // rdx
  unsigned __int64 *v31; // r12
  unsigned __int64 *i; // r14
  unsigned __int64 v33; // rsi
  __int64 *v34; // rdx
  int v35; // eax
  __int64 *v36; // rdx
  __int64 v37; // rax
  __int64 v38; // r8
  int v39; // eax
  int v40; // eax
  unsigned int v41; // edx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rdx
  __int64 v45; // r13
  __int64 v46; // r8
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rsi
  __int64 v50; // rsi
  __int64 v51; // rdi
  __int64 v52; // rdx
  const char *v53; // rsi
  __int64 v54; // r8
  const char *v55; // r14
  __int64 v56; // r12
  __int64 *v57; // r15
  __int64 v58; // rcx
  __int64 v59; // rbx
  __int64 *v60; // rdx
  __int64 v61; // rdx
  int v62; // eax
  int v63; // eax
  unsigned int v64; // edi
  __int64 v65; // rax
  __int64 v66; // rdi
  __int64 v67; // rdi
  __int64 v68; // rbx
  __int64 v69; // rbx
  int v70; // eax
  int v71; // eax
  unsigned int v72; // edx
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rdx
  __int64 v76; // rdi
  int v77; // eax
  __int64 *v78; // rdx
  __int64 v79; // rax
  __int64 v80; // r8
  int v81; // eax
  int v82; // eax
  unsigned int v83; // edx
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rdx
  __int64 v87; // [rsp+10h] [rbp-D0h]
  __int64 v88; // [rsp+18h] [rbp-C8h]
  __int64 v89; // [rsp+18h] [rbp-C8h]
  __int64 v90; // [rsp+18h] [rbp-C8h]
  __int64 v91; // [rsp+20h] [rbp-C0h]
  __int64 v93; // [rsp+30h] [rbp-B0h]
  unsigned __int8 *v94; // [rsp+38h] [rbp-A8h]
  int v95; // [rsp+38h] [rbp-A8h]
  __int64 v96; // [rsp+38h] [rbp-A8h]
  unsigned __int8 *v97; // [rsp+38h] [rbp-A8h]
  __int64 v98; // [rsp+38h] [rbp-A8h]
  __int64 *v99; // [rsp+40h] [rbp-A0h]
  __int64 v100; // [rsp+40h] [rbp-A0h]
  int v101; // [rsp+40h] [rbp-A0h]
  __int64 v102; // [rsp+48h] [rbp-98h]
  __int64 v103; // [rsp+48h] [rbp-98h]
  __int64 v104; // [rsp+48h] [rbp-98h]
  __int64 v105; // [rsp+48h] [rbp-98h]
  __int64 v106; // [rsp+50h] [rbp-90h]
  unsigned __int8 v107; // [rsp+58h] [rbp-88h]
  __int64 v108; // [rsp+58h] [rbp-88h]
  __int64 *v109; // [rsp+58h] [rbp-88h]
  __int64 v110[4]; // [rsp+60h] [rbp-80h] BYREF
  const char *v111; // [rsp+80h] [rbp-60h] BYREF
  __int64 *v112; // [rsp+88h] [rbp-58h]
  const char *v113; // [rsp+90h] [rbp-50h]
  __int64 *v114; // [rsp+98h] [rbp-48h]
  __int16 v115; // [rsp+A0h] [rbp-40h]

  v2 = a2;
  v3 = *(unsigned __int8 ***)(a2 - 8);
  v4 = *v3;
  v107 = **v3;
  if ( ((*v3)[7] & 0x40) != 0 )
    v5 = (__int64 *)*((_QWORD *)v4 - 1);
  else
    v5 = (__int64 *)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
  v6 = *v5;
  v7 = v5[4];
  v91 = a2;
  v106 = *(_QWORD *)(*v5 + 8);
  v8 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v102 = *(_QWORD *)(v7 + 8);
  if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
  {
    v9 = (__int64)&v3[v8];
  }
  else
  {
    v3 = (unsigned __int8 **)(v2 - v8 * 8);
    v9 = v2;
  }
  v10 = sub_116D080((__int64)v3, v9, 1);
  v99 = v12;
  v13 = (__int64 *)v10;
  if ( (__int64 *)v10 != v12 )
  {
    v94 = v4;
    v14 = v7;
    v15 = (const char *)v6;
    v88 = v2;
    v16 = v14;
    v17 = v13;
    do
    {
      v18 = (_BYTE *)*v17;
      v19 = *(_BYTE *)*v17;
      if ( v19 != v107 || v19 <= 0x1Cu || !(unsigned __int8)sub_BD36B0(*v17) )
        return 0;
      v20 = (v18[7] & 0x40) != 0
          ? (const char **)*((_QWORD *)v18 - 1)
          : (const char **)&v18[-32 * (*((_DWORD *)v18 + 1) & 0x7FFFFFF)];
      v11 = *v20;
      if ( *((_QWORD *)*v20 + 1) != v106 )
        return 0;
      v21 = v20[4];
      if ( *((_QWORD *)v21 + 1) != v102
        || (unsigned __int8)(*v18 - 82) <= 1u && (*((_WORD *)v94 + 1) & 0x3F) != (*((_WORD *)v18 + 1) & 0x3F) )
      {
        return 0;
      }
      if ( v11 != v15 )
        v15 = 0;
      if ( v21 != (const char *)v16 )
        v16 = 0;
      v17 += 4;
    }
    while ( v99 != v17 );
    v22 = v16;
    v6 = (__int64)v15;
    v4 = v94;
    v2 = v88;
    v23 = v15;
    v7 = v22;
    v24 = (unsigned __int8 *)(v22 | (unsigned __int64)v23);
    if ( !v24 )
      return v24;
    if ( (v94[7] & 0x40) != 0 )
    {
      v25 = (__int64 *)*((_QWORD *)v94 - 1);
      v26 = *v25;
      v108 = v25[4];
      if ( v6 )
      {
LABEL_24:
        if ( v7 )
        {
          v107 = *v94;
          goto LABEL_26;
        }
        v76 = v108;
        v96 = v88 + 24;
        goto LABEL_88;
      }
    }
    else
    {
      v34 = (__int64 *)&v94[-32 * (*((_DWORD *)v94 + 1) & 0x7FFFFFF)];
      v26 = *v34;
      v108 = v34[4];
      if ( v6 )
        goto LABEL_24;
    }
    v111 = sub_BD5D20(v26);
    v113 = ".pn";
    v35 = *(_DWORD *)(v88 + 4);
    v115 = 773;
    v112 = v36;
    v95 = v35 & 0x7FFFFFF;
    v37 = sub_BD2DA0(80);
    v6 = v37;
    if ( v37 )
    {
      sub_B44260(v37, v106, 55, 0x8000000u, 0, 0);
      *(_DWORD *)(v6 + 72) = v95;
      sub_BD6B50((unsigned __int8 *)v6, &v111);
      sub_BD2A10(v6, *(_DWORD *)(v6 + 72), 1);
    }
    v38 = *(_QWORD *)(*(_QWORD *)(v88 - 8) + 32LL * *(unsigned int *)(v88 + 72));
    v39 = *(_DWORD *)(v6 + 4) & 0x7FFFFFF;
    if ( v39 == *(_DWORD *)(v6 + 72) )
    {
      v98 = *(_QWORD *)(*(_QWORD *)(v88 - 8) + 32LL * *(unsigned int *)(v88 + 72));
      sub_B48D90(v6);
      v38 = v98;
      v39 = *(_DWORD *)(v6 + 4) & 0x7FFFFFF;
    }
    v40 = (v39 + 1) & 0x7FFFFFF;
    v41 = v40 | *(_DWORD *)(v6 + 4) & 0xF8000000;
    v42 = *(_QWORD *)(v6 - 8) + 32LL * (unsigned int)(v40 - 1);
    *(_DWORD *)(v6 + 4) = v41;
    if ( *(_QWORD *)v42 )
    {
      v43 = *(_QWORD *)(v42 + 8);
      **(_QWORD **)(v42 + 16) = v43;
      if ( v43 )
        *(_QWORD *)(v43 + 16) = *(_QWORD *)(v42 + 16);
    }
    *(_QWORD *)v42 = v26;
    if ( v26 )
    {
      v44 = *(_QWORD *)(v26 + 16);
      *(_QWORD *)(v42 + 8) = v44;
      if ( v44 )
        *(_QWORD *)(v44 + 16) = v42 + 8;
      *(_QWORD *)(v42 + 16) = v26 + 16;
      *(_QWORD *)(v26 + 16) = v42;
    }
    v45 = v6;
    *(_QWORD *)(*(_QWORD *)(v6 - 8) + 32LL * *(unsigned int *)(v6 + 72) + 8LL * ((*(_DWORD *)(v6 + 4) & 0x7FFFFFFu) - 1)) = v38;
    v96 = v88 + 24;
    sub_B44220((_QWORD *)v6, v88 + 24, 0);
    v111 = (const char *)v6;
    sub_11715E0(*(_QWORD *)(a1 + 40) + 2096LL, (__int64 *)&v111);
    v46 = 0;
    if ( v7 )
      goto LABEL_51;
    v7 = v6;
    if ( (v4[7] & 0x40) != 0 )
      v76 = *(_QWORD *)(*((_QWORD *)v4 - 1) + 32LL);
    else
      v76 = *(_QWORD *)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF) + 32];
LABEL_88:
    v45 = v7;
    v111 = sub_BD5D20(v76);
    v115 = 773;
    v113 = ".pn";
    v77 = *(_DWORD *)(v88 + 4);
    v112 = v78;
    v101 = v77 & 0x7FFFFFF;
    v79 = sub_BD2DA0(80);
    v7 = v79;
    if ( v79 )
    {
      sub_B44260(v79, v102, 55, 0x8000000u, 0, 0);
      *(_DWORD *)(v7 + 72) = v101;
      sub_BD6B50((unsigned __int8 *)v7, &v111);
      sub_BD2A10(v7, *(_DWORD *)(v7 + 72), 1);
    }
    v80 = *(_QWORD *)(*(_QWORD *)(v88 - 8) + 32LL * *(unsigned int *)(v88 + 72));
    v81 = *(_DWORD *)(v7 + 4) & 0x7FFFFFF;
    if ( v81 == *(_DWORD *)(v7 + 72) )
    {
      v105 = *(_QWORD *)(*(_QWORD *)(v88 - 8) + 32LL * *(unsigned int *)(v88 + 72));
      sub_B48D90(v7);
      v80 = v105;
      v81 = *(_DWORD *)(v7 + 4) & 0x7FFFFFF;
    }
    v82 = (v81 + 1) & 0x7FFFFFF;
    v83 = v82 | *(_DWORD *)(v7 + 4) & 0xF8000000;
    v84 = *(_QWORD *)(v7 - 8) + 32LL * (unsigned int)(v82 - 1);
    *(_DWORD *)(v7 + 4) = v83;
    if ( *(_QWORD *)v84 )
    {
      v85 = *(_QWORD *)(v84 + 8);
      **(_QWORD **)(v84 + 16) = v85;
      if ( v85 )
        *(_QWORD *)(v85 + 16) = *(_QWORD *)(v84 + 16);
    }
    *(_QWORD *)v84 = v108;
    if ( v108 )
    {
      v86 = *(_QWORD *)(v108 + 16);
      *(_QWORD *)(v84 + 8) = v86;
      if ( v86 )
        *(_QWORD *)(v86 + 16) = v84 + 8;
      *(_QWORD *)(v84 + 16) = v108 + 16;
      *(_QWORD *)(v108 + 16) = v84;
    }
    *(_QWORD *)(*(_QWORD *)(v7 - 8) + 32LL * *(unsigned int *)(v7 + 72) + 8LL * ((*(_DWORD *)(v7 + 4) & 0x7FFFFFFu) - 1)) = v80;
    sub_B44220((_QWORD *)v7, v96, 0);
    v111 = (const char *)v7;
    sub_11715E0(*(_QWORD *)(a1 + 40) + 2096LL, (__int64 *)&v111);
    v46 = v7;
LABEL_51:
    v47 = *(_QWORD *)(v88 - 8);
    v48 = *(_DWORD *)(v88 + 4) & 0x7FFFFFF;
    v49 = 32 * v48;
    if ( (*(_BYTE *)(v88 + 7) & 0x40) != 0 )
    {
      v50 = v47 + v49;
      v51 = *(_QWORD *)(v88 - 8);
    }
    else
    {
      v51 = v88 - v49;
      v50 = v88;
    }
    v52 = *(unsigned int *)(v88 + 72);
    v110[0] = v51;
    v110[1] = v50;
    v52 *= 32;
    v103 = v46;
    v110[2] = v47 + v52;
    v110[3] = v52 + 8 * v48 + v47;
    sub_116D910(&v111, v110, 1);
    v11 = v111;
    v53 = v113;
    v54 = v103;
    v109 = v114;
    if ( v111 != v113 && v112 != v114 )
    {
      v93 = v88;
      v104 = v6;
      v55 = v111;
      v100 = v7;
      v56 = v54;
      v97 = v4;
      v57 = v112;
      do
      {
        v58 = *v57;
        v59 = *(_QWORD *)v55;
        if ( v45 )
        {
          if ( (*(_BYTE *)(v59 + 7) & 0x40) != 0 )
            v60 = *(__int64 **)(v59 - 8);
          else
            v60 = (__int64 *)(v59 - 32LL * (*(_DWORD *)(v59 + 4) & 0x7FFFFFF));
          v61 = *v60;
          v62 = *(_DWORD *)(v45 + 4) & 0x7FFFFFF;
          if ( v62 == *(_DWORD *)(v45 + 72) )
          {
            v87 = v61;
            v90 = *v57;
            sub_B48D90(v45);
            v61 = v87;
            v58 = v90;
            v62 = *(_DWORD *)(v45 + 4) & 0x7FFFFFF;
          }
          v63 = (v62 + 1) & 0x7FFFFFF;
          v64 = v63 | *(_DWORD *)(v45 + 4) & 0xF8000000;
          v65 = *(_QWORD *)(v45 - 8) + 32LL * (unsigned int)(v63 - 1);
          *(_DWORD *)(v45 + 4) = v64;
          if ( *(_QWORD *)v65 )
          {
            v66 = *(_QWORD *)(v65 + 8);
            **(_QWORD **)(v65 + 16) = v66;
            if ( v66 )
              *(_QWORD *)(v66 + 16) = *(_QWORD *)(v65 + 16);
          }
          *(_QWORD *)v65 = v61;
          if ( v61 )
          {
            v67 = *(_QWORD *)(v61 + 16);
            *(_QWORD *)(v65 + 8) = v67;
            if ( v67 )
              *(_QWORD *)(v67 + 16) = v65 + 8;
            *(_QWORD *)(v65 + 16) = v61 + 16;
            *(_QWORD *)(v61 + 16) = v65;
          }
          *(_QWORD *)(*(_QWORD *)(v45 - 8)
                    + 32LL * *(unsigned int *)(v45 + 72)
                    + 8LL * ((*(_DWORD *)(v45 + 4) & 0x7FFFFFFu) - 1)) = v58;
        }
        if ( v56 )
        {
          if ( (*(_BYTE *)(v59 + 7) & 0x40) != 0 )
            v68 = *(_QWORD *)(v59 - 8);
          else
            v68 = v59 - 32LL * (*(_DWORD *)(v59 + 4) & 0x7FFFFFF);
          v69 = *(_QWORD *)(v68 + 32);
          v70 = *(_DWORD *)(v56 + 4) & 0x7FFFFFF;
          if ( v70 == *(_DWORD *)(v56 + 72) )
          {
            v89 = v58;
            sub_B48D90(v56);
            v58 = v89;
            v70 = *(_DWORD *)(v56 + 4) & 0x7FFFFFF;
          }
          v71 = (v70 + 1) & 0x7FFFFFF;
          v72 = v71 | *(_DWORD *)(v56 + 4) & 0xF8000000;
          v73 = *(_QWORD *)(v56 - 8) + 32LL * (unsigned int)(v71 - 1);
          *(_DWORD *)(v56 + 4) = v72;
          if ( *(_QWORD *)v73 )
          {
            v74 = *(_QWORD *)(v73 + 8);
            **(_QWORD **)(v73 + 16) = v74;
            if ( v74 )
              *(_QWORD *)(v74 + 16) = *(_QWORD *)(v73 + 16);
          }
          *(_QWORD *)v73 = v69;
          if ( v69 )
          {
            v75 = *(_QWORD *)(v69 + 16);
            *(_QWORD *)(v73 + 8) = v75;
            if ( v75 )
              *(_QWORD *)(v75 + 16) = v73 + 8;
            *(_QWORD *)(v73 + 16) = v69 + 16;
            *(_QWORD *)(v69 + 16) = v73;
          }
          *(_QWORD *)(*(_QWORD *)(v56 - 8)
                    + 32LL * *(unsigned int *)(v56 + 72)
                    + 8LL * ((*(_DWORD *)(v56 + 4) & 0x7FFFFFFu) - 1)) = v58;
        }
        v55 += 32;
        ++v57;
      }
      while ( v55 != v53 && v109 != v57 );
      v6 = v104;
      v7 = v100;
      v4 = v97;
      v2 = v93;
    }
    v107 = *v4;
  }
LABEL_26:
  v115 = 257;
  if ( (unsigned __int8)(v107 - 82) > 1u )
  {
    v24 = (unsigned __int8 *)sub_B504D0((unsigned int)*v4 - 29, v6, v7, (__int64)&v111, 0, 0);
    sub_B45260(v24, **(_QWORD **)(v2 - 8), 1);
    if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
    {
      v28 = *(_QWORD *)(v2 - 8);
      v91 = v28 + 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
    }
    else
    {
      v28 = v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
    }
    v29 = sub_116D080(v28, v91, 1);
    v31 = v30;
    for ( i = (unsigned __int64 *)v29; v31 != i; i += 4 )
    {
      v33 = *i;
      sub_B45560(v24, v33);
    }
    sub_116D800(a1, (__int64)v24, v2);
  }
  else
  {
    v24 = (unsigned __int8 *)sub_B52500(
                               (unsigned int)*v4 - 29,
                               *((_WORD *)v4 + 1) & 0x3F,
                               v6,
                               v7,
                               (__int64)&v111,
                               (__int64)v11,
                               0,
                               0);
    sub_116D800(a1, (__int64)v24, v2);
  }
  return v24;
}
