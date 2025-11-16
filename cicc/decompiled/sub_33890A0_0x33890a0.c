// Function: sub_33890A0
// Address: 0x33890a0
//
void __fastcall sub_33890A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r11
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v11; // r13
  __int64 v12; // rbx
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rdi
  bool v21; // al
  __int64 v22; // r11
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  _BYTE *v27; // rdi
  __int64 v28; // r9
  __int64 v29; // r8
  unsigned int v30; // r10d
  __int64 v31; // rdi
  __int64 (*v32)(); // rax
  unsigned int v33; // eax
  bool v34; // al
  __int64 v35; // rdx
  __int64 v36; // rcx
  unsigned int v37; // r12d
  __int64 v38; // r13
  int v39; // ebx
  __int64 v40; // rax
  _QWORD *v41; // r15
  __int64 v42; // r14
  unsigned __int64 *v43; // r9
  unsigned __int64 v44; // rcx
  __int64 v45; // r12
  __int64 v46; // rbx
  __int64 v47; // rsi
  __int64 v48; // rsi
  __int64 v49; // rsi
  __int64 v50; // r12
  __int64 v51; // rdx
  __int64 v52; // r13
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int128 v56; // rax
  int v57; // r9d
  __int128 v58; // rcx
  __int64 v59; // rax
  __int64 v60; // r11
  bool v61; // zf
  __int64 v62; // rsi
  __int64 v63; // rax
  __int64 v64; // r11
  __int64 v65; // rbx
  int v66; // edx
  int v67; // r12d
  _QWORD *v68; // rax
  __int64 v69; // r13
  bool v70; // al
  __int64 *v71; // rdx
  __int64 v72; // rdx
  _QWORD *v73; // rcx
  __int64 v74; // rax
  int v75; // edx
  __int64 *v76; // rdx
  unsigned int v77; // r14d
  int v78; // r13d
  __int64 v79; // rax
  __int64 v80; // rbx
  __int64 *v81; // rdx
  __int128 v82; // [rsp+0h] [rbp-120h]
  __int64 v83; // [rsp+10h] [rbp-110h]
  __int64 v84; // [rsp+10h] [rbp-110h]
  __int64 v85; // [rsp+10h] [rbp-110h]
  __int64 v86; // [rsp+18h] [rbp-108h]
  __int64 v87; // [rsp+18h] [rbp-108h]
  __int64 v88; // [rsp+18h] [rbp-108h]
  __int64 v89; // [rsp+18h] [rbp-108h]
  __int64 v90; // [rsp+18h] [rbp-108h]
  __int64 v91; // [rsp+20h] [rbp-100h]
  __int64 v92; // [rsp+20h] [rbp-100h]
  __int64 v93; // [rsp+20h] [rbp-100h]
  unsigned int v94; // [rsp+20h] [rbp-100h]
  unsigned int v95; // [rsp+20h] [rbp-100h]
  __int64 v96; // [rsp+20h] [rbp-100h]
  __int64 v97; // [rsp+20h] [rbp-100h]
  __int64 v98; // [rsp+20h] [rbp-100h]
  __int64 v99; // [rsp+20h] [rbp-100h]
  __int64 v100; // [rsp+20h] [rbp-100h]
  unsigned int v101; // [rsp+20h] [rbp-100h]
  __int64 v102; // [rsp+30h] [rbp-F0h]
  int v103; // [rsp+30h] [rbp-F0h]
  __int64 v104; // [rsp+30h] [rbp-F0h]
  __int64 v105; // [rsp+30h] [rbp-F0h]
  __int64 v106; // [rsp+30h] [rbp-F0h]
  __int64 v107; // [rsp+30h] [rbp-F0h]
  __int64 v108; // [rsp+30h] [rbp-F0h]
  bool v109; // [rsp+38h] [rbp-E8h]
  __int64 v111; // [rsp+38h] [rbp-E8h]
  __int64 v112; // [rsp+38h] [rbp-E8h]
  __int64 v113; // [rsp+74h] [rbp-ACh]
  int v114; // [rsp+7Ch] [rbp-A4h]
  __int64 v115; // [rsp+80h] [rbp-A0h] BYREF
  int v116; // [rsp+88h] [rbp-98h]
  __int64 v117; // [rsp+90h] [rbp-90h] BYREF
  __int64 v118; // [rsp+98h] [rbp-88h]
  __int64 v119; // [rsp+A0h] [rbp-80h]
  __int64 v120; // [rsp+A8h] [rbp-78h]
  __int64 v121; // [rsp+B0h] [rbp-70h]
  __int64 v122; // [rsp+B8h] [rbp-68h]
  __int64 v123; // [rsp+C0h] [rbp-60h]
  __int64 v124; // [rsp+C8h] [rbp-58h] BYREF
  int v125; // [rsp+D0h] [rbp-50h]
  __int64 v126; // [rsp+D8h] [rbp-48h] BYREF
  __int64 v127; // [rsp+E0h] [rbp-40h]
  bool v128; // [rsp+E8h] [rbp-38h]

  v6 = a2;
  v7 = a1;
  v8 = *(_QWORD *)(a1 + 960);
  v9 = *(_QWORD *)(v8 + 56);
  v10 = *(_QWORD *)(v8 + 744);
  v11 = *(_QWORD *)(v9 + 8LL * *(unsigned int *)(*(_QWORD *)(a2 - 32) + 44LL));
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) == 1 )
  {
    sub_2E33F80(v10, v11, -1, a4, a5, a6);
    if ( v11 != sub_3374B60(a1, v10) || !*(_DWORD *)(*(_QWORD *)(a1 + 856) + 648LL) )
    {
      v49 = v11;
      v98 = a2;
      v111 = *(_QWORD *)(a1 + 864);
      v50 = sub_33EEAD0(v111, v11);
      v52 = v51;
      *(_QWORD *)&v56 = sub_3373A60(a1, v49, v51, v53, v54, v55);
      v57 = v111;
      v117 = 0;
      v58 = v56;
      v59 = *(_QWORD *)a1;
      v60 = v98;
      v61 = *(_QWORD *)a1 == 0;
      LODWORD(v118) = *(_DWORD *)(a1 + 848);
      if ( !v61 && &v117 != (__int64 *)(v59 + 48) )
      {
        v62 = *(_QWORD *)(v59 + 48);
        v117 = v62;
        if ( v62 )
        {
          v106 = v98;
          v99 = v58;
          sub_B96E90((__int64)&v117, v62, 1);
          v60 = v106;
          *(_QWORD *)&v58 = v99;
          v57 = v111;
        }
      }
      *((_QWORD *)&v82 + 1) = v52;
      *(_QWORD *)&v82 = v50;
      v112 = v60;
      v63 = sub_3406EB0(v57, 301, (unsigned int)&v117, 1, 0, v57, v58, v82);
      v64 = v112;
      v65 = v63;
      v67 = v66;
      if ( v117 )
      {
        sub_B91220((__int64)&v117, v117);
        v64 = v112;
      }
      v117 = v64;
      v68 = sub_337DC20(a1 + 8, &v117);
      *v68 = v65;
      *((_DWORD *)v68 + 2) = v67;
      v69 = *(_QWORD *)(a1 + 864);
      if ( v65 )
      {
        nullsub_1875(v65, *(_QWORD *)(a1 + 864), 0);
        *(_QWORD *)(v69 + 384) = v65;
        *(_DWORD *)(v69 + 392) = v67;
        sub_33E2B60(v69, 0);
      }
      else
      {
        *(_QWORD *)(v69 + 384) = 0;
        *(_DWORD *)(v69 + 392) = v67;
      }
    }
    return;
  }
  v12 = *(_QWORD *)(a2 - 96);
  v13 = *(_QWORD *)(v9 + 8LL * *(unsigned int *)(*(_QWORD *)(a2 - 64) + 44LL));
  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
  {
    v14 = sub_B91C10(a2, 15);
    v6 = a2;
    v109 = v14 != 0;
  }
  else
  {
    v109 = 0;
    v14 = 0;
  }
  if ( *(_BYTE *)v12 <= 0x1Cu )
    goto LABEL_7;
  if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL) + 56LL) )
    goto LABEL_7;
  v15 = *(_QWORD *)(v12 + 16);
  if ( !v15 || *(_QWORD *)(v15 + 8) | v14 )
    goto LABEL_7;
  v20 = *(_QWORD *)(v12 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v20 + 8) - 17 <= 1 )
    v20 = **(_QWORD **)(v20 + 16);
  v91 = v6;
  v21 = sub_BCAC40(v20, 1);
  v22 = v91;
  if ( !v21 )
    goto LABEL_71;
  if ( *(_BYTE *)v12 == 57 )
  {
    if ( (*(_BYTE *)(v12 + 7) & 0x40) != 0 )
      v76 = *(__int64 **)(v12 - 8);
    else
      v76 = (__int64 *)(v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF));
    v29 = *v76;
    if ( !*v76 )
      goto LABEL_71;
    v28 = v76[4];
    if ( !v28 )
      goto LABEL_71;
  }
  else
  {
    if ( *(_BYTE *)v12 != 86 )
      goto LABEL_71;
    v23 = *(_QWORD *)(v12 + 8);
    v86 = *(_QWORD *)(v12 - 96);
    if ( v23 != *(_QWORD *)(v86 + 8) || **(_BYTE **)(v12 - 32) > 0x15u )
    {
LABEL_28:
      if ( (unsigned int)*(unsigned __int8 *)(v23 + 8) - 17 <= 1 )
        v23 = **(_QWORD **)(v23 + 16);
      v92 = v22;
      if ( !sub_BCAC40(v23, 1) )
        goto LABEL_7;
      v22 = v92;
      if ( *(_BYTE *)v12 == 58 )
      {
        if ( (*(_BYTE *)(v12 + 7) & 0x40) != 0 )
          v81 = *(__int64 **)(v12 - 8);
        else
          v81 = (__int64 *)(v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF));
        v29 = *v81;
        if ( *v81 )
        {
          v28 = v81[4];
          v30 = 29;
          if ( v28 )
            goto LABEL_38;
        }
      }
      else
      {
        v87 = v92;
        if ( *(_BYTE *)v12 == 86 )
        {
          v26 = *(_QWORD *)(v12 - 96);
          v102 = v26;
          if ( *(_QWORD *)(v26 + 8) == *(_QWORD *)(v12 + 8) )
          {
            v27 = *(_BYTE **)(v12 - 64);
            if ( *v27 <= 0x15u )
            {
              v93 = *(_QWORD *)(v12 - 32);
              if ( sub_AD7A80(v27, 1, v24, v25, v26) )
              {
                v28 = v93;
                if ( v93 )
                {
                  v22 = v87;
                  v29 = v102;
                  v30 = 29;
                  goto LABEL_38;
                }
              }
            }
          }
        }
      }
LABEL_7:
      v16 = *(_DWORD *)(v7 + 848);
      v17 = *(_QWORD *)v7;
      v115 = 0;
      v116 = v16;
      if ( v17 )
      {
        if ( &v115 != (__int64 *)(v17 + 48) )
        {
          v18 = *(_QWORD *)(v17 + 48);
          v115 = v18;
          if ( v18 )
            sub_B96E90((__int64)&v115, v18, 1);
        }
      }
      v19 = sub_ACD6D0(*(__int64 **)(*(_QWORD *)(v7 + 864) + 64LL));
      v118 = v12;
      LODWORD(v117) = 17;
      v119 = 0;
      v120 = v19;
      v121 = v11;
      v122 = v13;
      v123 = v10;
      v124 = v115;
      if ( v115 )
      {
        sub_B96E90((__int64)&v124, v115, 1);
        v126 = 0;
        v127 = -1;
        v125 = v116;
        v128 = v109;
        if ( v115 )
          sub_B91220((__int64)&v115, v115);
      }
      else
      {
        v126 = 0;
        v127 = -1;
        v125 = v116;
        v128 = v109;
      }
      sub_3391190(v7, &v117, v10);
      if ( v126 )
        sub_B91220((__int64)&v126, v126);
      if ( v124 )
        sub_B91220((__int64)&v124, v124);
      return;
    }
    v107 = v91;
    v100 = *(_QWORD *)(v12 - 64);
    v70 = sub_AC30F0(*(_QWORD *)(v12 - 32));
    v28 = v100;
    v22 = v107;
    v29 = v86;
    if ( !v70 || !v100 )
    {
LABEL_71:
      v23 = *(_QWORD *)(v12 + 8);
      goto LABEL_28;
    }
  }
  v30 = 28;
LABEL_38:
  if ( *(_BYTE *)v29 == 90 )
  {
    v71 = (*(_BYTE *)(v29 + 7) & 0x40) != 0
        ? *(__int64 **)(v29 - 8)
        : (__int64 *)(v29 - 32LL * (*(_DWORD *)(v29 + 4) & 0x7FFFFFF));
    v72 = *v71;
    if ( v72 && *(_BYTE *)v28 == 90 )
    {
      v73 = (*(_BYTE *)(v28 + 7) & 0x40) != 0
          ? *(_QWORD **)(v28 - 8)
          : (_QWORD *)(v28 - 32LL * (*(_DWORD *)(v28 + 4) & 0x7FFFFFF));
      if ( v72 == *v73 )
        goto LABEL_7;
    }
  }
  v31 = *(_QWORD *)(*(_QWORD *)(v7 + 864) + 16LL);
  v32 = *(__int64 (**)())(*(_QWORD *)v31 + 232LL);
  if ( v32 == sub_2FE2F70 )
  {
    v113 = -1;
    v114 = -1;
  }
  else
  {
    v85 = v22;
    v90 = v28;
    v108 = v29;
    v101 = v30;
    v74 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, __int64))v32)(v31, v30, v29, v28);
    v22 = v85;
    v28 = v90;
    v113 = v74;
    v29 = v108;
    v114 = v75;
    v30 = v101;
  }
  v94 = v30;
  if ( (unsigned __int8)sub_3388570(
                          v7,
                          *(_QWORD *)(v7 + 960),
                          v22,
                          v30,
                          (_BYTE *)v29,
                          (unsigned __int8 **)v28,
                          v113,
                          v114) )
    goto LABEL_7;
  v103 = v94;
  v95 = sub_3373D80(v7, v10, v13);
  v33 = sub_3373D80(v7, v10, v11);
  sub_3382F60(v7, v12, v11, v13, v10, v10, v103, v33, v95, 0);
  v34 = sub_3373E80(v7, (int **)(*(_QWORD *)(v7 + 896) + 8LL));
  v35 = *(_QWORD *)(v7 + 896);
  if ( !v34 )
  {
    v36 = *(_QWORD *)(v35 + 8);
    if ( -1431655765 * (unsigned int)((*(_QWORD *)(v35 + 16) - v36) >> 5) != 1 )
    {
      v96 = v10;
      v104 = v11;
      v37 = 1;
      v38 = v7;
      v83 = v12;
      v39 = -1431655765 * ((*(_QWORD *)(v35 + 16) - v36) >> 5);
      v88 = v13;
      while ( 1 )
      {
        v40 = v37++;
        v41 = *(_QWORD **)(v36 + 96 * v40 + 48);
        v42 = *(_QWORD *)(*(_QWORD *)(v38 + 960) + 8LL) + 320LL;
        sub_2E31020(v42, (__int64)v41);
        v43 = (unsigned __int64 *)v41[1];
        v44 = *v41 & 0xFFFFFFFFFFFFFFF8LL;
        *v43 = v44 | *v43 & 7;
        *(_QWORD *)(v44 + 8) = v43;
        *v41 &= 7uLL;
        v41[1] = 0;
        sub_2E79D60(v42, v41);
        if ( v39 == v37 )
          break;
        v36 = *(_QWORD *)(*(_QWORD *)(v38 + 896) + 8LL);
      }
      v7 = v38;
      v10 = v96;
      v13 = v88;
      v12 = v83;
      v11 = v104;
      v35 = *(_QWORD *)(v7 + 896);
    }
    v97 = *(_QWORD *)(v35 + 8);
    if ( v97 != *(_QWORD *)(v35 + 16) )
    {
      v84 = v35;
      v105 = v10;
      v45 = *(_QWORD *)(v35 + 8);
      v89 = v12;
      v46 = *(_QWORD *)(v35 + 16);
      do
      {
        v47 = *(_QWORD *)(v45 + 72);
        if ( v47 )
          sub_B91220(v45 + 72, v47);
        v48 = *(_QWORD *)(v45 + 56);
        if ( v48 )
          sub_B91220(v45 + 56, v48);
        v45 += 96;
      }
      while ( v46 != v45 );
      v10 = v105;
      v12 = v89;
      *(_QWORD *)(v84 + 16) = v97;
    }
    goto LABEL_7;
  }
  v77 = 1;
  v78 = -1431655765 * ((__int64)(*(_QWORD *)(v35 + 16) - *(_QWORD *)(v35 + 8)) >> 5);
  if ( v78 != 1 )
  {
    do
    {
      v79 = v77++;
      v80 = 96 * v79;
      sub_33C4170(v7, *(_QWORD *)(*(_QWORD *)(v35 + 8) + 96 * v79 + 8));
      sub_33C4170(v7, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v7 + 896) + 8LL) + v80 + 24));
      v35 = *(_QWORD *)(v7 + 896);
    }
    while ( v78 != v77 );
  }
  sub_3391190(v7, *(_QWORD *)(v35 + 8), v10);
  sub_3377160(*(_QWORD *)(v7 + 896) + 8LL, *(_QWORD *)(*(_QWORD *)(v7 + 896) + 8LL));
}
