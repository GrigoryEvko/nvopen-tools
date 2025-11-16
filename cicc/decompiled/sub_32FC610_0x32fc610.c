// Function: sub_32FC610
// Address: 0x32fc610
//
__int64 __fastcall sub_32FC610(
        __int64 *a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // r12
  int v10; // eax
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned int v13; // edx
  unsigned int *v14; // rdi
  __int64 v15; // rax
  int v16; // esi
  __int64 v17; // rax
  __int64 result; // rax
  unsigned int *v19; // r15
  __int64 v20; // r10
  __int64 v21; // rax
  unsigned __int64 *v22; // rax
  unsigned int v23; // r15d
  char v24; // al
  unsigned __int64 v25; // r9
  __int64 v26; // r10
  char v27; // r11
  __int64 v28; // rax
  unsigned __int16 v29; // r14
  __int64 v30; // r12
  __int64 v31; // rbx
  __int64 v32; // rsi
  unsigned int v33; // r8d
  int v34; // eax
  int v35; // r13d
  unsigned __int64 v36; // r14
  __int128 v37; // rax
  int v38; // r9d
  __int64 v39; // rax
  __int64 v40; // rax
  unsigned __int64 v41; // rax
  int v42; // edi
  int v43; // r8d
  __int64 v44; // rax
  unsigned int v45; // r15d
  __int64 v46; // r13
  __int64 v47; // r14
  __int64 v48; // rax
  __int64 v49; // rbx
  __int64 v50; // rdx
  __int64 v51; // r15
  unsigned __int16 *v52; // r13
  __int64 v53; // r8
  __int64 v54; // rcx
  unsigned __int16 (__fastcall *v55)(__int64, __int64, __int64, __int64, __int64); // r13
  __int64 v56; // rax
  int v57; // edx
  __int128 v58; // rax
  __int64 v59; // rax
  int v60; // edx
  __int64 v61; // rcx
  __int64 (__fastcall *v62)(__int64, __int64, __int64, __int64, __int64); // r14
  __int64 v63; // rax
  unsigned __int16 v64; // ax
  int v65; // edx
  int v66; // eax
  int v67; // eax
  __int64 v68; // rdx
  int v69; // eax
  __int128 v70; // [rsp-40h] [rbp-160h]
  __int128 v71; // [rsp-30h] [rbp-150h]
  __int128 v72; // [rsp-30h] [rbp-150h]
  __int128 v73; // [rsp-20h] [rbp-140h]
  __int64 v74; // [rsp+0h] [rbp-120h]
  char v75; // [rsp+Fh] [rbp-111h]
  __int64 v76; // [rsp+10h] [rbp-110h]
  unsigned __int64 v77; // [rsp+10h] [rbp-110h]
  int v78; // [rsp+18h] [rbp-108h]
  __int64 v79; // [rsp+18h] [rbp-108h]
  __int64 v80; // [rsp+18h] [rbp-108h]
  unsigned __int64 v81; // [rsp+20h] [rbp-100h]
  __int64 v82; // [rsp+20h] [rbp-100h]
  __int64 v83; // [rsp+20h] [rbp-100h]
  unsigned __int64 v84; // [rsp+28h] [rbp-F8h]
  __int64 v85; // [rsp+28h] [rbp-F8h]
  unsigned __int64 v86; // [rsp+28h] [rbp-F8h]
  __int64 v87; // [rsp+28h] [rbp-F8h]
  __int64 v88; // [rsp+30h] [rbp-F0h]
  __int64 v89; // [rsp+30h] [rbp-F0h]
  unsigned __int64 v90; // [rsp+30h] [rbp-F0h]
  __int64 v91; // [rsp+30h] [rbp-F0h]
  __int64 v92; // [rsp+30h] [rbp-F0h]
  __int64 v93; // [rsp+30h] [rbp-F0h]
  __int64 v94; // [rsp+30h] [rbp-F0h]
  __int64 v95; // [rsp+30h] [rbp-F0h]
  unsigned __int64 v96; // [rsp+30h] [rbp-F0h]
  unsigned __int64 v97; // [rsp+40h] [rbp-E0h]
  __int64 v98; // [rsp+40h] [rbp-E0h]
  __int64 v99; // [rsp+40h] [rbp-E0h]
  __int64 v100; // [rsp+40h] [rbp-E0h]
  int v101; // [rsp+48h] [rbp-D8h]
  __int64 v102; // [rsp+48h] [rbp-D8h]
  __int64 v103; // [rsp+48h] [rbp-D8h]
  __int64 v104; // [rsp+48h] [rbp-D8h]
  __int64 v105; // [rsp+50h] [rbp-D0h] BYREF
  int v106; // [rsp+58h] [rbp-C8h]
  __int64 v107[8]; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v108; // [rsp+A0h] [rbp-80h]
  int v109; // [rsp+A8h] [rbp-78h]
  __int64 v110; // [rsp+B0h] [rbp-70h]
  __int64 v111; // [rsp+B8h] [rbp-68h]
  __int64 v112; // [rsp+C0h] [rbp-60h] BYREF
  unsigned int v113; // [rsp+C8h] [rbp-58h]
  __int64 *v114; // [rsp+D0h] [rbp-50h]
  __int64 v115; // [rsp+D8h] [rbp-48h]
  __int64 v116; // [rsp+E0h] [rbp-40h] BYREF

  v6 = a3;
  v7 = a2;
  v8 = (unsigned int)a3;
  v10 = *(_DWORD *)(a2 + 24);
  if ( v10 != 192 )
  {
    if ( v10 != 216 )
      goto LABEL_3;
    v14 = *(unsigned int **)(a2 + 40);
    v7 = *(_QWORD *)v14;
    v8 = v14[2];
    v15 = *(_QWORD *)(*(_QWORD *)v14 + 56LL);
    if ( !v15 )
      return 0;
    v16 = 1;
    do
    {
      while ( (_DWORD)v8 != *(_DWORD *)(v15 + 8) )
      {
        v15 = *(_QWORD *)(v15 + 32);
        if ( !v15 )
          goto LABEL_21;
      }
      if ( !v16 )
        return 0;
      v17 = *(_QWORD *)(v15 + 32);
      if ( !v17 )
        goto LABEL_22;
      if ( (_DWORD)v8 == *(_DWORD *)(v17 + 8) )
        return 0;
      v15 = *(_QWORD *)(v17 + 32);
      v16 = 0;
    }
    while ( v15 );
LABEL_21:
    if ( v16 == 1 )
      return 0;
LABEL_22:
    if ( *(_DWORD *)(v7 + 24) != 192 )
      return 0;
    a2 = 0xFFFFFFFF00000000LL;
    v6 = a3 & 0xFFFFFFFF00000000LL | v14[2];
  }
  v19 = *(unsigned int **)(v7 + 40);
  v20 = *(_QWORD *)v19;
  if ( *(_DWORD *)(*(_QWORD *)v19 + 24LL) != 186 )
    return 0;
  a3 = *((_QWORD *)v19 + 5);
  if ( *(_DWORD *)(a3 + 24) != 11 )
    goto LABEL_28;
  v39 = *(_QWORD *)(*(_QWORD *)(v20 + 40) + 40LL);
  if ( *(_DWORD *)(v39 + 24) != 11 )
    goto LABEL_28;
  v40 = *(_QWORD *)(v39 + 96);
  a2 = *(unsigned int *)(v40 + 32);
  if ( (unsigned int)a2 > 0x40 )
  {
    v86 = *((_QWORD *)v19 + 5);
    v99 = *(_QWORD *)v19;
    v104 = v40 + 24;
    v66 = sub_C44630(v40 + 24);
    a2 = (unsigned int)a2;
    a3 = v86;
    if ( v66 != 1 )
      goto LABEL_28;
    v67 = sub_C444A0(v104);
    a3 = v86;
    v20 = v99;
    v42 = v67;
  }
  else
  {
    v41 = *(_QWORD *)(v40 + 24);
    if ( !v41 || (v41 & (v41 - 1)) != 0 )
      goto LABEL_28;
    _BitScanReverse64(&v41, v41);
    v42 = a2 + (v41 ^ 0x3F) - 64;
  }
  a3 = *(_QWORD *)(a3 + 96);
  v43 = a2 - 1;
  a2 = *(unsigned int *)(a3 + 32);
  a5 = (unsigned int)(v43 - v42);
  v101 = *(_DWORD *)(a3 + 32);
  if ( (unsigned int)a2 > 0x40 )
  {
    v87 = v20;
    v100 = a5;
    v96 = a3;
    v69 = sub_C444A0(a3 + 24);
    a3 = v96;
    a5 = v100;
    v20 = v87;
    a2 = (unsigned int)(v101 - v69);
    if ( (unsigned int)a2 > 0x40 )
      goto LABEL_28;
    v44 = **(_QWORD **)(v96 + 24);
  }
  else
  {
    v44 = *(_QWORD *)(a3 + 24);
  }
  if ( a5 != v44 )
  {
LABEL_28:
    v10 = *(_DWORD *)(v7 + 24);
LABEL_3:
    if ( v10 == 188 )
    {
      v109 = 0;
      v107[6] = sub_33ECD10(1, a2, a3, a4, a5, a6);
      v108 = 0x100000000LL;
      v111 = 0xFFFFFFFFLL;
      v116 = 0;
      memset(v107, 0, 24);
      v107[3] = 328;
      v107[4] = -65536;
      v107[7] = 0;
      v110 = 0;
      v115 = 0;
      v114 = v107;
      v112 = v7;
      v113 = v8;
      v11 = *(_QWORD *)(v7 + 56);
      v116 = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 24) = &v116;
      v115 = v7 + 56;
      *(_QWORD *)(v7 + 56) = &v112;
      LODWORD(v108) = 1;
      v107[5] = (__int64)&v112;
      if ( *(_DWORD *)(v7 + 24) != 188 )
        goto LABEL_29;
      while ( 1 )
      {
        v12 = sub_32FA5C0(a1, v7);
        if ( !v12 )
          break;
        if ( v7 == v12 )
        {
          v7 = v112;
          v8 = v113;
          v6 = v113 | v6 & 0xFFFFFFFF00000000LL;
        }
        else
        {
          v8 = v13;
          v7 = v12;
        }
        if ( *(_DWORD *)(v7 + 24) != 188 )
          goto LABEL_29;
      }
      if ( *(_DWORD *)(v7 + 24) != 188 )
      {
LABEL_29:
        v21 = v7;
LABEL_30:
        v88 = v21;
        sub_33CF710(v107);
        return v88;
      }
      v22 = *(unsigned __int64 **)(v7 + 40);
      v23 = *((_DWORD *)v22 + 2);
      v97 = v22[1];
      v89 = *v22;
      v84 = v22[6];
      if ( *(_DWORD *)(*v22 + 24) != 208 )
      {
        v81 = v22[5];
        if ( *(_DWORD *)(v81 + 24) != 208 )
        {
          v24 = sub_33DFCF0(v7, v8 | v6 & 0xFFFFFFFF00000000LL, 0);
          v25 = v81;
          v26 = v89;
          v27 = v24;
          if ( v24 )
          {
            v59 = *(_QWORD *)(v89 + 56);
            v60 = 1;
            if ( !v59 )
              goto LABEL_63;
            do
            {
              if ( v23 == *(_DWORD *)(v59 + 8) )
              {
                if ( !v60 )
                  goto LABEL_63;
                v59 = *(_QWORD *)(v59 + 32);
                if ( !v59 )
                  goto LABEL_66;
                if ( *(_DWORD *)(v59 + 8) == v23 )
                  goto LABEL_63;
                v60 = 0;
              }
              v59 = *(_QWORD *)(v59 + 32);
            }
            while ( v59 );
            if ( v60 == 1 )
              goto LABEL_63;
LABEL_66:
            if ( *(_DWORD *)(v89 + 24) == 188
              && (v28 = *(_QWORD *)(v89 + 48) + 16LL * v23, v29 = *(_WORD *)v28, *(_WORD *)v28 == 2) )
            {
              v68 = *(_QWORD *)(v89 + 40);
              v7 = v89;
              v23 = *(_DWORD *)(v68 + 8);
              v25 = *(_QWORD *)(v68 + 40);
              v26 = *(_QWORD *)v68;
              v97 = v23 | v97 & 0xFFFFFFFF00000000LL;
              v84 = *(unsigned int *)(v68 + 48) | v84 & 0xFFFFFFFF00000000LL;
            }
            else
            {
LABEL_63:
              v27 = 0;
              v28 = 16 * v8 + *(_QWORD *)(v7 + 48);
              v29 = *(_WORD *)v28;
            }
          }
          else
          {
            v28 = 16 * v8 + *(_QWORD *)(v7 + 48);
            v29 = *(_WORD *)v28;
          }
          v30 = *(_QWORD *)(v28 + 8);
          if ( *((_BYTE *)a1 + 34) )
          {
            v74 = v26;
            v61 = v29;
            v75 = v27;
            v95 = a1[1];
            v77 = v25;
            v62 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v95 + 528LL);
            v80 = v61;
            v83 = *(_QWORD *)(*a1 + 64);
            v63 = sub_2E79000(*(__int64 **)(*a1 + 40));
            v64 = v62(v95, v63, v83, v80, v30);
            v26 = v74;
            v27 = v75;
            v25 = v77;
            v29 = v64;
            LODWORD(v30) = v65;
          }
          v31 = *a1;
          if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)v31 + 544LL) - 42) > 1 )
          {
            v32 = *(_QWORD *)(v7 + 80);
            v105 = v32;
            v33 = v27 == 0 ? 22 : 17;
            if ( v32 )
            {
              v76 = v26;
              v78 = v27 == 0 ? 22 : 17;
              v90 = v25;
              sub_B96E90((__int64)&v105, v32, 1);
              v26 = v76;
              v33 = v78;
              v25 = v90;
            }
            v34 = *(_DWORD *)(v7 + 72);
            v91 = v26;
            v35 = v29;
            v106 = v34;
            v36 = v25;
            *(_QWORD *)&v37 = sub_33ED040(v31, v33);
            *((_QWORD *)&v73 + 1) = v84;
            *(_QWORD *)&v73 = v36;
            *((_QWORD *)&v71 + 1) = v23 | v97 & 0xFFFFFFFF00000000LL;
            *(_QWORD *)&v71 = v91;
            v21 = sub_340F900(v31, 208, (unsigned int)&v105, v35, v30, v38, v71, v73, v37);
            if ( v105 )
            {
              v92 = v21;
              sub_B91220((__int64)&v105, v105);
              v21 = v92;
            }
            goto LABEL_30;
          }
        }
      }
      sub_33CF710(v107);
    }
    return 0;
  }
  v45 = v19[2];
  v93 = v20;
  sub_3285E70((__int64)v107, v7);
  v46 = 16LL * v45;
  v102 = v45;
  v79 = *a1;
  v47 = sub_3400BD0(
          *a1,
          0,
          (unsigned int)v107,
          *(unsigned __int16 *)(*(_QWORD *)(v93 + 48) + v46),
          *(_QWORD *)(*(_QWORD *)(v93 + 48) + v46 + 8),
          0,
          0);
  v48 = *a1;
  v49 = a1[1];
  v51 = v50;
  v52 = (unsigned __int16 *)(*(_QWORD *)(v93 + 48) + v46);
  v82 = v93;
  v53 = *((_QWORD *)v52 + 1);
  v54 = *v52;
  v55 = *(unsigned __int16 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v49 + 528LL);
  v85 = v53;
  v98 = v54;
  v94 = *(_QWORD *)(v48 + 64);
  v56 = sub_2E79000(*(__int64 **)(v48 + 40));
  LODWORD(v49) = v55(v49, v56, v94, v98, v85);
  LODWORD(v55) = v57;
  *(_QWORD *)&v58 = sub_33ED040(v79, 22);
  *((_QWORD *)&v72 + 1) = v51;
  *(_QWORD *)&v72 = v47;
  *((_QWORD *)&v70 + 1) = v102;
  *(_QWORD *)&v70 = v82;
  result = sub_340F900(v79, 208, (unsigned int)v107, v49, (_DWORD)v55, v102, v70, v72, v58);
  if ( v107[0] )
  {
    v103 = result;
    sub_B91220((__int64)v107, v107[0]);
    return v103;
  }
  return result;
}
