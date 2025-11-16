// Function: sub_2B34820
// Address: 0x2b34820
//
void __fastcall sub_2B34820(__int64 *a1, __int64 a2, unsigned int a3, char a4)
{
  __int64 v6; // r12
  __int64 v8; // rdi
  __int64 v9; // r14
  int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // rax
  char *v17; // rax
  unsigned int **v18; // r15
  int v19; // esi
  char v20; // cl
  __int64 v21; // rax
  __int64 v22; // rax
  char v23; // cl
  unsigned int **v24; // r14
  int v25; // esi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 *v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // r13
  __int64 v31; // rax
  unsigned int v32; // r8d
  unsigned int v33; // eax
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // r9
  __int64 v37; // rdi
  unsigned __int8 *v38; // rax
  __int64 v39; // rax
  __int64 v40; // r9
  __int64 v41; // rdi
  __int64 v42; // r15
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  unsigned __int8 *v46; // rax
  __int64 v47; // rax
  __int64 v48; // r15
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 *v53; // rdx
  __int64 v54; // rdi
  __int64 v55; // r9
  __int64 v56; // r15
  __int64 v57; // rax
  unsigned int **v58; // r15
  int v59; // esi
  __int64 v60; // rax
  __int64 v61; // rax
  unsigned __int8 *v62; // rax
  __int64 v63; // rax
  __int64 *v64; // r11
  __int64 v65; // r15
  __int64 v66; // rdx
  unsigned int v67; // esi
  int v68; // eax
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rax
  __int64 v72; // r15
  __int64 v73; // r15
  __int64 v74; // rdx
  unsigned int v75; // esi
  __int64 v76; // rax
  __int64 v77; // r9
  unsigned __int64 v78; // rsi
  __int64 v79; // rdx
  __int64 v80; // rdi
  __int64 v81; // r10
  __int64 v82; // rax
  __int64 v83; // rsi
  __int64 v84; // rcx
  __int64 v85; // rax
  unsigned __int64 v86; // rcx
  __int64 v87; // rdx
  unsigned int **v88; // rdi
  unsigned int v89; // r14d
  __int64 v90; // rax
  __int64 v91; // rax
  unsigned int v92; // r15d
  __int64 v93; // r8
  __int64 v94; // rcx
  __int64 v95; // rax
  unsigned __int64 v96; // rsi
  __int64 v97; // rdx
  __int64 *v98; // rax
  __int64 v99; // r9
  __int64 v100; // r8
  __int64 v101; // rdx
  unsigned __int64 v102; // rsi
  __int64 v103; // rax
  unsigned __int64 v104; // rcx
  __int64 v105; // rdx
  unsigned int **v106; // rdi
  unsigned int **v107; // rdi
  __int64 *v108; // rbx
  unsigned int v109; // et0
  __int64 v110; // [rsp+8h] [rbp-108h]
  __int64 v111; // [rsp+10h] [rbp-100h]
  __int64 v112; // [rsp+10h] [rbp-100h]
  int v113; // [rsp+10h] [rbp-100h]
  int v114; // [rsp+10h] [rbp-100h]
  char v115; // [rsp+18h] [rbp-F8h]
  __int64 *v116; // [rsp+18h] [rbp-F8h]
  __int64 v117; // [rsp+18h] [rbp-F8h]
  __int64 *v118; // [rsp+18h] [rbp-F8h]
  __int64 v119; // [rsp+18h] [rbp-F8h]
  __int64 v120; // [rsp+18h] [rbp-F8h]
  __int64 v121; // [rsp+20h] [rbp-F0h]
  unsigned int v122; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v123[4]; // [rsp+30h] [rbp-E0h] BYREF
  __int16 v124; // [rsp+50h] [rbp-C0h]
  _BYTE *v125; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v126; // [rsp+68h] [rbp-A8h]
  _BYTE v127[16]; // [rsp+70h] [rbp-A0h] BYREF
  __int16 v128; // [rsp+80h] [rbp-90h]
  const char *v129; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v130; // [rsp+A8h] [rbp-68h]
  _BYTE v131[16]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v132; // [rsp+C0h] [rbp-50h]

  v6 = a2;
  v8 = *(_QWORD *)(a2 + 8);
  v9 = v8;
  v10 = *(unsigned __int8 *)(v8 + 8);
  if ( (unsigned int)(v10 - 17) <= 1 )
    v9 = **(_QWORD **)(v8 + 16);
  if ( a3 > 1 )
  {
    BYTE4(v121) = (_BYTE)v10 == 18;
    LODWORD(v121) = *(_DWORD *)(v8 + 32);
    switch ( *(_DWORD *)(*a1 + 1576) )
    {
      case 0:
      case 2:
      case 0xB:
      case 0x10:
      case 0x11:
      case 0x12:
      case 0x13:
      case 0x14:
        BUG();
      case 1:
        v50 = sub_BCB2A0(*(_QWORD **)(a1[1] + 72));
        v53 = (__int64 *)a1[2];
        if ( v9 != v50 )
        {
          v54 = *v53;
          v55 = a3;
          if ( (unsigned int)*(unsigned __int8 *)(*v53 + 8) - 17 <= 1 )
            goto LABEL_45;
          if ( v9 != v54 )
            goto LABEL_46;
          goto LABEL_59;
        }
        if ( *v53 == v9 )
        {
          v55 = a3;
          v54 = v9;
          if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
          {
LABEL_45:
            v56 = a2;
            v54 = **(_QWORD **)(v54 + 16);
            if ( v9 != v54 )
            {
LABEL_46:
              v57 = *(_QWORD *)(a2 + 8);
              v58 = (unsigned int **)a1[1];
              v132 = 257;
              v59 = 1;
              if ( *(_BYTE *)(v57 + 8) == 17 )
                v59 = *(_DWORD *)(v57 + 32);
              v111 = v55;
              v60 = sub_2B08680(*v53, v59);
              v61 = sub_921630(v58, v6, v60, a4, (__int64)&v129);
              v55 = v111;
              v56 = v61;
              v54 = *(_QWORD *)a1[2];
              if ( (unsigned int)*(unsigned __int8 *)(v54 + 8) - 17 <= 1 )
                v54 = **(_QWORD **)(v54 + 16);
            }
LABEL_50:
            v62 = (unsigned __int8 *)sub_AD64C0(v54, v55, 0);
            v63 = sub_AD5E10(v121, v62);
            v64 = (__int64 *)a1[1];
            v112 = v63;
            v128 = 257;
            v118 = v64;
            v6 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v64[10]
                                                                                               + 32LL))(
                   v64[10],
                   17,
                   v56,
                   v63,
                   0,
                   0);
            if ( !v6 )
            {
              v132 = 257;
              v6 = sub_B504D0(17, v56, v112, (__int64)&v129, 0, 0);
              (*(void (__fastcall **)(__int64, __int64, _BYTE **, __int64, __int64))(*(_QWORD *)v118[11] + 16LL))(
                v118[11],
                v6,
                &v125,
                v118[7],
                v118[8]);
              v65 = *v118;
              v119 = *v118 + 16LL * *((unsigned int *)v118 + 2);
              while ( v119 != v65 )
              {
                v66 = *(_QWORD *)(v65 + 8);
                v67 = *(_DWORD *)v65;
                v65 += 16;
                sub_B99FD0(v6, v67, v66);
              }
            }
            break;
          }
LABEL_59:
          v56 = a2;
          goto LABEL_50;
        }
        v76 = *(_QWORD *)(a2 + 8);
        if ( *(_BYTE *)(v76 + 8) == 17 )
        {
          v77 = *(unsigned int *)(v76 + 32);
          v78 = (unsigned int)v77 * a3;
        }
        else
        {
          v78 = a3;
          v77 = 1;
        }
        v114 = v77;
        v129 = v131;
        v130 = 0xC00000000LL;
        sub_11B1960((__int64)&v129, v78, -1, v51, v52, v77);
        v79 = (__int64)v129;
        LODWORD(v80) = 0;
        v81 = 0;
        do
        {
          v82 = (unsigned int)v80;
          v80 = (unsigned int)(v80 + v114);
          v83 = v79 + 4 * v82;
          v84 = v79 + 4 * v80;
          if ( v83 != v84 )
          {
            v85 = 0;
            v86 = (unsigned __int64)(v84 - 4 - v83) >> 2;
            do
            {
              v87 = v85;
              *(_DWORD *)(v83 + 4 * v85) = v85;
              ++v85;
            }
            while ( v86 != v87 );
            v79 = (__int64)v129;
          }
          ++v81;
        }
        while ( a3 != v81 );
        v88 = (unsigned int **)a1[1];
        v128 = 257;
        v6 = sub_A83DF0(v88, v6, v79, (unsigned int)v130, (__int64)&v125);
        if ( v129 != v131 )
          _libc_free((unsigned __int64)v129);
        break;
      case 5:
        if ( (a3 & 1) == 0 )
          v6 = sub_AD6530(v8, a2);
        break;
      case 0xA:
        v46 = sub_AD8DD0(v9, (double)(int)a3);
        v47 = sub_AD5E10(v121, v46);
        v48 = a1[1];
        v128 = 257;
        v123[0] = v122;
        if ( *(_BYTE *)(v48 + 108) )
        {
          v6 = sub_B35400(v48, 0x6Cu, a2, v47, v122, (__int64)&v125, 0, 0, 0);
        }
        else
        {
          v117 = v47;
          v49 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(v48 + 80) + 40LL))(
                  *(_QWORD *)(v48 + 80),
                  18,
                  a2,
                  v47,
                  *(unsigned int *)(v48 + 104));
          if ( v49 )
          {
            v6 = v49;
          }
          else
          {
            v68 = *(_DWORD *)(v48 + 104);
            v132 = 257;
            v113 = v68;
            v69 = sub_B504D0(18, a2, v117, (__int64)&v129, 0, 0);
            v70 = *(_QWORD *)(v48 + 96);
            v6 = v69;
            if ( v70 )
              sub_B99FD0(v69, 3u, v70);
            sub_B45150(v6, v113);
            (*(void (__fastcall **)(_QWORD, __int64, _BYTE **, _QWORD, _QWORD))(**(_QWORD **)(v48 + 88) + 16LL))(
              *(_QWORD *)(v48 + 88),
              v6,
              &v125,
              *(_QWORD *)(v48 + 56),
              *(_QWORD *)(v48 + 64));
            v71 = *(_QWORD *)v48;
            v72 = 16LL * *(unsigned int *)(v48 + 8);
            v120 = v71 + v72;
            if ( v71 != v71 + v72 )
            {
              v73 = v71;
              do
              {
                v74 = *(_QWORD *)(v73 + 8);
                v75 = *(_DWORD *)v73;
                v73 += 16;
                sub_B99FD0(v6, v75, v74);
              }
              while ( v120 != v73 );
            }
          }
        }
        break;
      default:
        break;
    }
  }
  v11 = (__int64 *)a1[3];
  if ( !*v11 )
  {
    *v11 = v6;
    *(_BYTE *)a1[4] = a4;
    return;
  }
  v12 = sub_BCB2A0(*(_QWORD **)(a1[1] + 72));
  v13 = (__int64 *)a1[2];
  if ( v9 != v12 )
    goto LABEL_9;
  if ( *v13 == v9 )
  {
    v14 = v9;
LABEL_10:
    v15 = a1[3];
    v16 = *(_QWORD *)(*(_QWORD *)v15 + 8LL);
    if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 <= 1 )
      v16 = **(_QWORD **)(v16 + 16);
    if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 > 1 )
    {
      if ( v16 == v14 )
        goto LABEL_18;
    }
    else if ( **(_QWORD **)(v14 + 16) == v16 )
    {
      goto LABEL_18;
    }
    v17 = (char *)a1[4];
    v18 = (unsigned int **)a1[1];
    v19 = 1;
    v132 = 257;
    v20 = *v17;
    v21 = *(_QWORD *)(*(_QWORD *)v15 + 8LL);
    if ( *(_BYTE *)(v21 + 8) == 17 )
      v19 = *(_DWORD *)(v21 + 32);
    v115 = v20;
    v22 = sub_2B08680(*v13, v19);
    v23 = v115;
    v116 = (__int64 *)a1[3];
    *v116 = sub_921630(v18, *v116, v22, v23, (__int64)&v129);
    v13 = (__int64 *)a1[2];
    v16 = *v13;
    if ( (unsigned int)*(unsigned __int8 *)(*v13 + 8) - 17 <= 1 )
      v16 = **(_QWORD **)(v16 + 16);
LABEL_18:
    if ( v9 != v16 )
    {
      v24 = (unsigned int **)a1[1];
      v25 = 1;
      v132 = 257;
      v26 = *(_QWORD *)(v6 + 8);
      if ( *(_BYTE *)(v26 + 8) == 17 )
        v25 = *(_DWORD *)(v26 + 32);
      v27 = sub_2B08680(*v13, v25);
      v6 = sub_921630(v24, v6, v27, a4, (__int64)&v129);
    }
    v28 = (__int64 *)a1[3];
    v29 = *(_QWORD *)(v6 + 8);
    v30 = *v28;
    v31 = *(_QWORD *)(*v28 + 8);
    if ( *(_BYTE *)(v31 + 8) == 17 )
    {
      v32 = *(_DWORD *)(v31 + 32);
      v33 = 1;
      if ( *(_BYTE *)(v29 + 8) != 17 )
      {
LABEL_25:
        if ( v33 > v32 )
        {
          *v28 = v6;
          v34 = *(_QWORD *)a1[3];
LABEL_27:
          v35 = sub_2B1F140(a1[1], v34, v32, 0);
          v36 = *a1;
          v37 = a1[1];
          v132 = 259;
          v129 = "rdx.op";
          v38 = sub_2B21B90(v37, *(_DWORD *)(v36 + 1576), v35, v30, (__int64)&v129, (unsigned __int8 ****)v36);
          v39 = sub_2B330C0(a1[1], *(_QWORD **)a1[3], (__int64)v38, 0, 0, 0);
LABEL_28:
          *(_QWORD *)a1[3] = v39;
          return;
        }
        if ( v33 != v32 )
        {
          v34 = *v28;
          v32 = v33;
          v30 = v6;
          goto LABEL_27;
        }
LABEL_32:
        v40 = *a1;
        v41 = a1[1];
        v129 = "rdx.op";
        v132 = 259;
        v39 = (__int64)sub_2B21B90(v41, *(_DWORD *)(v40 + 1576), v30, v6, (__int64)&v129, (unsigned __int8 ****)v40);
        goto LABEL_28;
      }
    }
    else
    {
      v32 = 1;
      if ( *(_BYTE *)(v29 + 8) != 17 )
        goto LABEL_32;
    }
    v33 = *(_DWORD *)(v29 + 32);
    goto LABEL_25;
  }
  v42 = *(_QWORD *)(*(_QWORD *)a1[3] + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v42 + 8) - 17 <= 1 )
    v42 = **(_QWORD **)(v42 + 16);
  if ( v42 != sub_BCB2A0(*(_QWORD **)(a1[1] + 72)) )
  {
    v13 = (__int64 *)a1[2];
LABEL_9:
    v14 = *v13;
    goto LABEL_10;
  }
  v89 = 1;
  v90 = *(_QWORD *)(*(_QWORD *)a1[3] + 8LL);
  if ( *(_BYTE *)(v90 + 8) == 17 )
    v89 = *(_DWORD *)(v90 + 32);
  v91 = *(_QWORD *)(v6 + 8);
  v92 = 1;
  if ( *(_BYTE *)(v91 + 8) == 17 )
    v92 = *(_DWORD *)(v91 + 32);
  v126 = 0xC00000000LL;
  v125 = v127;
  sub_11B1960((__int64)&v125, v92 + v89, -1, v43, v44, v45);
  v94 = (__int64)v125;
  if ( v125 != &v125[4 * (unsigned int)v126] )
  {
    v95 = 0;
    v96 = (4 * (unsigned __int64)(unsigned int)v126 - 4) >> 2;
    do
    {
      v97 = v95;
      *(_DWORD *)(v94 + 4 * v95) = v95;
      ++v95;
    }
    while ( v96 != v97 );
  }
  if ( v92 > v89 )
  {
    v98 = (__int64 *)a1[3];
    v99 = *v98;
    *v98 = v6;
LABEL_86:
    v129 = v131;
    v110 = v99;
    v130 = 0xC00000000LL;
    sub_11B1960((__int64)&v129, v92, -1, v94, v93, v99);
    v123[0] = (unsigned __int64)v129;
    sub_2B097A0(v123, v89);
    v101 = v123[0];
    v102 = (unsigned __int64)v129;
    if ( v129 != (const char *)v123[0] )
    {
      v103 = 0;
      v104 = (v123[0] - 4 - (unsigned __int64)v129) >> 2;
      do
      {
        v105 = v103;
        *(_DWORD *)(v102 + 4 * v103) = v103;
        ++v103;
      }
      while ( v105 != v104 );
      v101 = (__int64)v129;
    }
    v106 = (unsigned int **)a1[1];
    v124 = 257;
    v6 = sub_A83DF0(v106, v110, v101, (unsigned int)v130, v100);
    if ( v129 != v131 )
      _libc_free((unsigned __int64)v129);
    goto LABEL_92;
  }
  if ( v92 != v89 )
  {
    v109 = v89;
    v89 = v92;
    v92 = v109;
    v99 = v6;
    goto LABEL_86;
  }
LABEL_92:
  v107 = (unsigned int **)a1[1];
  v108 = (__int64 *)a1[3];
  v132 = 259;
  v129 = "rdx.op";
  *v108 = sub_A83CB0(v107, (_BYTE *)*v108, (_BYTE *)v6, (__int64)v125, (unsigned int)v126, (__int64)&v129);
  if ( v125 != v127 )
    _libc_free((unsigned __int64)v125);
}
