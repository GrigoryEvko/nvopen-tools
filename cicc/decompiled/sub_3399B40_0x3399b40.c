// Function: sub_3399B40
// Address: 0x3399b40
//
void __fastcall sub_3399B40(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // r15
  unsigned int v4; // eax
  unsigned int v5; // ebx
  __int64 v6; // rax
  __int64 *v7; // rdi
  __int64 v8; // r14
  int v9; // eax
  int v10; // eax
  __int64 v11; // r8
  __int64 v12; // r9
  _OWORD *v13; // rax
  __int64 v14; // r10
  _OWORD *v15; // rdx
  __int64 v16; // rax
  int v17; // edx
  __int64 v18; // r13
  __int64 v19; // rbx
  __int64 v20; // r14
  int v21; // r15d
  _OWORD *v22; // rax
  int v23; // edx
  __int64 v24; // rcx
  __int64 v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // rax
  __int64 v29; // r12
  unsigned int v30; // r13d
  _OWORD *v31; // rax
  int v32; // edx
  __int64 v33; // rcx
  __int64 v34; // r14
  __int64 v35; // rdi
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // rax
  __int64 v39; // r13
  int v40; // eax
  int v41; // r9d
  _OWORD *v42; // r10
  int v43; // r15d
  __int64 v44; // rax
  int v45; // edx
  int v46; // ebx
  __int64 v47; // r11
  bool v48; // zf
  __int64 v49; // rsi
  __int64 v50; // rbx
  int v51; // edx
  int v52; // r13d
  _QWORD *v53; // rax
  unsigned __int64 v54; // rdi
  __int64 v55; // rdi
  __int64 v56; // rbx
  int v57; // edx
  int v58; // r13d
  _QWORD *v59; // rax
  int v60; // edx
  int v61; // r9d
  unsigned int v62; // r15d
  unsigned int v63; // r14d
  int v64; // r15d
  _OWORD *v65; // rbx
  __int64 v66; // rax
  int v67; // edx
  __int64 v68; // rbx
  __int64 v69; // rdi
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r14
  int v73; // edx
  __int128 v74; // [rsp-20h] [rbp-1F0h]
  __int128 v75; // [rsp-10h] [rbp-1E0h]
  __int128 v76; // [rsp-10h] [rbp-1E0h]
  unsigned __int8 v77; // [rsp+0h] [rbp-1D0h]
  __int64 v78; // [rsp+8h] [rbp-1C8h]
  int v79; // [rsp+8h] [rbp-1C8h]
  unsigned int v80; // [rsp+10h] [rbp-1C0h]
  int v81; // [rsp+14h] [rbp-1BCh]
  int v82; // [rsp+18h] [rbp-1B8h]
  __int64 v83; // [rsp+18h] [rbp-1B8h]
  unsigned __int8 *v84; // [rsp+20h] [rbp-1B0h]
  __int64 v85; // [rsp+20h] [rbp-1B0h]
  int v86; // [rsp+20h] [rbp-1B0h]
  unsigned int v88; // [rsp+48h] [rbp-188h]
  __int64 v89; // [rsp+48h] [rbp-188h]
  unsigned __int8 v90; // [rsp+50h] [rbp-180h]
  int v91; // [rsp+50h] [rbp-180h]
  __int64 v92; // [rsp+50h] [rbp-180h]
  __int64 v93; // [rsp+50h] [rbp-180h]
  __int64 v94; // [rsp+58h] [rbp-178h]
  int v95; // [rsp+58h] [rbp-178h]
  unsigned int v96; // [rsp+60h] [rbp-170h]
  _OWORD *v97; // [rsp+60h] [rbp-170h]
  __int64 v98; // [rsp+68h] [rbp-168h]
  __int64 v99; // [rsp+98h] [rbp-138h] BYREF
  __int64 v100; // [rsp+A0h] [rbp-130h] BYREF
  int v101; // [rsp+A8h] [rbp-128h]
  _BYTE *v102; // [rsp+B0h] [rbp-120h] BYREF
  __int64 v103; // [rsp+B8h] [rbp-118h]
  _BYTE v104[64]; // [rsp+C0h] [rbp-110h] BYREF
  _BYTE *v105; // [rsp+100h] [rbp-D0h] BYREF
  __int64 v106; // [rsp+108h] [rbp-C8h]
  _BYTE v107[64]; // [rsp+110h] [rbp-C0h] BYREF
  _OWORD *v108; // [rsp+150h] [rbp-80h] BYREF
  __int64 v109; // [rsp+158h] [rbp-78h]
  _OWORD v110[7]; // [rsp+160h] [rbp-70h] BYREF

  v2 = a1;
  v3 = *(_QWORD *)(a2 + 8);
  v84 = *(unsigned __int8 **)(a2 - 64);
  v78 = *(_QWORD *)(a2 - 32);
  v90 = *v84;
  v94 = *(_QWORD *)(v78 + 8);
  v77 = *(_BYTE *)v78;
  v4 = sub_34B8B90(v3, *(_QWORD *)(a2 + 72), *(_QWORD *)(a2 + 72) + 4LL * *(unsigned int *)(a2 + 80), 0);
  v103 = 0x400000000LL;
  v5 = v4;
  v6 = *(_QWORD *)(a1 + 864);
  v7 = *(__int64 **)(v6 + 40);
  v8 = *(_QWORD *)(v6 + 16);
  v102 = v104;
  v9 = sub_2E79000(v7);
  LOBYTE(v109) = 0;
  *((_QWORD *)&v75 + 1) = v109;
  v108 = 0;
  *(_QWORD *)&v75 = 0;
  sub_34B8C80(v8, v9, v3, (unsigned int)&v102, 0, 0, v75);
  v106 = 0x400000000LL;
  v105 = v107;
  v10 = sub_2E79000(*(__int64 **)(*(_QWORD *)(v2 + 864) + 40LL));
  LOBYTE(v109) = 0;
  *((_QWORD *)&v74 + 1) = v109;
  v108 = 0;
  *(_QWORD *)&v74 = 0;
  sub_34B8C80(v8, v10, v94, (unsigned int)&v105, 0, 0, v74);
  v109 = 0x400000000LL;
  v88 = v106;
  v13 = v110;
  v95 = v103;
  v108 = v110;
  if ( !(_DWORD)v103 )
  {
    v55 = *(_QWORD *)(v2 + 864);
    v100 = 0;
    v101 = 0;
    v56 = sub_33F17F0(v55, 51, &v100, 1, 0);
    v58 = v57;
    if ( v100 )
      sub_B91220((__int64)&v100, v100);
    v100 = a2;
    v59 = sub_337DC20(v2 + 8, &v100);
    *v59 = v56;
    *((_DWORD *)v59 + 2) = v58;
    v54 = (unsigned __int64)v108;
    if ( v108 != v110 )
      goto LABEL_28;
    goto LABEL_29;
  }
  v14 = (__int64)v84;
  if ( (unsigned int)v103 > 4uLL )
  {
    v72 = (unsigned int)v103;
    sub_C8D5F0((__int64)&v108, v110, (unsigned int)v103, 0x10u, v11, v12);
    v14 = (__int64)v84;
    v15 = &v108[v72];
    v13 = &v108[(unsigned int)v109];
    if ( &v108[v72] == v13 )
      goto LABEL_7;
  }
  else
  {
    v15 = &v110[(unsigned int)v103];
    if ( v15 == v110 )
      goto LABEL_7;
  }
  do
  {
    if ( v13 )
    {
      *(_QWORD *)v13 = 0;
      *((_DWORD *)v13 + 2) = 0;
    }
    ++v13;
  }
  while ( v15 != v13 );
LABEL_7:
  v96 = v90 - 12;
  LODWORD(v109) = v95;
  v16 = sub_338B750(v2, v14);
  v81 = v17;
  v18 = v16;
  if ( v5 )
  {
    v80 = v5;
    v91 = v5 + v17;
    v19 = v16;
    v20 = 0;
    v21 = v17;
    do
    {
      v23 = v21;
      v24 = v19;
      if ( v96 <= 1 )
      {
        v25 = *(_QWORD *)(v2 + 864);
        v26 = *(_QWORD *)&v102[v20 * 16];
        v27 = *(_QWORD *)&v102[v20 * 16 + 8];
        v100 = 0;
        v101 = 0;
        v28 = sub_33F17F0(v25, 51, &v100, v26, v27);
        v24 = v28;
        if ( v100 )
        {
          v82 = v23;
          v85 = v28;
          sub_B91220((__int64)&v100, v100);
          v23 = v82;
          v24 = v85;
        }
      }
      ++v21;
      v22 = &v108[v20++];
      *(_QWORD *)v22 = v24;
      *((_DWORD *)v22 + 2) = v23;
    }
    while ( v21 != v91 );
    v18 = v19;
    v5 = v80;
    if ( !v88 )
    {
LABEL_14:
      if ( v95 == v5 )
        goto LABEL_21;
      goto LABEL_15;
    }
    v88 += v80;
    v93 = sub_338B750(v2, v78);
    v61 = v60;
    if ( v80 == v88 )
    {
LABEL_47:
      v5 = v88;
      goto LABEL_14;
    }
    v62 = v80;
LABEL_40:
    v63 = v62;
    v64 = v61 - v5;
    do
    {
      v66 = v93;
      v67 = v64 + v63;
      v68 = v63;
      if ( (unsigned int)v77 - 12 <= 1 )
      {
        v69 = *(_QWORD *)(v2 + 864);
        v70 = *(_QWORD *)&v102[16 * v63];
        v71 = *(_QWORD *)&v102[v68 * 16 + 8];
        v100 = 0;
        v101 = 0;
        v66 = sub_33F17F0(v69, 51, &v100, v70, v71);
        if ( v100 )
        {
          v79 = v67;
          v83 = v66;
          sub_B91220((__int64)&v100, v100);
          v67 = v79;
          v66 = v83;
        }
      }
      v65 = &v108[v68];
      ++v63;
      *(_QWORD *)v65 = v66;
      *((_DWORD *)v65 + 2) = v67;
    }
    while ( v63 != v88 );
    goto LABEL_47;
  }
  if ( v88 )
  {
    v62 = 0;
    v93 = sub_338B750(v2, v78);
    v61 = v73;
    goto LABEL_40;
  }
LABEL_15:
  v92 = v2;
  v29 = v18;
  v30 = v5;
  do
  {
    v32 = v81 + v30;
    v33 = v29;
    v34 = v30;
    if ( v96 <= 1 )
    {
      v35 = *(_QWORD *)(v92 + 864);
      v36 = *(_QWORD *)&v102[16 * v30];
      v37 = *(_QWORD *)&v102[v34 * 16 + 8];
      v100 = 0;
      v101 = 0;
      v38 = sub_33F17F0(v35, 51, &v100, v36, v37);
      v33 = v38;
      if ( v100 )
      {
        v86 = v32;
        v89 = v38;
        sub_B91220((__int64)&v100, v100);
        v32 = v86;
        v33 = v89;
      }
    }
    ++v30;
    v31 = &v108[v34];
    *(_QWORD *)v31 = v33;
    *((_DWORD *)v31 + 2) = v32;
  }
  while ( v95 != v30 );
  v2 = v92;
LABEL_21:
  v39 = *(_QWORD *)(v2 + 864);
  v97 = v108;
  v98 = (unsigned int)v109;
  v40 = sub_33E5830(v39, v102);
  v42 = v97;
  v100 = 0;
  v43 = v40;
  v44 = *(_QWORD *)v2;
  v46 = v45;
  v47 = v98;
  v48 = *(_QWORD *)v2 == 0;
  v101 = *(_DWORD *)(v2 + 848);
  if ( !v48 && &v100 != (__int64 *)(v44 + 48) )
  {
    v49 = *(_QWORD *)(v44 + 48);
    v100 = v49;
    if ( v49 )
    {
      sub_B96E90((__int64)&v100, v49, 1);
      v42 = v97;
      v47 = v98;
    }
  }
  *((_QWORD *)&v76 + 1) = v47;
  *(_QWORD *)&v76 = v42;
  v50 = sub_3411630(v39, 55, (unsigned int)&v100, v43, v46, v41, v76);
  v52 = v51;
  v99 = a2;
  v53 = sub_337DC20(v2 + 8, &v99);
  *v53 = v50;
  *((_DWORD *)v53 + 2) = v52;
  if ( v100 )
    sub_B91220((__int64)&v100, v100);
  v54 = (unsigned __int64)v108;
  if ( v108 != v110 )
LABEL_28:
    _libc_free(v54);
LABEL_29:
  if ( v105 != v107 )
    _libc_free((unsigned __int64)v105);
  if ( v102 != v104 )
    _libc_free((unsigned __int64)v102);
}
