// Function: sub_33276F0
// Address: 0x33276f0
//
__int64 __fastcall sub_33276F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r11
  unsigned __int16 *v6; // rdx
  bool v7; // zf
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int16 v11; // dx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // r11
  __int128 v17; // kr00_16
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  __int64 v21; // r11
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // rbx
  __int64 v25; // rdi
  unsigned int v26; // ecx
  char v27; // si
  __int64 v28; // r8
  __int64 v29; // r14
  __int64 v30; // r15
  __int64 *v31; // r8
  __int64 v32; // r9
  __int64 v33; // r8
  char v34; // al
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // r14
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r15
  unsigned __int64 v41; // rdx
  __int64 *v42; // rax
  __int64 v43; // rdi
  __int64 v44; // r10
  __int64 v45; // rdx
  __int64 v46; // r11
  unsigned __int64 v47; // rdx
  __int64 v48; // rdi
  unsigned int v49; // esi
  char v50; // r9
  __int64 v51; // rcx
  _QWORD *v52; // rcx
  __int64 v53; // r14
  __int64 v54; // r15
  __int64 v55; // rax
  __int64 v56; // rcx
  char v57; // al
  __int64 v58; // rdx
  int v59; // r9d
  __int64 v60; // rdi
  int v61; // r8d
  __int64 v62; // r12
  __int64 v64; // rsi
  __int64 v65; // rsi
  unsigned int v66; // ecx
  int v67; // edx
  __int16 v68; // ax
  __int64 v69; // rdx
  unsigned __int16 v70; // bx
  __int64 v71; // rax
  unsigned __int16 v72; // dx
  __int64 v73; // rax
  __int64 v74; // rax
  unsigned __int64 v75; // r12
  __int64 v76; // rdx
  char v77; // r13
  __int64 v78; // rax
  __int64 v79; // rcx
  __int64 v80; // rdx
  __int64 v81; // rax
  unsigned __int64 v82; // rdx
  __int128 v83; // [rsp-10h] [rbp-260h]
  __int64 v84; // [rsp+10h] [rbp-240h]
  __int64 v85; // [rsp+18h] [rbp-238h]
  __int64 v86; // [rsp+20h] [rbp-230h]
  unsigned int v87; // [rsp+2Ch] [rbp-224h]
  __int64 v88; // [rsp+30h] [rbp-220h]
  __int64 v89; // [rsp+38h] [rbp-218h]
  int v90; // [rsp+40h] [rbp-210h]
  __int64 v91; // [rsp+40h] [rbp-210h]
  int v92; // [rsp+48h] [rbp-208h]
  __int64 v93; // [rsp+48h] [rbp-208h]
  __int64 v94; // [rsp+58h] [rbp-1F8h]
  __int64 v95; // [rsp+88h] [rbp-1C8h]
  char v96; // [rsp+92h] [rbp-1BEh]
  bool v97; // [rsp+93h] [rbp-1BDh]
  int v98; // [rsp+94h] [rbp-1BCh]
  __int64 v100; // [rsp+A8h] [rbp-1A8h]
  __int64 v101; // [rsp+A8h] [rbp-1A8h]
  __int64 v102; // [rsp+A8h] [rbp-1A8h]
  unsigned int v103; // [rsp+A8h] [rbp-1A8h]
  __int64 v104; // [rsp+A8h] [rbp-1A8h]
  unsigned int v105; // [rsp+C0h] [rbp-190h] BYREF
  __int64 v106; // [rsp+C8h] [rbp-188h]
  __int64 v107; // [rsp+D0h] [rbp-180h] BYREF
  __int64 v108; // [rsp+D8h] [rbp-178h]
  __int64 v109; // [rsp+E0h] [rbp-170h] BYREF
  int v110; // [rsp+E8h] [rbp-168h]
  __int64 v111; // [rsp+F0h] [rbp-160h]
  __int64 v112; // [rsp+F8h] [rbp-158h]
  __int64 v113; // [rsp+100h] [rbp-150h]
  __int64 v114; // [rsp+108h] [rbp-148h]
  __int64 v115; // [rsp+110h] [rbp-140h]
  __int64 v116; // [rsp+118h] [rbp-138h]
  __int64 v117; // [rsp+120h] [rbp-130h]
  __int64 v118; // [rsp+128h] [rbp-128h]
  __int128 v119; // [rsp+130h] [rbp-120h] BYREF
  __int64 v120; // [rsp+140h] [rbp-110h]
  __int128 v121; // [rsp+150h] [rbp-100h]
  __int64 v122; // [rsp+160h] [rbp-F0h]
  __int64 v123; // [rsp+170h] [rbp-E0h] BYREF
  __int64 v124; // [rsp+178h] [rbp-D8h]
  __int64 v125; // [rsp+180h] [rbp-D0h]
  __int64 v126; // [rsp+188h] [rbp-C8h]
  _BYTE *v127; // [rsp+190h] [rbp-C0h] BYREF
  __int64 v128; // [rsp+198h] [rbp-B8h]
  _BYTE v129[176]; // [rsp+1A0h] [rbp-B0h] BYREF

  v5 = a2;
  v6 = *(unsigned __int16 **)(a2 + 48);
  v7 = *(_DWORD *)(a2 + 24) == 156;
  v8 = *v6;
  v9 = *((_QWORD *)v6 + 1);
  LOWORD(v105) = v8;
  v106 = v9;
  if ( v7 )
  {
    if ( (_WORD)v8 )
    {
      v68 = word_4456580[v8 - 1];
      v69 = 0;
    }
    else
    {
      v68 = sub_3009970((__int64)&v105, a2, v9, a4, a5);
      v5 = a2;
    }
    LOWORD(v107) = v68;
    v108 = v69;
  }
  else
  {
    v10 = *(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL);
    v11 = *(_WORD *)v10;
    v12 = *(_QWORD *)(v10 + 8);
    LOWORD(v107) = v11;
    v108 = v12;
  }
  v13 = *(_QWORD *)(v5 + 80);
  v109 = v13;
  if ( v13 )
  {
    v100 = v5;
    sub_B96E90((__int64)&v109, v13, 1);
    v5 = v100;
  }
  v101 = v5;
  v14 = *(_QWORD *)(a1 + 16);
  v110 = *(_DWORD *)(v5 + 72);
  v84 = sub_33EDFE0(v14, v105, v106, 1);
  v85 = v15;
  sub_2EAC300((__int64)&v119, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 40LL), *(_DWORD *)(v84 + 96), 0);
  v16 = v101;
  v17 = v119;
  v87 = v120;
  v96 = BYTE4(v120);
  v127 = v129;
  v128 = 0x800000000LL;
  if ( (_WORD)v107 )
  {
    if ( (_WORD)v107 == 1 || (unsigned __int16)(v107 - 504) <= 7u )
      goto LABEL_62;
    v19 = 16LL * ((unsigned __int16)v107 - 1);
    v18 = *(_QWORD *)&byte_444C4A0[v19];
    LOBYTE(v19) = byte_444C4A0[v19 + 8];
  }
  else
  {
    v18 = sub_3007260((__int64)&v107);
    v16 = v101;
    v113 = v18;
    v114 = v19;
  }
  v123 = v18;
  v102 = v16;
  LOBYTE(v124) = v19;
  v20 = sub_CA1930(&v123);
  v21 = v102;
  v97 = 0;
  v98 = v20 >> 3;
  if ( *(_DWORD *)(v102 + 24) != 156 )
    goto LABEL_8;
  v70 = v107;
  v71 = *(_QWORD *)(**(_QWORD **)(v102 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v102 + 40) + 8LL);
  v72 = *(_WORD *)v71;
  v73 = *(_QWORD *)(v71 + 8);
  if ( v72 == (_WORD)v107 )
  {
    if ( v72 || v73 == v108 )
      goto LABEL_8;
    v124 = v73;
    LOWORD(v123) = 0;
    goto LABEL_46;
  }
  LOWORD(v123) = v72;
  v124 = v73;
  if ( !v72 )
  {
LABEL_46:
    v74 = sub_3007260((__int64)&v123);
    v21 = v102;
    v117 = v74;
    v75 = v74;
    v118 = v76;
    v77 = v76;
    goto LABEL_47;
  }
  if ( v72 == 1 || (unsigned __int16)(v72 - 504) <= 7u )
    goto LABEL_62;
  v75 = *(_QWORD *)&byte_444C4A0[16 * v72 - 16];
  v77 = byte_444C4A0[16 * v72 - 8];
LABEL_47:
  if ( v70 )
  {
    if ( v70 != 1 && (unsigned __int16)(v70 - 504) > 7u )
    {
      v82 = *(_QWORD *)&byte_444C4A0[16 * v70 - 16];
      LOBYTE(v81) = byte_444C4A0[16 * v70 - 8];
      goto LABEL_49;
    }
LABEL_62:
    BUG();
  }
  v104 = v21;
  v78 = sub_3007260((__int64)&v107);
  v21 = v104;
  v79 = v78;
  v81 = v80;
  v115 = v79;
  v82 = v79;
  v116 = v81;
LABEL_49:
  if ( !(_BYTE)v81 || (v97 = v77) )
    v97 = v75 > v82;
LABEL_8:
  v22 = *(unsigned int *)(v21 + 64);
  if ( (_DWORD)v22 )
  {
    v23 = v21;
    v24 = 0;
    v103 = 0;
    v95 = 40 * v22;
    v86 = ((__int64)v17 >> 2) & 1;
    do
    {
      if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v23 + 40) + v24) + 24LL) == 51 )
        goto LABEL_14;
      LOBYTE(v112) = 0;
      v43 = *(_QWORD *)(a1 + 16);
      v111 = v103;
      v44 = sub_3409320(v43, v84, v85, v103, v112, (unsigned int)&v109, 0);
      v46 = v45;
      v47 = v17 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v97 )
      {
        v123 = 0;
        v25 = *(_QWORD *)(a1 + 16);
        v124 = 0;
        v125 = 0;
        v126 = 0;
        if ( v47 )
        {
          if ( v86 )
          {
            v26 = *(_DWORD *)(v47 + 12);
            v27 = v96;
            v47 |= 4u;
          }
          else
          {
            v65 = *(_QWORD *)(v47 + 8);
            if ( (unsigned int)*(unsigned __int8 *)(v65 + 8) - 17 <= 1 )
              v65 = **(_QWORD **)(v65 + 16);
            v66 = *(_DWORD *)(v65 + 8);
            v27 = v96;
            v26 = v66 >> 8;
          }
        }
        else
        {
          v26 = v87;
          v27 = 0;
        }
        v28 = *(_QWORD *)(v23 + 40);
        v29 = v107;
        *(_QWORD *)&v121 = v47;
        v30 = v108;
        BYTE4(v122) = v27;
        v31 = (__int64 *)(v24 + v28);
        v88 = v44;
        v32 = v31[1];
        v33 = *v31;
        v89 = v46;
        LODWORD(v122) = v26;
        v90 = v33;
        v92 = v32;
        *((_QWORD *)&v121 + 1) = v103 + *((_QWORD *)&v17 + 1);
        v34 = sub_33CC4A0(v25, v107, v108);
        v37 = sub_33F5040(
                v25,
                (int)v25 + 288,
                0,
                (unsigned int)&v109,
                v90,
                v92,
                v88,
                v89,
                v121,
                v122,
                v29,
                v30,
                v34,
                0,
                (__int64)&v123);
        v38 = (unsigned int)v128;
        v40 = v39;
        v41 = (unsigned int)v128 + 1LL;
        if ( v41 <= HIDWORD(v128) )
          goto LABEL_13;
      }
      else
      {
        v123 = 0;
        v124 = 0;
        v125 = 0;
        v48 = *(_QWORD *)(a1 + 16);
        v126 = 0;
        if ( v47 )
        {
          if ( v86 )
          {
            v49 = *(_DWORD *)(v47 + 12);
            v50 = v96;
            v47 |= 4u;
          }
          else
          {
            v64 = *(_QWORD *)(v47 + 8);
            if ( (unsigned int)*(unsigned __int8 *)(v64 + 8) - 17 <= 1 )
              v64 = **(_QWORD **)(v64 + 16);
            v50 = v96;
            v49 = *(_DWORD *)(v64 + 8) >> 8;
          }
        }
        else
        {
          v49 = v87;
          v50 = 0;
        }
        v51 = *(_QWORD *)(v23 + 40);
        *((_QWORD *)&v121 + 1) = v103 + *((_QWORD *)&v17 + 1);
        *(_QWORD *)&v121 = v47;
        v52 = (_QWORD *)(v24 + v51);
        LODWORD(v122) = v49;
        v91 = v44;
        v53 = *v52;
        v54 = v52[1];
        v93 = v46;
        v55 = *(_QWORD *)(*v52 + 48LL) + 16LL * *((unsigned int *)v52 + 2);
        v56 = v94;
        BYTE4(v122) = v50;
        LOWORD(v56) = *(_WORD *)v55;
        v94 = v56;
        v57 = sub_33CC4A0(v48, (unsigned int)v56, *(_QWORD *)(v55 + 8));
        v37 = sub_33F4560(
                v48,
                (int)v48 + 288,
                0,
                (unsigned int)&v109,
                v53,
                v54,
                v91,
                v93,
                v121,
                v122,
                v57,
                0,
                (__int64)&v123);
        v38 = (unsigned int)v128;
        v40 = v58;
        v41 = (unsigned int)v128 + 1LL;
        if ( v41 <= HIDWORD(v128) )
          goto LABEL_13;
      }
      sub_C8D5F0((__int64)&v127, v129, v41, 0x10u, v35, v36);
      v38 = (unsigned int)v128;
LABEL_13:
      v42 = (__int64 *)&v127[16 * v38];
      *v42 = v37;
      v42[1] = v40;
      LODWORD(v128) = v128 + 1;
LABEL_14:
      v24 += 40;
      v103 += v98;
    }
    while ( v24 != v95 );
  }
  v59 = 0;
  v60 = *(_QWORD *)(a1 + 16);
  v61 = v60 + 288;
  if ( (_DWORD)v128 )
  {
    *((_QWORD *)&v83 + 1) = (unsigned int)v128;
    *(_QWORD *)&v83 = v127;
    v61 = sub_33FC220(v60, 2, (unsigned int)&v109, 1, 0, 0, v83);
    v59 = v67;
    v60 = *(_QWORD *)(a1 + 16);
  }
  v123 = 0;
  v124 = 0;
  v125 = 0;
  v126 = 0;
  v62 = sub_33F1F00(v60, v105, v106, (unsigned int)&v109, v61, v59, v84, v85, v119, v120, 0, 0, (__int64)&v123, 0);
  if ( v127 != v129 )
    _libc_free((unsigned __int64)v127);
  if ( v109 )
    sub_B91220((__int64)&v109, v109);
  return v62;
}
