// Function: sub_2CCCE20
// Address: 0x2ccce20
//
void __fastcall sub_2CCCE20(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 **a4,
        unsigned __int64 a5,
        __int64 **a6,
        __int64 a7,
        unsigned int a8,
        char a9,
        char a10,
        __int64 a11,
        __int64 a12)
{
  _QWORD *v14; // rax
  _QWORD *v15; // rbx
  unsigned __int64 v16; // rax
  int v17; // edx
  __int64 v18; // r10
  unsigned __int64 v19; // rax
  __int64 v20; // r15
  unsigned __int64 v21; // r14
  _BYTE *v22; // rax
  __int64 v23; // r13
  unsigned __int8 *v24; // rax
  unsigned __int8 *v25; // r12
  __int64 v26; // rax
  __int64 v27; // r15
  __int64 v28; // r12
  __int64 v29; // r15
  int v30; // eax
  int v31; // eax
  unsigned int v32; // edx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rbx
  unsigned __int64 v37; // rax
  _QWORD *v38; // rax
  __int64 v39; // r15
  unsigned int *v40; // r14
  unsigned int *v41; // rbx
  __int64 v42; // rdx
  unsigned int v43; // esi
  __int64 v44; // rbx
  __int64 v45; // r9
  _QWORD *v46; // r14
  unsigned int *v47; // r15
  unsigned int *v48; // rbx
  __int64 v49; // rdx
  unsigned int v50; // esi
  _BYTE *v51; // rax
  __int64 v52; // rdx
  int v53; // eax
  int v54; // eax
  unsigned int v55; // ecx
  __int64 v56; // rax
  __int64 v57; // rcx
  __int64 v58; // rcx
  __int64 v59; // rax
  __int64 v60; // r12
  _QWORD *v61; // rax
  __int64 v62; // r9
  __int64 v63; // rbx
  unsigned int *v64; // r13
  unsigned int *v65; // r12
  __int64 v66; // rdx
  unsigned int v67; // esi
  __int64 v68; // rax
  __int64 v69; // rsi
  __int64 v70; // r12
  unsigned __int64 v71; // rax
  _QWORD *v72; // rax
  __int64 v73; // rbx
  __int64 v74; // r8
  __int64 v75; // r9
  unsigned int *v76; // r15
  unsigned int *v77; // r12
  __int64 v78; // rdx
  unsigned int v79; // esi
  __int64 v80; // rax
  unsigned __int64 v81; // rdx
  __int64 v82; // rbx
  __int64 v83; // r14
  _QWORD *v84; // rax
  __int64 v85; // r9
  __int64 v86; // r12
  unsigned int *v87; // r14
  unsigned int *v88; // rbx
  __int64 v89; // rdx
  unsigned int v90; // esi
  __int64 v91; // [rsp-8h] [rbp-238h]
  __int64 v92; // [rsp-8h] [rbp-238h]
  __int64 v93; // [rsp+10h] [rbp-220h]
  __int64 v94; // [rsp+18h] [rbp-218h]
  unsigned __int64 v95; // [rsp+20h] [rbp-210h]
  __int64 v96; // [rsp+28h] [rbp-208h]
  __int64 v97; // [rsp+28h] [rbp-208h]
  __int64 v98; // [rsp+48h] [rbp-1E8h]
  unsigned int v99; // [rsp+48h] [rbp-1E8h]
  __int64 v100; // [rsp+48h] [rbp-1E8h]
  char v102; // [rsp+50h] [rbp-1E0h]
  char v103; // [rsp+50h] [rbp-1E0h]
  _QWORD *v104; // [rsp+58h] [rbp-1D8h]
  __int64 v105; // [rsp+58h] [rbp-1D8h]
  __int64 v108; // [rsp+70h] [rbp-1C0h]
  unsigned int v109; // [rsp+78h] [rbp-1B8h]
  _BYTE *v110[4]; // [rsp+80h] [rbp-1B0h] BYREF
  __int16 v111; // [rsp+A0h] [rbp-190h]
  int v112[8]; // [rsp+B0h] [rbp-180h] BYREF
  __int16 v113; // [rsp+D0h] [rbp-160h]
  unsigned int *v114; // [rsp+E0h] [rbp-150h] BYREF
  int v115; // [rsp+E8h] [rbp-148h]
  char v116; // [rsp+F0h] [rbp-140h] BYREF
  __int64 v117; // [rsp+118h] [rbp-118h]
  __int64 v118; // [rsp+120h] [rbp-110h]
  __int64 v119; // [rsp+138h] [rbp-F8h]
  void *v120; // [rsp+160h] [rbp-D0h]
  char *v121; // [rsp+170h] [rbp-C0h] BYREF
  __int64 v122; // [rsp+178h] [rbp-B8h]
  _BYTE v123[16]; // [rsp+180h] [rbp-B0h] BYREF
  __int16 v124; // [rsp+190h] [rbp-A0h]
  __int64 v125; // [rsp+1A0h] [rbp-90h]
  __int64 v126; // [rsp+1A8h] [rbp-88h]
  __int64 v127; // [rsp+1B0h] [rbp-80h]
  __int64 v128; // [rsp+1B8h] [rbp-78h]
  void **v129; // [rsp+1C0h] [rbp-70h]
  void **v130; // [rsp+1C8h] [rbp-68h]
  __int64 v131; // [rsp+1D0h] [rbp-60h]
  int v132; // [rsp+1D8h] [rbp-58h]
  __int16 v133; // [rsp+1DCh] [rbp-54h]
  char v134; // [rsp+1DEh] [rbp-52h]
  __int64 v135; // [rsp+1E0h] [rbp-50h]
  __int64 v136; // [rsp+1E8h] [rbp-48h]
  void *v137; // [rsp+1F0h] [rbp-40h] BYREF
  void *v138; // [rsp+1F8h] [rbp-38h] BYREF

  if ( *(_BYTE *)a7 != 17 )
    goto LABEL_5;
  v14 = *(_QWORD **)(a7 + 24);
  if ( *(_DWORD *)(a7 + 32) > 0x40u )
    v14 = (_QWORD *)*v14;
  v104 = v14;
  if ( (unsigned int)qword_5013628 >= (unsigned __int64)v14 )
  {
    if ( !v14 )
      return;
    sub_23D0AB0((__int64)&v114, a1, 0, 0, 0);
    v124 = 257;
    v68 = sub_A83570(&v114, a3, (__int64)a4, (__int64)&v121);
    v124 = 257;
    v94 = v68;
    v69 = 0;
    v99 = 0;
    v93 = sub_A83570(&v114, a5, (__int64)a6, (__int64)&v121);
    v97 = *(_QWORD *)(a7 + 8);
    v121 = v123;
    v122 = 0x1000000000LL;
    do
    {
      *(_QWORD *)v112 = "src.memcpy.gep.unroll";
      v113 = 259;
      v110[0] = (_BYTE *)sub_AD64C0(v97, v69, 0);
      v103 = -1;
      v70 = sub_921130(&v114, a2, v94, v110, 1, (__int64)v112, 0);
      v111 = 257;
      if ( a8 )
      {
        _BitScanReverse64(&v71, a8);
        v103 = 63 - (v71 ^ 0x3F);
      }
      v113 = 257;
      v72 = sub_BD2C40(80, 1u);
      v73 = (__int64)v72;
      if ( v72 )
        sub_B4D190((__int64)v72, a2, v70, (__int64)v112, a9, v103, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _BYTE **, __int64, __int64))(*(_QWORD *)v119 + 16LL))(
        v119,
        v73,
        v110,
        v117,
        v118);
      v76 = v114;
      v77 = &v114[4 * v115];
      if ( v114 != v77 )
      {
        do
        {
          v78 = *((_QWORD *)v76 + 1);
          v79 = *v76;
          v76 += 4;
          sub_B99FD0(v73, v79, v78);
        }
        while ( v77 != v76 );
      }
      v80 = (unsigned int)v122;
      v81 = (unsigned int)v122 + 1LL;
      if ( v81 > HIDWORD(v122) )
      {
        sub_C8D5F0((__int64)&v121, v123, v81, 8u, v74, v75);
        v80 = (unsigned int)v122;
      }
      v69 = ++v99;
      *(_QWORD *)&v121[8 * v80] = v73;
      LODWORD(v122) = v122 + 1;
    }
    while ( (_QWORD *)v99 != v104 );
    v82 = 0;
    v109 = 0;
    do
    {
      *(_QWORD *)v112 = "dst.memcpy.gep.unroll";
      v113 = 259;
      v110[0] = (_BYTE *)sub_AD64C0(v97, v82, 0);
      v100 = sub_921130(&v114, a2, v93, v110, 1, (__int64)v112, 0);
      v83 = *(_QWORD *)&v121[8 * v82];
      v113 = 257;
      v84 = sub_BD2C40(80, unk_3F10A10);
      v86 = (__int64)v84;
      if ( v84 )
      {
        sub_B4D3C0((__int64)v84, v83, v100, a10, v103, v85, 0, 0);
        v85 = v92;
      }
      (*(void (__fastcall **)(__int64, __int64, int *, __int64, __int64, __int64))(*(_QWORD *)v119 + 16LL))(
        v119,
        v86,
        v112,
        v117,
        v118,
        v85);
      v87 = v114;
      v88 = &v114[4 * v115];
      if ( v114 != v88 )
      {
        do
        {
          v89 = *((_QWORD *)v87 + 1);
          v90 = *v87;
          v87 += 4;
          sub_B99FD0(v86, v90, v89);
        }
        while ( v88 != v87 );
      }
      v82 = ++v109;
    }
    while ( (_QWORD *)v109 != v104 );
  }
  else
  {
LABEL_5:
    v15 = *(_QWORD **)(a1 + 40);
    v121 = "split";
    v124 = 259;
    v98 = sub_AA8550(v15, (__int64 *)(a1 + 24), 0, (__int64)&v121, 0);
    v121 = "loadstoreloop";
    v124 = 259;
    v105 = sub_22077B0(0x50u);
    if ( v105 )
      sub_AA4D50(v105, a11, (__int64)&v121, a12, v98);
    v16 = v15[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (_QWORD *)v16 == v15 + 6 )
    {
      v20 = 0;
    }
    else
    {
      if ( !v16 )
        BUG();
      v17 = *(unsigned __int8 *)(v16 - 24);
      v18 = 0;
      v19 = v16 - 24;
      if ( (unsigned int)(v17 - 30) < 0xB )
        v18 = v19;
      v20 = v18;
    }
    sub_23D0AB0((__int64)&v114, v20, 0, 0, 0);
    v124 = 257;
    v21 = sub_2CC9490((__int64 *)&v114, 0x31u, a3, a4, (__int64)&v121, 0, v112[0], 0);
    v124 = 257;
    v95 = sub_2CC9490((__int64 *)&v114, 0x31u, a5, a6, (__int64)&v121, 0, v112[0], 0);
    v124 = 257;
    v22 = (_BYTE *)sub_AD64C0(*(_QWORD *)(a7 + 8), 0, 0);
    v23 = sub_92B530(&v114, 0x22u, a7, v22, (__int64)&v121);
    v24 = (unsigned __int8 *)sub_BD2C40(72, 3u);
    v25 = v24;
    if ( v24 )
      sub_B4C9A0((__int64)v24, v105, v98, v23, 3u, 0, 0, 0);
    sub_F34910(v20, v25);
    v26 = sub_AA48A0(v105);
    v125 = v105;
    v128 = v26;
    v129 = &v137;
    v130 = &v138;
    v121 = v123;
    v137 = &unk_49DA100;
    v133 = 512;
    v122 = 0x200000000LL;
    v138 = &unk_49DA0B0;
    v126 = v105 + 48;
    LOWORD(v127) = 0;
    v27 = *(_QWORD *)(a7 + 8);
    v131 = 0;
    v132 = 0;
    v96 = v27;
    v134 = 7;
    v135 = 0;
    v136 = 0;
    v113 = 257;
    v28 = sub_D5C860((__int64 *)&v121, v27, 0, (__int64)v112);
    v29 = sub_AD64C0(v27, 0, 0);
    v30 = *(_DWORD *)(v28 + 4) & 0x7FFFFFF;
    if ( v30 == *(_DWORD *)(v28 + 72) )
    {
      sub_B48D90(v28);
      v30 = *(_DWORD *)(v28 + 4) & 0x7FFFFFF;
    }
    v31 = (v30 + 1) & 0x7FFFFFF;
    v32 = v31 | *(_DWORD *)(v28 + 4) & 0xF8000000;
    v33 = *(_QWORD *)(v28 - 8) + 32LL * (unsigned int)(v31 - 1);
    *(_DWORD *)(v28 + 4) = v32;
    if ( *(_QWORD *)v33 )
    {
      v34 = *(_QWORD *)(v33 + 8);
      **(_QWORD **)(v33 + 16) = v34;
      if ( v34 )
        *(_QWORD *)(v34 + 16) = *(_QWORD *)(v33 + 16);
    }
    *(_QWORD *)v33 = v29;
    if ( v29 )
    {
      v35 = *(_QWORD *)(v29 + 16);
      *(_QWORD *)(v33 + 8) = v35;
      if ( v35 )
        *(_QWORD *)(v35 + 16) = v33 + 8;
      *(_QWORD *)(v33 + 16) = v29 + 16;
      *(_QWORD *)(v29 + 16) = v33;
    }
    *(_QWORD *)(*(_QWORD *)(v28 - 8)
              + 32LL * *(unsigned int *)(v28 + 72)
              + 8LL * ((*(_DWORD *)(v28 + 4) & 0x7FFFFFFu) - 1)) = v15;
    v113 = 257;
    v110[0] = (_BYTE *)v28;
    v102 = -1;
    v36 = sub_921130((unsigned int **)&v121, a2, v21, v110, 1, (__int64)v112, 0);
    v111 = 257;
    if ( a8 )
    {
      _BitScanReverse64(&v37, a8);
      v102 = 63 - (v37 ^ 0x3F);
    }
    v113 = 257;
    v38 = sub_BD2C40(80, 1u);
    v39 = (__int64)v38;
    if ( v38 )
      sub_B4D190((__int64)v38, a2, v36, (__int64)v112, a9, v102, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _BYTE **, __int64, __int64))*v130 + 2))(v130, v39, v110, v126, v127);
    v40 = (unsigned int *)v121;
    v41 = (unsigned int *)&v121[16 * (unsigned int)v122];
    if ( v121 != (char *)v41 )
    {
      do
      {
        v42 = *((_QWORD *)v40 + 1);
        v43 = *v40;
        v40 += 4;
        sub_B99FD0(v39, v43, v42);
      }
      while ( v41 != v40 );
    }
    v113 = 257;
    v110[0] = (_BYTE *)v28;
    v44 = sub_921130((unsigned int **)&v121, a2, v95, v110, 1, (__int64)v112, 0);
    v113 = 257;
    v46 = sub_BD2C40(80, unk_3F10A10);
    if ( v46 )
    {
      sub_B4D3C0((__int64)v46, v39, v44, a10, v102, v45, 0, 0);
      v45 = v91;
    }
    (*((void (__fastcall **)(void **, _QWORD *, int *, __int64, __int64, __int64))*v130 + 2))(
      v130,
      v46,
      v112,
      v126,
      v127,
      v45);
    v47 = (unsigned int *)&v121[16 * (unsigned int)v122];
    v48 = (unsigned int *)v121;
    if ( v121 != (char *)v47 )
    {
      do
      {
        v49 = *((_QWORD *)v48 + 1);
        v50 = *v48;
        v48 += 4;
        sub_B99FD0((__int64)v46, v50, v49);
      }
      while ( v47 != v48 );
    }
    v113 = 257;
    v51 = (_BYTE *)sub_AD64C0(v96, 1, 0);
    v52 = sub_929C50((unsigned int **)&v121, (_BYTE *)v28, v51, (__int64)v112, 0, 0);
    v53 = *(_DWORD *)(v28 + 4) & 0x7FFFFFF;
    if ( v53 == *(_DWORD *)(v28 + 72) )
    {
      v108 = v52;
      sub_B48D90(v28);
      v52 = v108;
      v53 = *(_DWORD *)(v28 + 4) & 0x7FFFFFF;
    }
    v54 = (v53 + 1) & 0x7FFFFFF;
    v55 = v54 | *(_DWORD *)(v28 + 4) & 0xF8000000;
    v56 = *(_QWORD *)(v28 - 8) + 32LL * (unsigned int)(v54 - 1);
    *(_DWORD *)(v28 + 4) = v55;
    if ( *(_QWORD *)v56 )
    {
      v57 = *(_QWORD *)(v56 + 8);
      **(_QWORD **)(v56 + 16) = v57;
      if ( v57 )
        *(_QWORD *)(v57 + 16) = *(_QWORD *)(v56 + 16);
    }
    *(_QWORD *)v56 = v52;
    if ( v52 )
    {
      v58 = *(_QWORD *)(v52 + 16);
      *(_QWORD *)(v56 + 8) = v58;
      if ( v58 )
        *(_QWORD *)(v58 + 16) = v56 + 8;
      *(_QWORD *)(v56 + 16) = v52 + 16;
      *(_QWORD *)(v52 + 16) = v56;
    }
    *(_QWORD *)(*(_QWORD *)(v28 - 8)
              + 32LL * *(unsigned int *)(v28 + 72)
              + 8LL * ((*(_DWORD *)(v28 + 4) & 0x7FFFFFFu) - 1)) = v105;
    v111 = 257;
    v59 = sub_92B530((unsigned int **)&v121, 0x24u, v52, (_BYTE *)a7, (__int64)v110);
    v113 = 257;
    v60 = v59;
    v61 = sub_BD2C40(72, 3u);
    v63 = (__int64)v61;
    if ( v61 )
      sub_B4C9A0((__int64)v61, v105, v98, v60, 3u, v62, 0, 0);
    (*((void (__fastcall **)(void **, __int64, int *, __int64, __int64))*v130 + 2))(v130, v63, v112, v126, v127);
    v64 = (unsigned int *)v121;
    v65 = (unsigned int *)&v121[16 * (unsigned int)v122];
    if ( v121 != (char *)v65 )
    {
      do
      {
        v66 = *((_QWORD *)v64 + 1);
        v67 = *v64;
        v64 += 4;
        sub_B99FD0(v63, v67, v66);
      }
      while ( v65 != v64 );
    }
    nullsub_61();
    v137 = &unk_49DA100;
    nullsub_63();
  }
  if ( v121 != v123 )
    _libc_free((unsigned __int64)v121);
  nullsub_61();
  v120 = &unk_49DA100;
  nullsub_63();
  if ( v114 != (unsigned int *)&v116 )
    _libc_free((unsigned __int64)v114);
}
