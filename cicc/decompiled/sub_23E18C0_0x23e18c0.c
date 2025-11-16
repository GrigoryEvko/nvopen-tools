// Function: sub_23E18C0
// Address: 0x23e18c0
//
void __fastcall sub_23E18C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int64 a7,
        char a8,
        char a9,
        char a10,
        __int64 a11,
        char a12,
        char a13,
        char a14,
        __int64 a15)
{
  __int64 v18; // rsi
  int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rbx
  __int64 v26; // r10
  __int64 v27; // rbx
  unsigned int v28; // eax
  int v29; // eax
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // r15
  __int64 v33; // rbx
  unsigned int v34; // eax
  _QWORD *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  _QWORD *v38; // rax
  _QWORD *v39; // r10
  __int64 v40; // rdx
  int v41; // ecx
  int v42; // eax
  _QWORD *v43; // rdi
  __int64 *v44; // rax
  __int64 v45; // rax
  __int64 v46; // rbx
  _BYTE *v47; // rbx
  __int64 v48; // r13
  unsigned __int64 v49; // r12
  __int64 v50; // rdx
  unsigned int v51; // esi
  int v52; // eax
  __int64 v53; // rbx
  _BYTE *v54; // rbx
  unsigned __int64 v55; // r13
  __int64 v56; // rdx
  unsigned int v57; // esi
  __int64 v58; // rax
  _QWORD *v59; // rax
  __int64 v60; // rbx
  unsigned __int64 v61; // r15
  _BYTE *v62; // r13
  __int64 v63; // rdx
  unsigned int v64; // esi
  _QWORD *v65; // rax
  __int64 v66; // rbx
  unsigned __int64 v67; // r13
  _BYTE *v68; // r12
  __int64 v69; // rdx
  unsigned int v70; // esi
  unsigned __int64 v71; // r13
  _BYTE *v72; // rbx
  __int64 v73; // rdx
  unsigned int v74; // esi
  unsigned int v75; // [rsp+8h] [rbp-1A8h]
  unsigned int v76; // [rsp+8h] [rbp-1A8h]
  __int64 v77; // [rsp+8h] [rbp-1A8h]
  __int64 v78; // [rsp+8h] [rbp-1A8h]
  __int64 v79; // [rsp+8h] [rbp-1A8h]
  unsigned __int64 v80; // [rsp+28h] [rbp-188h]
  __int64 v81; // [rsp+28h] [rbp-188h]
  __int64 v82; // [rsp+28h] [rbp-188h]
  __int64 v83; // [rsp+28h] [rbp-188h]
  char v84; // [rsp+30h] [rbp-180h] BYREF
  char v85; // [rsp+34h] [rbp-17Ch] BYREF
  __int64 v86; // [rsp+38h] [rbp-178h] BYREF
  __int64 v87; // [rsp+40h] [rbp-170h] BYREF
  __int64 v88; // [rsp+48h] [rbp-168h] BYREF
  __int64 v89; // [rsp+50h] [rbp-160h] BYREF
  __int64 v90; // [rsp+58h] [rbp-158h] BYREF
  __int64 v91; // [rsp+60h] [rbp-150h] BYREF
  __int64 v92; // [rsp+68h] [rbp-148h]
  __int64 v93; // [rsp+70h] [rbp-140h]
  __int64 v94; // [rsp+78h] [rbp-138h]
  unsigned __int64 v95; // [rsp+80h] [rbp-130h] BYREF
  char v96; // [rsp+88h] [rbp-128h]
  _DWORD v97[8]; // [rsp+90h] [rbp-120h] BYREF
  __int16 v98; // [rsp+B0h] [rbp-100h]
  _QWORD v99[2]; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 (__fastcall *v100)(unsigned __int64 *, const __m128i **, int); // [rsp+D0h] [rbp-E0h]
  __int64 (__fastcall *v101)(); // [rsp+D8h] [rbp-D8h]
  __int16 v102; // [rsp+E0h] [rbp-D0h]
  _BYTE *v103; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v104; // [rsp+F8h] [rbp-B8h]
  _BYTE v105[32]; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v106; // [rsp+120h] [rbp-90h]
  __int64 v107; // [rsp+128h] [rbp-88h]
  __int64 v108; // [rsp+130h] [rbp-80h]
  __int64 v109; // [rsp+138h] [rbp-78h]
  void **v110; // [rsp+140h] [rbp-70h]
  void **v111; // [rsp+148h] [rbp-68h]
  __int64 v112; // [rsp+150h] [rbp-60h]
  int v113; // [rsp+158h] [rbp-58h]
  __int16 v114; // [rsp+15Ch] [rbp-54h]
  char v115; // [rsp+15Eh] [rbp-52h]
  __int64 v116; // [rsp+160h] [rbp-50h]
  __int64 v117; // [rsp+168h] [rbp-48h]
  void *v118; // [rsp+170h] [rbp-40h] BYREF
  void *v119; // [rsp+178h] [rbp-38h] BYREF

  v88 = a1;
  v18 = a11;
  v87 = a4;
  v85 = a12;
  v86 = a6;
  v84 = a13;
  v19 = *(unsigned __int8 *)(a11 + 8);
  v89 = 0;
  v90 = a11;
  if ( (unsigned int)(v19 - 17) <= 1 )
    v18 = **(_QWORD **)(a11 + 16);
  v103 = (_BYTE *)sub_9208B0(a2, v18);
  v104 = v20;
  v95 = (unsigned __int64)(v103 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v96 = v20;
  v21 = sub_AD64C0(a3, 0, 0);
  v22 = a7;
  v91 = v21;
  v23 = sub_BD5C60(a7);
  v115 = 7;
  v109 = v23;
  v110 = &v118;
  v111 = &v119;
  v114 = 512;
  LOWORD(v108) = 0;
  v103 = v105;
  v118 = &unk_49DA100;
  v104 = 0x200000000LL;
  v112 = 0;
  v113 = 0;
  v116 = 0;
  v117 = 0;
  v106 = 0;
  v107 = 0;
  v119 = &unk_49DA0B0;
  sub_D5F1F0((__int64)&v103, v22);
  if ( a5 )
  {
    v24 = *(_QWORD *)(a5 + 8);
    v98 = 257;
    v25 = sub_AD64C0(v24, 0, 0);
    v26 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v110 + 7))(v110, 33, a5, v25);
    if ( !v26 )
    {
      v102 = 257;
      v38 = sub_BD2C40(72, unk_3F10FD0);
      v39 = v38;
      if ( v38 )
      {
        v40 = *(_QWORD *)(a5 + 8);
        v81 = (__int64)v38;
        v41 = *(unsigned __int8 *)(v40 + 8);
        if ( (unsigned int)(v41 - 17) > 1 )
        {
          v45 = sub_BCB2A0(*(_QWORD **)v40);
        }
        else
        {
          v42 = *(_DWORD *)(v40 + 32);
          v43 = *(_QWORD **)v40;
          BYTE4(v94) = (_BYTE)v41 == 18;
          LODWORD(v94) = v42;
          v44 = (__int64 *)sub_BCB2A0(v43);
          v45 = sub_BCE1B0(v44, v94);
        }
        sub_B523C0(v81, v45, 53, 33, a5, v25, (__int64)v99, 0, 0, 0);
        v39 = (_QWORD *)v81;
      }
      v82 = (__int64)v39;
      (*((void (__fastcall **)(void **, _QWORD *, _DWORD *, __int64, __int64))*v111 + 2))(v111, v39, v97, v107, v108);
      v26 = v82;
      v46 = 16LL * (unsigned int)v104;
      if ( v103 != &v103[v46] )
      {
        v83 = a3;
        v47 = &v103[v46];
        v48 = v26;
        v77 = a5;
        v49 = (unsigned __int64)v103;
        do
        {
          v50 = *(_QWORD *)(v49 + 8);
          v51 = *(_DWORD *)v49;
          v49 += 16LL;
          sub_B99FD0(v48, v51, v50);
        }
        while ( v47 != (_BYTE *)v49 );
        v26 = v48;
        a5 = v77;
        a3 = v83;
      }
    }
    v80 = sub_F38250(v26, (__int64 *)(a7 + 24), 0, 0, 0, 0, 0, 0);
    sub_D5F1F0((__int64)&v103, v80);
    v27 = *(_QWORD *)(a5 + 8);
    v98 = 257;
    v75 = sub_BCB060(v27);
    v28 = sub_BCB060(a3);
    if ( v75 < v28 )
    {
      if ( a3 == v27 )
      {
LABEL_8:
        v29 = *(_DWORD *)(v90 + 32);
        BYTE4(v93) = *(_BYTE *)(v90 + 8) == 18;
        LODWORD(v93) = v29;
        v30 = sub_B33F10((__int64)&v103, a3, v93);
        v97[1] = 0;
        v102 = 257;
        v31 = sub_B33C40((__int64)&v103, 0x16Eu, a5, v30, v97[0], (__int64)v99);
        goto LABEL_9;
      }
      v37 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v110 + 15))(v110, 39, a5, a3);
      if ( !v37 )
      {
        v102 = 257;
        v65 = sub_BD2C40(72, unk_3F10A14);
        v66 = (__int64)v65;
        if ( v65 )
          sub_B515B0((__int64)v65, a5, a3, (__int64)v99, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _DWORD *, __int64, __int64))*v111 + 2))(v111, v66, v97, v107, v108);
        if ( v103 != &v103[16 * (unsigned int)v104] )
        {
          v79 = a3;
          v67 = (unsigned __int64)v103;
          v68 = &v103[16 * (unsigned int)v104];
          do
          {
            v69 = *(_QWORD *)(v67 + 8);
            v70 = *(_DWORD *)v67;
            v67 += 16LL;
            sub_B99FD0(v66, v70, v69);
          }
          while ( v68 != (_BYTE *)v67 );
          a3 = v79;
        }
        a5 = v66;
        goto LABEL_8;
      }
    }
    else
    {
      if ( v75 == v28 || a3 == v27 )
        goto LABEL_8;
      v37 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v110 + 15))(v110, 38, a5, a3);
      if ( !v37 )
      {
        v102 = 257;
        a5 = sub_B51D30(38, a5, a3, (__int64)v99, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _DWORD *, __int64, __int64))*v111 + 2))(v111, a5, v97, v107, v108);
        v53 = 16LL * (unsigned int)v104;
        if ( v103 != &v103[v53] )
        {
          v78 = a3;
          v54 = &v103[v53];
          v55 = (unsigned __int64)v103;
          do
          {
            v56 = *(_QWORD *)(v55 + 8);
            v57 = *(_DWORD *)v55;
            v55 += 16LL;
            sub_B99FD0(a5, v57, v56);
          }
          while ( v54 != (_BYTE *)v55 );
          a3 = v78;
        }
        goto LABEL_8;
      }
    }
    a5 = v37;
    goto LABEL_8;
  }
  v80 = a7;
  v52 = *(_DWORD *)(v90 + 32);
  BYTE4(v92) = *(_BYTE *)(v90 + 8) == 18;
  LODWORD(v92) = v52;
  v31 = sub_B33F10((__int64)&v103, a3, v92);
LABEL_9:
  v32 = v86;
  if ( !v86 )
    goto LABEL_14;
  v98 = 257;
  v33 = *(_QWORD *)(v86 + 8);
  v76 = sub_BCB060(v33);
  v34 = sub_BCB060(a3);
  if ( v76 < v34 )
  {
    if ( a3 == v33 )
      goto LABEL_13;
    v58 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v110 + 15))(v110, 39, v86, a3);
    if ( !v58 )
    {
      v102 = 257;
      v59 = sub_BD2C40(72, unk_3F10A14);
      v60 = (__int64)v59;
      if ( v59 )
        sub_B515B0((__int64)v59, v86, a3, (__int64)v99, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _DWORD *, __int64, __int64))*v111 + 2))(v111, v60, v97, v107, v108);
      v61 = (unsigned __int64)v103;
      v62 = &v103[16 * (unsigned int)v104];
      if ( v103 != v62 )
      {
        do
        {
          v63 = *(_QWORD *)(v61 + 8);
          v64 = *(_DWORD *)v61;
          v61 += 16LL;
          sub_B99FD0(v60, v64, v63);
        }
        while ( v62 != (_BYTE *)v61 );
      }
      v32 = v60;
      goto LABEL_13;
    }
  }
  else
  {
    if ( v76 == v34 || a3 == v33 )
      goto LABEL_13;
    v58 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v110 + 15))(v110, 38, v86, a3);
    if ( !v58 )
    {
      v102 = 257;
      v32 = sub_B51D30(38, v86, a3, (__int64)v99, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _DWORD *, __int64, __int64))*v111 + 2))(v111, v32, v97, v107, v108);
      v71 = (unsigned __int64)v103;
      v72 = &v103[16 * (unsigned int)v104];
      if ( v103 != v72 )
      {
        do
        {
          v73 = *(_QWORD *)(v71 + 8);
          v74 = *(_DWORD *)v71;
          v71 += 16LL;
          sub_B99FD0(v32, v74, v73);
        }
        while ( v72 != (_BYTE *)v71 );
      }
      goto LABEL_13;
    }
  }
  v32 = v58;
LABEL_13:
  v86 = v32;
LABEL_14:
  v100 = 0;
  v35 = (_QWORD *)sub_22077B0(0x78u);
  if ( v35 )
  {
    *v35 = &v87;
    v35[1] = &a8;
    v35[2] = &v86;
    v35[3] = &v90;
    v35[4] = &v91;
    v35[5] = &v88;
    v35[7] = &a9;
    v35[8] = &a10;
    v35[9] = &v95;
    v35[10] = &v85;
    v35[11] = &v89;
    v35[12] = &v84;
    v35[13] = &a14;
    v36 = a15;
    v35[6] = &a7;
    v35[14] = v36;
  }
  v99[0] = v35;
  v101 = sub_23E5B80;
  v100 = sub_23DC350;
  sub_F37280(v31, v80 + 24, 0, (__int64)v99);
  if ( v100 )
    v100(v99, (const __m128i **)v99, 3);
  nullsub_61();
  v118 = &unk_49DA100;
  nullsub_63();
  if ( v103 != v105 )
    _libc_free((unsigned __int64)v103);
}
