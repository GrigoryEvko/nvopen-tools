// Function: sub_293BC00
// Address: 0x293bc00
//
__int64 __fastcall sub_293BC00(__int64 a1, unsigned int a2)
{
  __int64 v2; // r12
  _QWORD *v3; // rbx
  __int64 v4; // r15
  __int64 v6; // r15
  __int64 v8; // r8
  __int64 v9; // rdi
  __int64 v10; // r9
  unsigned int v11; // r15d
  __int64 v12; // rdx
  unsigned __int64 *v13; // rcx
  __int64 v14; // r12
  int v15; // ebx
  __int64 v16; // rdi
  __int128 v17; // rax
  unsigned __int64 *v18; // r14
  __int64 v19; // r15
  __int64 v20; // rax
  _BYTE *v21; // r11
  _BYTE *v22; // r10
  __int64 (__fastcall *v23)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v24; // rax
  __int64 v25; // r13
  _BYTE *v26; // rdi
  __int128 v27; // rax
  __int64 v28; // r15
  _BYTE *v29; // r14
  __int64 v30; // rax
  unsigned __int8 *v31; // r15
  __int64 (__fastcall *v32)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v33; // r13
  __int64 v34; // rax
  __int64 v35; // rax
  int v36; // ecx
  _BYTE *v37; // rdx
  __int128 v38; // rax
  unsigned __int8 *v39; // r15
  __int64 v40; // r14
  __int64 v41; // rax
  __int64 (__fastcall *v42)(__int64, __int64, unsigned __int8 *, _BYTE **, __int64, int); // rax
  _BYTE **v43; // rax
  _BYTE **v44; // rcx
  __int64 v45; // r15
  _QWORD *v46; // rax
  unsigned __int64 v47; // r15
  _BYTE *v48; // r14
  __int64 v49; // rdx
  unsigned int v50; // esi
  _QWORD *v51; // rax
  unsigned __int64 v52; // r15
  _BYTE *v53; // r14
  __int64 v54; // rdx
  unsigned int v55; // esi
  __int64 v56; // rax
  __int64 v57; // r10
  unsigned __int64 v58; // r13
  _BYTE *v59; // r14
  __int64 v60; // rdx
  unsigned int v61; // esi
  __int64 v62; // rdx
  int v63; // eax
  char v64; // al
  int v65; // edx
  _QWORD *v66; // rax
  _BYTE *v67; // [rsp+18h] [rbp-208h]
  __int64 v68; // [rsp+18h] [rbp-208h]
  _BYTE *v69; // [rsp+18h] [rbp-208h]
  __int64 v70; // [rsp+20h] [rbp-200h]
  __int64 v71; // [rsp+20h] [rbp-200h]
  __int64 v72; // [rsp+28h] [rbp-1F8h]
  _QWORD *v73; // [rsp+30h] [rbp-1F0h]
  __int64 v74; // [rsp+30h] [rbp-1F0h]
  __int64 v75; // [rsp+30h] [rbp-1F0h]
  _BYTE *v76; // [rsp+30h] [rbp-1F0h]
  _BYTE *v77; // [rsp+58h] [rbp-1C8h] BYREF
  __int128 v78; // [rsp+60h] [rbp-1C0h] BYREF
  __int128 v79; // [rsp+70h] [rbp-1B0h]
  __int64 v80; // [rsp+80h] [rbp-1A0h]
  __int128 v81; // [rsp+90h] [rbp-190h] BYREF
  __int128 v82; // [rsp+A0h] [rbp-180h]
  __int64 v83; // [rsp+B0h] [rbp-170h]
  _OWORD v84[2]; // [rsp+C0h] [rbp-160h] BYREF
  __int64 v85; // [rsp+E0h] [rbp-140h]
  __int128 v86; // [rsp+F0h] [rbp-130h] BYREF
  __int128 v87; // [rsp+100h] [rbp-120h]
  __int64 v88; // [rsp+110h] [rbp-110h]
  unsigned __int64 *v89; // [rsp+120h] [rbp-100h] BYREF
  __int64 v90; // [rsp+128h] [rbp-F8h]
  unsigned __int64 v91; // [rsp+130h] [rbp-F0h] BYREF
  unsigned int v92; // [rsp+138h] [rbp-E8h]
  __int16 v93; // [rsp+140h] [rbp-E0h]
  _BYTE *v94; // [rsp+160h] [rbp-C0h] BYREF
  __int64 v95; // [rsp+168h] [rbp-B8h]
  _BYTE v96[32]; // [rsp+170h] [rbp-B0h] BYREF
  __int64 v97; // [rsp+190h] [rbp-90h]
  __int64 v98; // [rsp+198h] [rbp-88h]
  __int64 v99; // [rsp+1A0h] [rbp-80h]
  _QWORD *v100; // [rsp+1A8h] [rbp-78h]
  void **v101; // [rsp+1B0h] [rbp-70h]
  void **v102; // [rsp+1B8h] [rbp-68h]
  __int64 v103; // [rsp+1C0h] [rbp-60h]
  int v104; // [rsp+1C8h] [rbp-58h]
  __int16 v105; // [rsp+1CCh] [rbp-54h]
  char v106; // [rsp+1CEh] [rbp-52h]
  __int64 v107; // [rsp+1D0h] [rbp-50h]
  __int64 v108; // [rsp+1D8h] [rbp-48h]
  void *v109; // [rsp+1E0h] [rbp-40h] BYREF
  void *v110; // [rsp+1E8h] [rbp-38h] BYREF

  v2 = a2;
  v3 = *(_QWORD **)(a1 + 72);
  if ( !v3 )
    v3 = (_QWORD *)(a1 + 80);
  v4 = *(_QWORD *)(*v3 + 8LL * a2);
  if ( v4 )
    return v4;
  v6 = *(_QWORD *)a1;
  v70 = *(_QWORD *)(a1 + 8);
  v72 = *(_QWORD *)(a1 + 16);
  v100 = (_QWORD *)sub_AA48A0(*(_QWORD *)a1);
  v101 = &v109;
  v102 = &v110;
  v94 = v96;
  v109 = &unk_49DA100;
  v95 = 0x200000000LL;
  v103 = 0;
  v104 = 0;
  v105 = 512;
  v106 = 7;
  v107 = 0;
  v108 = 0;
  v97 = 0;
  v98 = 0;
  LOWORD(v99) = 0;
  v110 = &unk_49DA0B0;
  sub_A88F30((__int64)&v94, v6, v70, v72);
  if ( *(_BYTE *)(a1 + 64) )
  {
    v9 = *(_QWORD *)(a1 + 24);
    if ( !a2 )
    {
      *(_QWORD *)*v3 = v9;
      goto LABEL_8;
    }
    LODWORD(v84[0]) = a2;
    LOWORD(v85) = 265;
    *(_QWORD *)&v38 = sub_BD5D20(v9);
    v81 = v38;
    *(_QWORD *)&v82 = ".i";
    LOWORD(v83) = 773;
    v87 = v84[0];
    *(_QWORD *)&v86 = &v81;
    LOWORD(v88) = 2306;
    v39 = *(unsigned __int8 **)(a1 + 24);
    v40 = *(_QWORD *)(a1 + 48);
    v71 = (__int64)v39;
    v41 = sub_BCB2D0(v100);
    v77 = (_BYTE *)sub_ACD640(v41, a2, 0);
    v42 = (__int64 (__fastcall *)(__int64, __int64, unsigned __int8 *, _BYTE **, __int64, int))*((_QWORD *)*v101 + 8);
    if ( v42 == sub_920540 )
    {
      if ( sub_BCEA30(v40) )
        goto LABEL_62;
      if ( *v39 > 0x15u )
        goto LABEL_62;
      v43 = sub_293A090(&v77, (__int64)&v78);
      if ( v44 != v43 )
        goto LABEL_62;
      LOBYTE(v93) = 0;
      v45 = sub_AD9FD0(v40, v39, (__int64 *)&v77, 1, 0, (__int64)&v89, 0);
      if ( (_BYTE)v93 )
      {
        LOBYTE(v93) = 0;
        if ( v92 > 0x40 && v91 )
          j_j___libc_free_0_0(v91);
        if ( (unsigned int)v90 > 0x40 && v89 )
          j_j___libc_free_0_0((unsigned __int64)v89);
      }
    }
    else
    {
      v45 = v42((__int64)v101, v40, v39, &v77, 1, 0);
    }
    if ( v45 )
    {
LABEL_50:
      *(_QWORD *)(*v3 + 8 * v2) = v45;
      goto LABEL_8;
    }
LABEL_62:
    v93 = 257;
    v45 = (__int64)sub_BD2C40(88, 2u);
    if ( !v45 )
      goto LABEL_65;
    v57 = *(_QWORD *)(v71 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v57 + 8) - 17 <= 1 )
    {
LABEL_64:
      sub_B44260(v45, v57, 34, 2u, 0, 0);
      *(_QWORD *)(v45 + 72) = v40;
      *(_QWORD *)(v45 + 80) = sub_B4DC50(v40, (__int64)&v77, 1);
      sub_B4D9A0(v45, v71, (__int64 *)&v77, 1, (__int64)&v89);
LABEL_65:
      (*((void (__fastcall **)(void **, __int64, __int128 *, __int64, __int64))*v102 + 2))(v102, v45, &v86, v98, v99);
      v58 = (unsigned __int64)v94;
      v59 = &v94[16 * (unsigned int)v95];
      if ( v94 != v59 )
      {
        do
        {
          v60 = *(_QWORD *)(v58 + 8);
          v61 = *(_DWORD *)v58;
          v58 += 16LL;
          sub_B99FD0(v45, v61, v60);
        }
        while ( v59 != (_BYTE *)v58 );
      }
      goto LABEL_50;
    }
    v62 = *((_QWORD *)v77 + 1);
    v63 = *(unsigned __int8 *)(v62 + 8);
    if ( v63 == 17 )
    {
      v64 = 0;
    }
    else
    {
      if ( v63 != 18 )
        goto LABEL_64;
      v64 = 1;
    }
    v65 = *(_DWORD *)(v62 + 32);
    BYTE4(v78) = v64;
    LODWORD(v78) = v65;
    v57 = sub_BCE1B0((__int64 *)v57, v78);
    goto LABEL_64;
  }
  v10 = *(_QWORD *)(a1 + 56);
  if ( !v10 || a2 != *(_DWORD *)(a1 + 44) - 1 )
    v10 = *(_QWORD *)(a1 + 48);
  if ( *(_BYTE *)(v10 + 8) != 17 )
  {
    v26 = *(_BYTE **)(a1 + 24);
    while ( *v26 == 91 )
    {
      v34 = *((_QWORD *)v26 - 4);
      if ( *(_BYTE *)v34 != 17 )
        break;
      if ( *(_DWORD *)(v34 + 32) <= 0x40u )
        v35 = *(_QWORD *)(v34 + 24);
      else
        v35 = **(_QWORD **)(v34 + 24);
      v36 = *(_DWORD *)(a1 + 40);
      v37 = (_BYTE *)*((_QWORD *)v26 - 12);
      *(_QWORD *)(a1 + 24) = v37;
      if ( a2 * v36 == (_DWORD)v35 )
      {
        *(_QWORD *)(*v3 + 8LL * a2) = *((_QWORD *)v26 - 8);
        v4 = *(_QWORD *)(*v3 + 8LL * a2);
        goto LABEL_9;
      }
      if ( v36 != 1 || (v66 = (_QWORD *)(*v3 + 8LL * (unsigned int)v35), *v66) )
      {
        v26 = v37;
      }
      else
      {
        *v66 = *((_QWORD *)v26 - 8);
        v26 = *(_BYTE **)(a1 + 24);
      }
    }
    LODWORD(v84[0]) = a2;
    LOWORD(v85) = 265;
    *(_QWORD *)&v27 = sub_BD5D20((__int64)v26);
    v81 = v27;
    *(_QWORD *)&v82 = ".i";
    LOWORD(v83) = 773;
    v87 = v84[0];
    *(_QWORD *)&v86 = &v81;
    LOWORD(v88) = 2306;
    v28 = *(_DWORD *)(a1 + 40) * a2;
    v29 = *(_BYTE **)(a1 + 24);
    v30 = sub_BCB2E0(v100);
    v31 = (unsigned __int8 *)sub_ACD640(v30, v28, 0);
    v32 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v101 + 12);
    if ( v32 == sub_948070 )
    {
      if ( *v29 > 0x15u || *v31 > 0x15u )
        goto LABEL_51;
      v33 = sub_AD5840((__int64)v29, v31, 0);
    }
    else
    {
      v33 = v32((__int64)v101, v29, v31);
    }
    if ( v33 )
    {
LABEL_35:
      *(_QWORD *)(*v3 + 8 * v2) = v33;
      goto LABEL_8;
    }
LABEL_51:
    v93 = 257;
    v46 = sub_BD2C40(72, 2u);
    v33 = (__int64)v46;
    if ( v46 )
      sub_B4DE80((__int64)v46, (__int64)v29, (__int64)v31, (__int64)&v89, 0, 0);
    (*((void (__fastcall **)(void **, __int64, __int128 *, __int64, __int64))*v102 + 2))(v102, v33, &v86, v98, v99);
    v47 = (unsigned __int64)v94;
    v48 = &v94[16 * (unsigned int)v95];
    if ( v94 != v48 )
    {
      do
      {
        v49 = *(_QWORD *)(v47 + 8);
        v50 = *(_DWORD *)v47;
        v47 += 16LL;
        sub_B99FD0(v33, v50, v49);
      }
      while ( v48 != (_BYTE *)v47 );
    }
    goto LABEL_35;
  }
  v89 = &v91;
  v90 = 0xC00000000LL;
  if ( *(_DWORD *)(v10 + 32) )
  {
    v73 = v3;
    v11 = 0;
    v12 = 0;
    v13 = &v91;
    v14 = v10;
    v15 = a2 * *(_DWORD *)(a1 + 40);
    while ( 1 )
    {
      *((_DWORD *)v13 + v12) = v15;
      ++v11;
      v12 = (unsigned int)(v90 + 1);
      LODWORD(v90) = v90 + 1;
      if ( v11 >= *(_DWORD *)(v14 + 32) )
        break;
      v15 = v11 + a2 * *(_DWORD *)(a1 + 40);
      if ( v12 + 1 > (unsigned __int64)HIDWORD(v90) )
      {
        sub_C8D5F0((__int64)&v89, &v91, v12 + 1, 4u, v8, v12 + 1);
        v12 = (unsigned int)v90;
      }
      v13 = v89;
    }
    v3 = v73;
    v2 = a2;
  }
  v16 = *(_QWORD *)(a1 + 24);
  LODWORD(v81) = a2;
  LOWORD(v83) = 265;
  *(_QWORD *)&v17 = sub_BD5D20(v16);
  LOWORD(v80) = 773;
  v78 = v17;
  *(_QWORD *)&v79 = ".i";
  v84[1] = v81;
  *(_QWORD *)&v84[0] = &v78;
  LOWORD(v85) = 2306;
  v18 = v89;
  v19 = (unsigned int)v90;
  v20 = sub_ACADE0(*(__int64 ***)(*(_QWORD *)(a1 + 24) + 8LL));
  v21 = *(_BYTE **)(a1 + 24);
  v22 = (_BYTE *)v20;
  v23 = (__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))*((_QWORD *)*v101 + 14);
  if ( v23 == sub_9B6630 )
  {
    if ( *v21 > 0x15u || *v22 > 0x15u )
      goto LABEL_56;
    v67 = v22;
    v74 = *(_QWORD *)(a1 + 24);
    v24 = sub_AD5CE0(v74, (__int64)v22, v18, v19, 0);
    v21 = (_BYTE *)v74;
    v22 = v67;
    v25 = v24;
  }
  else
  {
    v69 = v22;
    v76 = *(_BYTE **)(a1 + 24);
    v56 = ((__int64 (__fastcall *)(void **, _BYTE *, _BYTE *, unsigned __int64 *, __int64))v23)(
            v101,
            v21,
            v22,
            v18,
            v19);
    v22 = v69;
    v21 = v76;
    v25 = v56;
  }
  if ( !v25 )
  {
LABEL_56:
    v68 = (__int64)v21;
    v75 = (__int64)v22;
    LOWORD(v88) = 257;
    v51 = sub_BD2C40(112, unk_3F1FE60);
    v25 = (__int64)v51;
    if ( v51 )
      sub_B4E9E0((__int64)v51, v68, v75, v18, v19, (__int64)&v86, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _OWORD *, __int64, __int64))*v102 + 2))(v102, v25, v84, v98, v99);
    v52 = (unsigned __int64)v94;
    v53 = &v94[16 * (unsigned int)v95];
    if ( v94 != v53 )
    {
      do
      {
        v54 = *(_QWORD *)(v52 + 8);
        v55 = *(_DWORD *)v52;
        v52 += 16LL;
        sub_B99FD0(v25, v55, v54);
      }
      while ( v53 != (_BYTE *)v52 );
    }
  }
  *(_QWORD *)(*v3 + 8 * v2) = v25;
  if ( v89 != &v91 )
    _libc_free((unsigned __int64)v89);
LABEL_8:
  v4 = *(_QWORD *)(*v3 + 8 * v2);
LABEL_9:
  nullsub_61();
  v109 = &unk_49DA100;
  nullsub_63();
  if ( v94 != v96 )
    _libc_free((unsigned __int64)v94);
  return v4;
}
