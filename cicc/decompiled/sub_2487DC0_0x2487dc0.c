// Function: sub_2487DC0
// Address: 0x2487dc0
//
void __fastcall sub_2487DC0(__int64 **a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v7; // r10
  __int64 v8; // r9
  _BYTE *v9; // r14
  _BYTE *v10; // r12
  __int64 v11; // rdx
  unsigned int v12; // esi
  __int64 *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // r14
  __int64 v17; // r15
  __int64 v18; // rbx
  __int64 v19; // r14
  __int64 v20; // rbx
  __int64 v21; // r15
  __int64 v22; // r14
  __int64 v23; // rax
  char v24; // bl
  _QWORD *v25; // rax
  __int64 v26; // r12
  _BYTE *v27; // r15
  _BYTE *v28; // rbx
  __int64 v29; // rdx
  unsigned int v30; // esi
  __int64 v31; // rax
  __int64 v32; // rbx
  __int64 v33; // r13
  __int64 v34; // rax
  char v35; // r12
  _QWORD *v36; // rax
  __int64 v37; // r9
  __int64 v38; // rbx
  _BYTE *v39; // r13
  _BYTE *v40; // r12
  __int64 v41; // rdx
  unsigned int v42; // esi
  unsigned __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rbx
  _QWORD *v47; // r13
  unsigned __int64 v48; // rax
  __int64 v49; // r9
  _BYTE *v50; // r14
  _BYTE *v51; // r12
  __int64 v52; // rdx
  unsigned int v53; // esi
  _BYTE *v54; // r14
  _BYTE *v55; // rbx
  __int64 v56; // rdx
  unsigned int v57; // esi
  _BYTE *v58; // r12
  _BYTE *v59; // rbx
  __int64 v60; // rdx
  unsigned int v61; // esi
  _BYTE *v62; // r14
  _BYTE *v63; // rbx
  __int64 v64; // rdx
  unsigned int v65; // esi
  __int64 v66; // rdx
  int v67; // ecx
  int v68; // eax
  _QWORD *v69; // rdi
  __int64 *v70; // rax
  __int64 v71; // rsi
  _BYTE *v72; // r15
  _BYTE *i; // rbx
  __int64 v74; // rdx
  unsigned int v75; // esi
  int v76; // r12d
  _BYTE *v77; // r12
  _BYTE *v78; // rbx
  __int64 v79; // rdx
  unsigned int v80; // esi
  _BYTE *v81; // r15
  _BYTE *v82; // rbx
  __int64 v83; // rdx
  unsigned int v84; // esi
  __int64 v85; // [rsp+0h] [rbp-170h]
  __int64 *v86; // [rsp+8h] [rbp-168h]
  __int64 v87; // [rsp+28h] [rbp-148h]
  __int64 v88; // [rsp+40h] [rbp-130h] BYREF
  __int64 v89; // [rsp+48h] [rbp-128h]
  _BYTE v90[32]; // [rsp+50h] [rbp-120h] BYREF
  __int16 v91; // [rsp+70h] [rbp-100h]
  _BYTE v92[32]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v93; // [rsp+A0h] [rbp-D0h]
  _BYTE *v94; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v95; // [rsp+B8h] [rbp-B8h]
  _BYTE v96[32]; // [rsp+C0h] [rbp-B0h] BYREF
  __int64 v97; // [rsp+E0h] [rbp-90h]
  __int64 v98; // [rsp+E8h] [rbp-88h]
  __int64 v99; // [rsp+F0h] [rbp-80h]
  __int64 v100; // [rsp+F8h] [rbp-78h]
  void **v101; // [rsp+100h] [rbp-70h]
  void **v102; // [rsp+108h] [rbp-68h]
  __int64 v103; // [rsp+110h] [rbp-60h]
  int v104; // [rsp+118h] [rbp-58h]
  __int16 v105; // [rsp+11Ch] [rbp-54h]
  char v106; // [rsp+11Eh] [rbp-52h]
  __int64 v107; // [rsp+120h] [rbp-50h]
  __int64 v108; // [rsp+128h] [rbp-48h]
  void *v109; // [rsp+130h] [rbp-40h] BYREF
  void *v110; // [rsp+138h] [rbp-38h] BYREF

  v85 = a2;
  v100 = sub_BD5C60(a2);
  v101 = &v109;
  v102 = &v110;
  v105 = 512;
  LOWORD(v99) = 0;
  v94 = v96;
  v109 = &unk_49DA100;
  v86 = (__int64 *)&v94;
  v95 = 0x200000000LL;
  v110 = &unk_49DA0B0;
  v103 = 0;
  v104 = 0;
  v106 = 7;
  v107 = 0;
  v108 = 0;
  v97 = 0;
  v98 = 0;
  sub_D5F1F0((__int64)&v94, a2);
  v7 = (__int64)a1[2];
  v91 = 257;
  if ( v7 == *(_QWORD *)(a3 + 8) )
    goto LABEL_6;
  if ( *(_BYTE *)a3 <= 0x15u )
  {
    a3 = (*((__int64 (__fastcall **)(void **, __int64, __int64))*v101 + 17))(v101, a3, v7);
    if ( *(_BYTE *)a3 > 0x1Cu )
    {
      (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64, __int64, __int64, _BYTE **))*v102 + 2))(
        v102,
        a3,
        v90,
        v98,
        v99,
        v8,
        a2,
        &v94);
      v9 = v94;
      v10 = &v94[16 * (unsigned int)v95];
      if ( v94 != v10 )
      {
        do
        {
          v11 = *((_QWORD *)v9 + 1);
          v12 = *(_DWORD *)v9;
          v9 += 16;
          sub_B99FD0(a3, v12, v11);
        }
        while ( v10 != v9 );
      }
    }
LABEL_6:
    v88 = a3;
    if ( !(_BYTE)qword_4FE9CC8 )
      goto LABEL_7;
    goto LABEL_33;
  }
  v93 = 257;
  a3 = sub_B52210(a3, v7, (__int64)v92, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64, __int64, __int64, _BYTE **))*v102 + 2))(
    v102,
    a3,
    v90,
    v98,
    v99,
    v49,
    a2,
    &v94);
  v50 = v94;
  v51 = &v94[16 * (unsigned int)v95];
  if ( v94 == v51 )
    goto LABEL_6;
  do
  {
    v52 = *((_QWORD *)v50 + 1);
    v53 = *(_DWORD *)v50;
    v50 += 16;
    sub_B99FD0(a3, v53, v52);
  }
  while ( v51 != v50 );
  v88 = a3;
  if ( !(_BYTE)qword_4FE9CC8 )
  {
LABEL_7:
    v13 = *a1;
    if ( (_BYTE)qword_4FE93C8 )
      v87 = sub_BCB2B0(v13);
    else
      v87 = sub_BCB2E0(v13);
    v14 = sub_BCE3C0(*a1, 0);
    v91 = 257;
    v15 = v14;
    v16 = sub_AD64C0(*(_QWORD *)(v88 + 8), (__int64)a1[5], 0);
    v17 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v101 + 2))(v101, 28, v88, v16);
    if ( !v17 )
    {
      v93 = 257;
      v17 = sub_B504D0(28, v88, v16, (__int64)v92, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v102 + 2))(v102, v17, v90, v98, v99);
      v54 = v94;
      v55 = &v94[16 * (unsigned int)v95];
      if ( v94 != v55 )
      {
        do
        {
          v56 = *((_QWORD *)v54 + 1);
          v57 = *(_DWORD *)v54;
          v54 += 16;
          sub_B99FD0(v17, v57, v56);
        }
        while ( v55 != v54 );
      }
    }
    v91 = 257;
    v18 = sub_AD64C0(*(_QWORD *)(v17 + 8), *((int *)a1 + 8), 0);
    v19 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, _QWORD))*v101 + 3))(v101, 26, v17, v18, 0);
    if ( !v19 )
    {
      v93 = 257;
      v19 = sub_B504D0(26, v17, v18, (__int64)v92, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v102 + 2))(v102, v19, v90, v98, v99);
      v81 = v94;
      v82 = &v94[16 * (unsigned int)v95];
      if ( v94 != v82 )
      {
        do
        {
          v83 = *((_QWORD *)v81 + 1);
          v84 = *(_DWORD *)v81;
          v81 += 16;
          sub_B99FD0(v19, v84, v83);
        }
        while ( v82 != v81 );
      }
    }
    v91 = 257;
    v20 = (__int64)a1[16];
    v21 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, _QWORD, _QWORD))*v101 + 4))(
            v101,
            13,
            v19,
            v20,
            0,
            0);
    if ( !v21 )
    {
      v93 = 257;
      v21 = sub_B504D0(13, v19, v20, (__int64)v92, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v102 + 2))(v102, v21, v90, v98, v99);
      v62 = v94;
      v63 = &v94[16 * (unsigned int)v95];
      if ( v94 != v63 )
      {
        do
        {
          v64 = *((_QWORD *)v62 + 1);
          v65 = *(_DWORD *)v62;
          v62 += 16;
          sub_B99FD0(v21, v65, v64);
        }
        while ( v63 != v62 );
      }
    }
    v91 = 257;
    if ( v15 == *(_QWORD *)(v21 + 8) )
    {
      v22 = v21;
    }
    else
    {
      v22 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v101 + 15))(v101, 48, v21, v15);
      if ( !v22 )
      {
        v93 = 257;
        v22 = sub_B51D30(48, v21, v15, (__int64)v92, 0, 0);
        if ( (unsigned __int8)sub_920620(v22) )
        {
          v76 = v104;
          if ( v103 )
            sub_B99FD0(v22, 3u, v103);
          sub_B45150(v22, v76);
        }
        (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v102 + 2))(v102, v22, v90, v98, v99);
        v77 = v94;
        v78 = &v94[16 * (unsigned int)v95];
        if ( v94 != v78 )
        {
          do
          {
            v79 = *((_QWORD *)v77 + 1);
            v80 = *(_DWORD *)v77;
            v77 += 16;
            sub_B99FD0(v22, v80, v79);
          }
          while ( v78 != v77 );
        }
      }
    }
    v91 = 257;
    v23 = sub_AA4E30(v97);
    v24 = sub_AE5020(v23, v87);
    v93 = 257;
    v25 = sub_BD2C40(80, unk_3F10A14);
    v26 = (__int64)v25;
    if ( v25 )
      sub_B4D190((__int64)v25, v87, v22, (__int64)v92, 0, v24, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v102 + 2))(v102, v26, v90, v98, v99);
    v27 = v94;
    v28 = &v94[16 * (unsigned int)v95];
    if ( v94 != v28 )
    {
      do
      {
        v29 = *((_QWORD *)v27 + 1);
        v30 = *(_DWORD *)v27;
        v27 += 16;
        sub_B99FD0(v26, v30, v29);
      }
      while ( v28 != v27 );
    }
    if ( (_BYTE)qword_4FE93C8 )
    {
      v44 = sub_BCB2B0(*a1);
      v45 = sub_ACD640(v44, 255, 0);
      v91 = 257;
      v46 = v45;
      v47 = (_QWORD *)(*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v101 + 7))(v101, 36, v26, v45);
      if ( !v47 )
      {
        v93 = 257;
        v47 = sub_BD2C40(72, unk_3F10FD0);
        if ( v47 )
        {
          v66 = *(_QWORD *)(v26 + 8);
          v67 = *(unsigned __int8 *)(v66 + 8);
          if ( (unsigned int)(v67 - 17) > 1 )
          {
            v71 = sub_BCB2A0(*(_QWORD **)v66);
          }
          else
          {
            v68 = *(_DWORD *)(v66 + 32);
            v69 = *(_QWORD **)v66;
            BYTE4(v89) = (_BYTE)v67 == 18;
            LODWORD(v89) = v68;
            v70 = (__int64 *)sub_BCB2A0(v69);
            v71 = sub_BCE1B0(v70, v89);
          }
          sub_B523C0((__int64)v47, v71, 53, 36, v26, v46, (__int64)v92, 0, 0, 0);
        }
        (*((void (__fastcall **)(void **, _QWORD *, _BYTE *, __int64, __int64))*v102 + 2))(v102, v47, v90, v98, v99);
        v72 = &v94[16 * (unsigned int)v95];
        for ( i = v94; v72 != i; i += 16 )
        {
          v74 = *((_QWORD *)i + 1);
          v75 = *(_DWORD *)i;
          sub_B99FD0((__int64)v47, v75, v74);
        }
      }
      v48 = sub_F38250((__int64)v47, (__int64 *)(v85 + 24), 0, 0, 0, 0, 0, 0);
      sub_D5F1F0((__int64)v86, v48);
    }
    v31 = sub_AD64C0(v87, 1, 0);
    v91 = 257;
    v32 = v31;
    v33 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64, _QWORD, _QWORD))*v101 + 4))(
            v101,
            13,
            v26,
            v31,
            0,
            0);
    if ( !v33 )
    {
      v93 = 257;
      v33 = sub_B504D0(13, v26, v32, (__int64)v92, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v102 + 2))(v102, v33, v90, v98, v99);
      v58 = v94;
      v59 = &v94[16 * (unsigned int)v95];
      if ( v94 != v59 )
      {
        do
        {
          v60 = *((_QWORD *)v58 + 1);
          v61 = *(_DWORD *)v58;
          v58 += 16;
          sub_B99FD0(v33, v61, v60);
        }
        while ( v59 != v58 );
      }
    }
    v34 = sub_AA4E30(v97);
    v35 = sub_AE5020(v34, *(_QWORD *)(v33 + 8));
    v93 = 257;
    v36 = sub_BD2C40(80, unk_3F10A10);
    v38 = (__int64)v36;
    if ( v36 )
      sub_B4D3C0((__int64)v36, v33, v22, 0, v35, v37, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v102 + 2))(v102, v38, v92, v98, v99);
    v39 = v94;
    v40 = &v94[16 * (unsigned int)v95];
    if ( v94 != v40 )
    {
      do
      {
        v41 = *((_QWORD *)v39 + 1);
        v42 = *(_DWORD *)v39;
        v39 += 16;
        sub_B99FD0(v38, v42, v41);
      }
      while ( v40 != v39 );
    }
    nullsub_61();
    v109 = &unk_49DA100;
    nullsub_63();
    v43 = (unsigned __int64)v94;
    if ( v94 != v96 )
      goto LABEL_25;
    return;
  }
LABEL_33:
  v93 = 257;
  sub_24869A0(v86, (__int64)a1[2 * a4 + 6], (__int64)a1[2 * a4 + 7], &v88, 1, (__int64)v92, 0);
  nullsub_61();
  v109 = &unk_49DA100;
  nullsub_63();
  v43 = (unsigned __int64)v94;
  if ( v94 != v96 )
LABEL_25:
    _libc_free(v43);
}
