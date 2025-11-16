// Function: sub_954F10
// Address: 0x954f10
//
__int64 __fastcall sub_954F10(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // r14
  __m128i *v10; // rax
  __m128i *v11; // rax
  __int64 v12; // rdi
  _QWORD *v13; // rax
  __int64 v14; // rdx
  char *v15; // r8
  char *v16; // rdi
  char *v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rcx
  char *v20; // rsi
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // r14
  __int64 v27; // r15
  __int64 v28; // rsi
  __int64 v29; // rdi
  __int64 (__fastcall *v30)(__int64, __int64, __int64); // rax
  unsigned int *v31; // r13
  unsigned int *v32; // r12
  __int64 v33; // rdx
  __int64 v34; // rsi
  __int64 v35; // rdi
  __int64 v36; // rax
  char v37; // al
  int v38; // r13d
  __int64 v39; // rax
  __int64 v40; // r9
  __int64 v41; // r12
  unsigned int *v42; // r15
  unsigned int *v43; // r13
  __int64 v44; // rdx
  __int64 v45; // rsi
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  __int64 *v48; // rdi
  int v49; // r13d
  int v50; // r12d
  __int64 v51; // rax
  unsigned __int64 v52; // rsi
  __int64 v53; // rax
  int v54; // r14d
  __int64 v55; // r13
  __int64 v56; // rax
  unsigned __int8 v57; // al
  __int64 v58; // rax
  int v59; // r9d
  __int64 v60; // r12
  __int64 v61; // rsi
  __int64 v62; // r13
  unsigned int *v63; // rbx
  unsigned int *v64; // r13
  __int64 v65; // rdx
  _BYTE *v66; // rdi
  __int64 v68; // rcx
  __int64 v69; // rax
  unsigned int *v70; // r13
  unsigned int *v71; // r12
  __int64 v72; // rdx
  __int64 v73; // rsi
  int *v74; // r14
  __int64 v75; // rax
  int *v76; // rdx
  int *v77; // r15
  _BOOL4 v78; // r12d
  __int64 v79; // rax
  int *v80; // r14
  __int64 v81; // rax
  int *v82; // rdx
  int *v83; // r13
  _BOOL4 v84; // r12d
  __int64 v85; // rax
  __int64 v86; // [rsp-8h] [rbp-1B8h]
  unsigned int v88; // [rsp+10h] [rbp-1A0h]
  unsigned __int16 v89; // [rsp+16h] [rbp-19Ah]
  int v90; // [rsp+28h] [rbp-188h]
  __int64 v91; // [rsp+30h] [rbp-180h]
  int v92; // [rsp+30h] [rbp-180h]
  __int64 v93; // [rsp+38h] [rbp-178h]
  __int64 v94; // [rsp+38h] [rbp-178h]
  __m128i *v95; // [rsp+40h] [rbp-170h]
  __m128i *v96; // [rsp+48h] [rbp-168h]
  __m128i *v97; // [rsp+50h] [rbp-160h]
  __m128i *v98; // [rsp+58h] [rbp-158h]
  _BYTE *v99; // [rsp+60h] [rbp-150h] BYREF
  __int64 v100; // [rsp+68h] [rbp-148h]
  _BYTE v101[32]; // [rsp+70h] [rbp-140h] BYREF
  _BYTE v102[32]; // [rsp+90h] [rbp-120h] BYREF
  __int16 v103; // [rsp+B0h] [rbp-100h]
  __int64 v104; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v105; // [rsp+C8h] [rbp-E8h]
  __int64 v106; // [rsp+D0h] [rbp-E0h]
  __int64 v107; // [rsp+D8h] [rbp-D8h]
  __int64 v108; // [rsp+E0h] [rbp-D0h]
  __int64 v109; // [rsp+E8h] [rbp-C8h]
  __int64 v110; // [rsp+F0h] [rbp-C0h]
  __int64 v111; // [rsp+F8h] [rbp-B8h]
  __int64 v112; // [rsp+100h] [rbp-B0h]
  __int64 v113; // [rsp+108h] [rbp-A8h]
  __int64 v114; // [rsp+110h] [rbp-A0h]
  __int64 v115; // [rsp+118h] [rbp-98h]
  _QWORD v116[12]; // [rsp+120h] [rbp-90h] BYREF
  char v117; // [rsp+180h] [rbp-30h] BYREF

  v6 = *(_QWORD *)(a4 + 16);
  v7 = *(_QWORD *)(v6 + 16);
  v8 = *(_QWORD *)(v7 + 16);
  v9 = *(_QWORD *)(v8 + 16);
  if ( v9 )
  {
    v95 = sub_92F410(a2, v6);
    v96 = sub_92F410(a2, v7);
    v10 = sub_92F410(a2, v8);
  }
  else
  {
    v9 = *(_QWORD *)(v7 + 16);
    v8 = 0;
    v95 = sub_92F410(a2, v6);
    v96 = sub_92F410(a2, v7);
    v10 = 0;
  }
  v97 = v10;
  v11 = sub_92F410(a2, v9);
  v12 = *(_QWORD *)(a2 + 40);
  v98 = v11;
  v91 = sub_BCB2E0(v12);
  v99 = v101;
  v100 = 0x300000000LL;
  if ( byte_4F6D3B0 || !(unsigned int)sub_2207590(&byte_4F6D3B0) )
  {
    if ( byte_4F6D378 )
      goto LABEL_5;
    goto LABEL_62;
  }
  dword_4F6D3C8 = 0;
  v74 = (int *)&v104;
  v104 = 0x21CA0000018BLL;
  v105 = 0x18D00000001LL;
  v106 = 0x4000021CALL;
  v107 = 0x21CA0000018FLL;
  v108 = 0x19100000002LL;
  v109 = 0x3000021CALL;
  v110 = 0x21CB0000018CLL;
  v111 = 0x18E00000001LL;
  v112 = 0x4000021CBLL;
  v113 = 0x21CB00000190LL;
  v114 = 0x19200000002LL;
  v115 = 0x3000021CBLL;
  qword_4F6D3D0 = 0;
  qword_4F6D3D8 = (__int64)&dword_4F6D3C8;
  qword_4F6D3E0 = (__int64)&dword_4F6D3C8;
  qword_4F6D3E8 = 0;
  v93 = v8;
  v90 = a3;
  do
  {
    v75 = sub_954E10((_QWORD *)&dword_4F6D3C8 - 1, (__int64)&dword_4F6D3C8, v74);
    v77 = v76;
    if ( v76 )
    {
      v78 = v75 || v76 == &dword_4F6D3C8 || *v74 < v76[8];
      v79 = sub_22077B0(48);
      *(_QWORD *)(v79 + 32) = *(_QWORD *)v74;
      *(_DWORD *)(v79 + 40) = v74[2];
      sub_220F040(v78, v79, v77, &dword_4F6D3C8);
      ++qword_4F6D3E8;
    }
    v74 += 3;
  }
  while ( v74 != (int *)v116 );
  v8 = v93;
  a3 = v90;
  __cxa_atexit((void (*)(void *))sub_948AA0, &unk_4F6D3C0, &qword_4A427C0);
  sub_2207640(&byte_4F6D3B0);
  if ( !byte_4F6D378 )
  {
LABEL_62:
    if ( (unsigned int)sub_2207590(&byte_4F6D378) )
    {
      dword_4F6D388 = 0;
      v80 = (int *)&v104;
      v104 = 0x1FC600000152LL;
      v105 = 0x15400000001LL;
      v106 = 0x400001FC6LL;
      v107 = 0x1FC600000156LL;
      v108 = 0x15800000002LL;
      v109 = 0x300001FC6LL;
      v110 = 0x1FC700000153LL;
      v111 = 0x15500000001LL;
      v112 = 0x400001FC7LL;
      v113 = 0x1FC700000157LL;
      v114 = 0x15900000002LL;
      v115 = 0x300001FC7LL;
      v116[0] = 0x23C50000012ELL;
      v116[1] = 0x13000000001LL;
      v116[2] = 0x4000023C5LL;
      v116[3] = 0x23C500000132LL;
      v116[4] = 0x13400000002LL;
      v116[5] = 0x3000023C5LL;
      v116[6] = 0x23C80000012FLL;
      v116[7] = 0x13100000001LL;
      v116[8] = 0x4000023C8LL;
      v116[9] = 0x23C800000133LL;
      v116[10] = 0x13500000002LL;
      v116[11] = 0x3000023C8LL;
      qword_4F6D390 = 0;
      qword_4F6D398 = (__int64)&dword_4F6D388;
      qword_4F6D3A0 = (__int64)&dword_4F6D388;
      qword_4F6D3A8 = 0;
      v94 = v8;
      do
      {
        v81 = sub_954E10((_QWORD *)&dword_4F6D388 - 1, (__int64)&dword_4F6D388, v80);
        v83 = v82;
        if ( v82 )
        {
          v84 = v81 || v82 == &dword_4F6D388 || *v80 < v82[8];
          v85 = sub_22077B0(48);
          *(_QWORD *)(v85 + 32) = *(_QWORD *)v80;
          *(_DWORD *)(v85 + 40) = v80[2];
          sub_220F040(v84, v85, v83, &dword_4F6D388);
          ++qword_4F6D3A8;
        }
        v80 += 3;
      }
      while ( v80 != (int *)&v117 );
      v8 = v94;
      __cxa_atexit((void (*)(void *))sub_948AA0, &unk_4F6D380, &qword_4A427C0);
      sub_2207640(&byte_4F6D378);
    }
  }
LABEL_5:
  v13 = &unk_4F6D380;
  if ( v8 )
    v13 = &unk_4F6D3C0;
  v14 = v13[2];
  v15 = (char *)(v13 + 1);
  if ( !v14 )
    goto LABEL_14;
  v16 = (char *)(v13 + 1);
  v17 = (char *)v13[2];
  do
  {
    while ( 1 )
    {
      v18 = *((_QWORD *)v17 + 2);
      v19 = *((_QWORD *)v17 + 3);
      if ( *((_DWORD *)v17 + 8) >= a3 )
        break;
      v17 = (char *)*((_QWORD *)v17 + 3);
      if ( !v19 )
        goto LABEL_12;
    }
    v16 = v17;
    v17 = (char *)*((_QWORD *)v17 + 2);
  }
  while ( v18 );
LABEL_12:
  if ( v15 != v16 && (v20 = v15, *((_DWORD *)v16 + 8) <= a3) )
  {
    do
    {
      while ( 1 )
      {
        v68 = *(_QWORD *)(v14 + 16);
        v69 = *(_QWORD *)(v14 + 24);
        if ( *(_DWORD *)(v14 + 32) >= a3 )
          break;
        v14 = *(_QWORD *)(v14 + 24);
        if ( !v69 )
          goto LABEL_47;
      }
      v20 = (char *)v14;
      v14 = *(_QWORD *)(v14 + 16);
    }
    while ( v68 );
LABEL_47:
    if ( v15 == v20 || *((_DWORD *)v20 + 8) > a3 )
      sub_426320((__int64)"map::at");
    v21 = *((unsigned int *)v20 + 10);
    v88 = *((_DWORD *)v20 + 9);
  }
  else
  {
LABEL_14:
    v88 = 0;
    v21 = 1;
  }
  v22 = sub_BCB2D0(*(_QWORD *)(a2 + 40));
  v23 = sub_ACD640(v22, v21, 0);
  v24 = (unsigned int)v100;
  v25 = (unsigned int)v100 + 1LL;
  if ( v25 > HIDWORD(v100) )
  {
    sub_C8D5F0(&v99, v101, v25, 8);
    v24 = (unsigned int)v100;
  }
  v26 = 0;
  *(_QWORD *)&v99[8 * v24] = v23;
  LODWORD(v100) = v100 + 1;
  while ( 1 )
  {
    v27 = (__int64)*(&v95 + v26);
    if ( v27 )
      break;
LABEL_34:
    if ( ++v26 == 4 )
      goto LABEL_35;
  }
  v28 = sub_BCE770(v91, *(_DWORD *)(*(_QWORD *)(v27 + 8) + 8LL) >> 8);
  v103 = 257;
  if ( v28 != *(_QWORD *)(v27 + 8) )
  {
    if ( *(_BYTE *)v27 > 0x15u )
    {
      LOWORD(v108) = 257;
      v27 = sub_B52210(v27, v28, &v104, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
        *(_QWORD *)(a2 + 136),
        v27,
        v102,
        *(_QWORD *)(a2 + 104),
        *(_QWORD *)(a2 + 112));
      v70 = *(unsigned int **)(a2 + 48);
      v71 = &v70[4 * *(unsigned int *)(a2 + 56)];
      while ( v71 != v70 )
      {
        v72 = *((_QWORD *)v70 + 1);
        v73 = *v70;
        v70 += 4;
        sub_B99FD0(v27, v73, v72);
      }
    }
    else
    {
      v29 = *(_QWORD *)(a2 + 128);
      v30 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v29 + 136LL);
      if ( v30 == sub_928970 )
        v27 = sub_ADAFB0(v27, v28);
      else
        v27 = v30(v29, v27, v28);
      if ( *(_BYTE *)v27 > 0x1Cu )
      {
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
          *(_QWORD *)(a2 + 136),
          v27,
          v102,
          *(_QWORD *)(a2 + 104),
          *(_QWORD *)(a2 + 112));
        v31 = *(unsigned int **)(a2 + 48);
        v32 = &v31[4 * *(unsigned int *)(a2 + 56)];
        while ( v32 != v31 )
        {
          v33 = *((_QWORD *)v31 + 1);
          v34 = *v31;
          v31 += 4;
          sub_B99FD0(v27, v34, v33);
        }
      }
    }
  }
  *(&v95 + v26) = (__m128i *)v27;
  if ( v26 != 3 )
  {
    v35 = *(_QWORD *)(a2 + 96);
    v103 = 257;
    v36 = sub_AA4E30(v35);
    v37 = sub_AE5020(v36, v91);
    v38 = v89;
    LOWORD(v108) = 257;
    LOBYTE(v38) = v37;
    v89 = v38;
    v39 = sub_BD2C40(80, unk_3F10A14);
    v41 = v39;
    if ( v39 )
    {
      sub_B4D190(v39, v91, v27, (unsigned int)&v104, 0, v38, 0, 0);
      v40 = v86;
    }
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a2 + 136) + 16LL))(
      *(_QWORD *)(a2 + 136),
      v41,
      v102,
      *(_QWORD *)(a2 + 104),
      *(_QWORD *)(a2 + 112),
      v40);
    v42 = *(unsigned int **)(a2 + 48);
    v43 = &v42[4 * *(unsigned int *)(a2 + 56)];
    while ( v43 != v42 )
    {
      v44 = *((_QWORD *)v42 + 1);
      v45 = *v42;
      v42 += 4;
      sub_B99FD0(v41, v45, v44);
    }
    v46 = (unsigned int)v100;
    v47 = (unsigned int)v100 + 1LL;
    if ( v47 > HIDWORD(v100) )
    {
      sub_C8D5F0(&v99, v101, v47, 8);
      v46 = (unsigned int)v100;
    }
    *(_QWORD *)&v99[8 * v46] = v41;
    LODWORD(v100) = v100 + 1;
    goto LABEL_34;
  }
LABEL_35:
  v48 = *(__int64 **)(a2 + 32);
  v49 = (int)v99;
  LOWORD(v108) = 257;
  v50 = v100;
  v51 = sub_90A810(v48, v88, 0, 0);
  v52 = 0;
  if ( v51 )
    v52 = *(_QWORD *)(v51 + 24);
  v53 = sub_921880((unsigned int **)(a2 + 48), v52, v51, v49, v50, (__int64)&v104, 0);
  v54 = (int)v98;
  v55 = v53;
  v56 = sub_AA4E30(*(_QWORD *)(a2 + 96));
  v57 = sub_AE5020(v56, *(_QWORD *)(v55 + 8));
  LOWORD(v108) = 257;
  v92 = v57;
  v58 = sub_BD2C40(80, unk_3F10A10);
  v60 = v58;
  if ( v58 )
    sub_B4D3C0(v58, v55, v54, 0, v92, v59, 0, 0);
  v61 = v60;
  (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
    *(_QWORD *)(a2 + 136),
    v60,
    &v104,
    *(_QWORD *)(a2 + 104),
    *(_QWORD *)(a2 + 112));
  v62 = 4LL * *(unsigned int *)(a2 + 56);
  v63 = *(unsigned int **)(a2 + 48);
  v64 = &v63[v62];
  while ( v64 != v63 )
  {
    v65 = *((_QWORD *)v63 + 1);
    v61 = *v63;
    v63 += 4;
    sub_B99FD0(v60, v61, v65);
  }
  v66 = v99;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v66 != v101 )
    _libc_free(v66, v61);
  return a1;
}
