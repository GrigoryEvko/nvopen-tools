// Function: sub_2CCA9F0
// Address: 0x2cca9f0
//
void __fastcall sub_2CCA9F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __int64 a7,
        __int64 a8)
{
  _QWORD *v9; // rax
  _QWORD *v10; // r12
  __int64 v11; // rax
  __int64 v12; // r14
  unsigned __int64 v13; // rax
  int v14; // edx
  __int64 v15; // rbx
  __int64 v16; // rax
  _BYTE *v17; // rax
  unsigned __int8 *v18; // rax
  unsigned __int8 *v19; // r9
  __int64 v20; // rbx
  __int64 v21; // rdx
  int v22; // eax
  int v23; // eax
  unsigned int v24; // ecx
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rcx
  __int64 v28; // r12
  __int64 v29; // rax
  _QWORD *v30; // rax
  __int64 v31; // r9
  __int64 v32; // r13
  __int64 v33; // r12
  unsigned int *v34; // r12
  unsigned int *v35; // rbx
  __int64 v36; // rdx
  unsigned int v37; // esi
  _BYTE *v38; // rax
  __int64 v39; // rdx
  int v40; // eax
  int v41; // eax
  unsigned int v42; // ecx
  __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rcx
  __int64 v46; // rax
  __int64 v47; // r15
  _QWORD *v48; // rax
  __int64 v49; // r9
  __int64 v50; // rbx
  unsigned int *v51; // r13
  unsigned int *v52; // r12
  __int64 v53; // rdx
  unsigned int v54; // esi
  unsigned int *v55; // rdi
  __int64 v56; // rsi
  __int64 v57; // r15
  __int64 v58; // rax
  char v59; // r14
  _QWORD *v60; // rax
  __int64 v61; // r9
  __int64 v62; // r12
  unsigned int *v63; // r15
  unsigned int *v64; // rbx
  __int64 v65; // rdx
  unsigned int v66; // esi
  __int64 v67; // [rsp-10h] [rbp-240h]
  char v68; // [rsp+Ch] [rbp-224h]
  __int64 v69; // [rsp+10h] [rbp-220h]
  __int64 v70; // [rsp+28h] [rbp-208h]
  __int64 v71; // [rsp+28h] [rbp-208h]
  __int64 v72; // [rsp+30h] [rbp-200h]
  char v74; // [rsp+38h] [rbp-1F8h]
  __int64 v77; // [rsp+48h] [rbp-1E8h]
  _QWORD *v78; // [rsp+50h] [rbp-1E0h]
  __int64 v79; // [rsp+50h] [rbp-1E0h]
  __int64 v80; // [rsp+50h] [rbp-1E0h]
  unsigned int v82; // [rsp+58h] [rbp-1D8h]
  unsigned __int8 *v83; // [rsp+60h] [rbp-1D0h]
  _BYTE *v84; // [rsp+78h] [rbp-1B8h] BYREF
  _QWORD v85[4]; // [rsp+80h] [rbp-1B0h] BYREF
  char v86; // [rsp+A0h] [rbp-190h]
  char v87; // [rsp+A1h] [rbp-18Fh]
  _BYTE *v88[4]; // [rsp+B0h] [rbp-180h] BYREF
  __int16 v89; // [rsp+D0h] [rbp-160h]
  unsigned int *v90[2]; // [rsp+E0h] [rbp-150h] BYREF
  char v91; // [rsp+F0h] [rbp-140h] BYREF
  __int16 v92; // [rsp+100h] [rbp-130h]
  void *v93; // [rsp+160h] [rbp-D0h]
  unsigned int *v94; // [rsp+170h] [rbp-C0h] BYREF
  __int64 v95; // [rsp+178h] [rbp-B8h]
  _BYTE v96[16]; // [rsp+180h] [rbp-B0h] BYREF
  __int16 v97; // [rsp+190h] [rbp-A0h]
  __int64 v98; // [rsp+1A0h] [rbp-90h]
  __int64 v99; // [rsp+1A8h] [rbp-88h]
  __int64 v100; // [rsp+1B0h] [rbp-80h]
  __int64 v101; // [rsp+1B8h] [rbp-78h]
  void **v102; // [rsp+1C0h] [rbp-70h]
  void **v103; // [rsp+1C8h] [rbp-68h]
  __int64 v104; // [rsp+1D0h] [rbp-60h]
  int v105; // [rsp+1D8h] [rbp-58h]
  __int16 v106; // [rsp+1DCh] [rbp-54h]
  char v107; // [rsp+1DEh] [rbp-52h]
  __int64 v108; // [rsp+1E0h] [rbp-50h]
  __int64 v109; // [rsp+1E8h] [rbp-48h]
  void *v110; // [rsp+1F0h] [rbp-40h] BYREF
  void *v111; // [rsp+1F8h] [rbp-38h] BYREF

  if ( *(_BYTE *)a4 != 17 )
    goto LABEL_5;
  v9 = *(_QWORD **)(a4 + 24);
  if ( *(_DWORD *)(a4 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  v78 = v9;
  if ( (unsigned int)qword_5013628 >= (unsigned __int64)v9 )
  {
    sub_23D0AB0((__int64)&v94, a1, 0, 0, 0);
    v71 = *(_QWORD *)(a4 + 8);
    if ( v78 )
    {
      v82 = 0;
      v56 = 0;
      do
      {
        v90[0] = (unsigned int *)"dst.gep.unroll";
        v92 = 259;
        v88[0] = (_BYTE *)sub_AD64C0(v71, v56, 0);
        v57 = sub_921130(&v94, a2, a3, v88, 1, (__int64)v90, 0);
        v58 = sub_AA4E30(v98);
        v59 = sub_AE5020(v58, *(_QWORD *)(a5 + 8));
        v92 = 257;
        v60 = sub_BD2C40(80, unk_3F10A10);
        v62 = (__int64)v60;
        if ( v60 )
          sub_B4D3C0((__int64)v60, a5, v57, a6, v59, v61, 0, 0);
        (*((void (__fastcall **)(void **, __int64, unsigned int **, __int64, __int64))*v103 + 2))(
          v103,
          v62,
          v90,
          v99,
          v100);
        v63 = v94;
        v64 = &v94[4 * (unsigned int)v95];
        if ( v94 != v64 )
        {
          do
          {
            v65 = *((_QWORD *)v63 + 1);
            v66 = *v63;
            v63 += 4;
            sub_B99FD0(v62, v66, v65);
          }
          while ( v64 != v63 );
        }
        v56 = ++v82;
      }
      while ( (_QWORD *)v82 != v78 );
    }
    nullsub_61();
    v110 = &unk_49DA100;
    nullsub_63();
    v55 = v94;
    if ( v94 != (unsigned int *)v96 )
      goto LABEL_44;
  }
  else
  {
LABEL_5:
    v10 = *(_QWORD **)(a1 + 40);
    v94 = (unsigned int *)"memset.exit";
    v97 = 259;
    v72 = sub_AA8550(v10, (__int64 *)(a1 + 24), 0, (__int64)&v94, 0);
    v94 = (unsigned int *)"memset.loop";
    v97 = 259;
    v11 = sub_22077B0(0x50u);
    v12 = v11;
    if ( v11 )
      sub_AA4D50(v11, a7, (__int64)&v94, a8, v72);
    v13 = v10[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (_QWORD *)v13 == v10 + 6 )
    {
      v15 = 0;
    }
    else
    {
      if ( !v13 )
        BUG();
      v14 = *(unsigned __int8 *)(v13 - 24);
      v15 = 0;
      v16 = v13 - 24;
      if ( (unsigned int)(v14 - 30) < 0xB )
        v15 = v16;
    }
    sub_23D0AB0((__int64)v90, v15, 0, 0, 0);
    v97 = 257;
    v79 = *(_QWORD *)(a4 + 8);
    v17 = (_BYTE *)sub_AD64C0(v79, 0, 0);
    v70 = sub_92B530(v90, 0x22u, a4, v17, (__int64)&v94);
    v18 = (unsigned __int8 *)sub_BD2C40(72, 3u);
    v19 = v18;
    if ( v18 )
    {
      v83 = v18;
      sub_B4C9A0((__int64)v18, v12, v72, v70, 3u, (__int64)v18, 0, 0);
      v19 = v83;
    }
    sub_F34910(v15, v19);
    v101 = sub_AA48A0(v12);
    v102 = &v110;
    v103 = &v111;
    v94 = (unsigned int *)v96;
    v110 = &unk_49DA100;
    v95 = 0x200000000LL;
    LOWORD(v100) = 0;
    v111 = &unk_49DA0B0;
    v99 = v12 + 48;
    v88[0] = "index";
    v106 = 512;
    v104 = 0;
    v105 = 0;
    v107 = 7;
    v108 = 0;
    v109 = 0;
    v98 = v12;
    v89 = 259;
    v20 = sub_D5C860((__int64 *)&v94, v79, 0, (__int64)v88);
    v21 = sub_AD64C0(v79, 0, 0);
    v22 = *(_DWORD *)(v20 + 4) & 0x7FFFFFF;
    if ( v22 == *(_DWORD *)(v20 + 72) )
    {
      v69 = v21;
      sub_B48D90(v20);
      v21 = v69;
      v22 = *(_DWORD *)(v20 + 4) & 0x7FFFFFF;
    }
    v23 = (v22 + 1) & 0x7FFFFFF;
    v24 = v23 | *(_DWORD *)(v20 + 4) & 0xF8000000;
    v25 = *(_QWORD *)(v20 - 8) + 32LL * (unsigned int)(v23 - 1);
    *(_DWORD *)(v20 + 4) = v24;
    if ( *(_QWORD *)v25 )
    {
      v26 = *(_QWORD *)(v25 + 8);
      **(_QWORD **)(v25 + 16) = v26;
      if ( v26 )
        *(_QWORD *)(v26 + 16) = *(_QWORD *)(v25 + 16);
    }
    *(_QWORD *)v25 = v21;
    if ( v21 )
    {
      v27 = *(_QWORD *)(v21 + 16);
      *(_QWORD *)(v25 + 8) = v27;
      if ( v27 )
        *(_QWORD *)(v27 + 16) = v25 + 8;
      *(_QWORD *)(v25 + 16) = v21 + 16;
      *(_QWORD *)(v21 + 16) = v25;
    }
    v68 = a6;
    *(_QWORD *)(*(_QWORD *)(v20 - 8)
              + 32LL * *(unsigned int *)(v20 + 72)
              + 8LL * ((*(_DWORD *)(v20 + 4) & 0x7FFFFFFu) - 1)) = v10;
    v85[0] = "dst.gep";
    v87 = 1;
    v86 = 3;
    v84 = (_BYTE *)v20;
    v28 = sub_921130(&v94, a2, a3, &v84, 1, (__int64)v85, 0);
    v29 = sub_AA4E30(v98);
    v74 = sub_AE5020(v29, *(_QWORD *)(a5 + 8));
    v89 = 257;
    v30 = sub_BD2C40(80, unk_3F10A10);
    v31 = v67;
    v32 = (__int64)v30;
    if ( v30 )
      sub_B4D3C0((__int64)v30, a5, v28, v68, v74, v67, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _BYTE **, __int64, __int64, __int64))*v103 + 2))(
      v103,
      v32,
      v88,
      v99,
      v100,
      v31);
    v33 = 4LL * (unsigned int)v95;
    if ( v94 != &v94[v33] )
    {
      v77 = v20;
      v34 = &v94[v33];
      v35 = v94;
      do
      {
        v36 = *((_QWORD *)v35 + 1);
        v37 = *v35;
        v35 += 4;
        sub_B99FD0(v32, v37, v36);
      }
      while ( v34 != v35 );
      v20 = v77;
    }
    v88[0] = "inc";
    v89 = 259;
    v38 = (_BYTE *)sub_AD64C0(v79, 1, 0);
    v39 = sub_929C50(&v94, (_BYTE *)v20, v38, (__int64)v88, 0, 0);
    v40 = *(_DWORD *)(v20 + 4) & 0x7FFFFFF;
    if ( v40 == *(_DWORD *)(v20 + 72) )
    {
      v80 = v39;
      sub_B48D90(v20);
      v39 = v80;
      v40 = *(_DWORD *)(v20 + 4) & 0x7FFFFFF;
    }
    v41 = (v40 + 1) & 0x7FFFFFF;
    v42 = v41 | *(_DWORD *)(v20 + 4) & 0xF8000000;
    v43 = *(_QWORD *)(v20 - 8) + 32LL * (unsigned int)(v41 - 1);
    *(_DWORD *)(v20 + 4) = v42;
    if ( *(_QWORD *)v43 )
    {
      v44 = *(_QWORD *)(v43 + 8);
      **(_QWORD **)(v43 + 16) = v44;
      if ( v44 )
        *(_QWORD *)(v44 + 16) = *(_QWORD *)(v43 + 16);
    }
    *(_QWORD *)v43 = v39;
    if ( v39 )
    {
      v45 = *(_QWORD *)(v39 + 16);
      *(_QWORD *)(v43 + 8) = v45;
      if ( v45 )
        *(_QWORD *)(v45 + 16) = v43 + 8;
      *(_QWORD *)(v43 + 16) = v39 + 16;
      *(_QWORD *)(v39 + 16) = v43;
    }
    *(_QWORD *)(*(_QWORD *)(v20 - 8)
              + 32LL * *(unsigned int *)(v20 + 72)
              + 8LL * ((*(_DWORD *)(v20 + 4) & 0x7FFFFFFu) - 1)) = v12;
    v87 = 1;
    v85[0] = "cmp";
    v86 = 3;
    v46 = sub_92B530(&v94, 0x24u, v39, (_BYTE *)a4, (__int64)v85);
    v89 = 257;
    v47 = v46;
    v48 = sub_BD2C40(72, 3u);
    v50 = (__int64)v48;
    if ( v48 )
      sub_B4C9A0((__int64)v48, v12, v72, v47, 3u, v49, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _BYTE **, __int64, __int64))*v103 + 2))(v103, v50, v88, v99, v100);
    v51 = v94;
    v52 = &v94[4 * (unsigned int)v95];
    if ( v94 != v52 )
    {
      do
      {
        v53 = *((_QWORD *)v51 + 1);
        v54 = *v51;
        v51 += 4;
        sub_B99FD0(v50, v54, v53);
      }
      while ( v52 != v51 );
    }
    nullsub_61();
    v110 = &unk_49DA100;
    nullsub_63();
    if ( v94 != (unsigned int *)v96 )
      _libc_free((unsigned __int64)v94);
    nullsub_61();
    v93 = &unk_49DA100;
    nullsub_63();
    v55 = v90[0];
    if ( (char *)v90[0] != &v91 )
LABEL_44:
      _libc_free((unsigned __int64)v55);
  }
}
