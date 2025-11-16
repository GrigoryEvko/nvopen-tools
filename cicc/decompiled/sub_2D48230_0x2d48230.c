// Function: sub_2D48230
// Address: 0x2d48230
//
void __fastcall sub_2D48230(unsigned __int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 v9; // r15
  __int64 v10; // r14
  unsigned __int64 v11; // rdi
  __int64 v12; // rax
  __int16 v13; // cx
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 *v18; // r13
  __int64 v19; // r15
  __int64 v20; // r9
  _QWORD *v21; // r15
  __int64 v22; // r14
  char *v23; // rbx
  char *v24; // r15
  __int64 v25; // rdx
  unsigned int v26; // esi
  char *v27; // r15
  char *v28; // r14
  __int64 v29; // rdx
  unsigned int v30; // esi
  _QWORD *v31; // rax
  char *v32; // rbx
  char *v33; // r13
  __int64 v34; // rdx
  unsigned int v35; // esi
  _QWORD **v36; // rdx
  int v37; // ecx
  __int64 *v38; // rax
  __int64 v39; // rax
  char *v40; // rbx
  char *v41; // r14
  __int64 v42; // rdx
  unsigned int v43; // esi
  char *v44; // rbx
  char *v45; // r15
  __int64 v46; // r14
  __int64 v47; // rdx
  unsigned int v48; // esi
  __int64 *v49; // rax
  char *v50; // rbx
  char *v51; // r15
  __int64 v52; // rdx
  unsigned int v53; // esi
  _QWORD *v54; // rax
  char *v55; // rbx
  char *v56; // r14
  __int64 v57; // rdx
  unsigned int v58; // esi
  _QWORD *v59; // rax
  char *v60; // r12
  char *v61; // rbx
  __int64 v62; // rdx
  unsigned int v63; // esi
  __int64 v64; // [rsp+8h] [rbp-248h]
  __int64 v65; // [rsp+8h] [rbp-248h]
  __int64 v67; // [rsp+20h] [rbp-230h]
  __int64 v68; // [rsp+20h] [rbp-230h]
  __int64 v69; // [rsp+20h] [rbp-230h]
  __int64 v70; // [rsp+20h] [rbp-230h]
  __int64 v71; // [rsp+20h] [rbp-230h]
  __int64 v72; // [rsp+20h] [rbp-230h]
  __int64 v73; // [rsp+20h] [rbp-230h]
  __int64 v74; // [rsp+20h] [rbp-230h]
  __int64 v76; // [rsp+38h] [rbp-218h]
  _DWORD v77[8]; // [rsp+40h] [rbp-210h] BYREF
  __int16 v78; // [rsp+60h] [rbp-1F0h]
  _QWORD v79[4]; // [rsp+70h] [rbp-1E0h] BYREF
  __int16 v80; // [rsp+90h] [rbp-1C0h]
  _BYTE v81[32]; // [rsp+A0h] [rbp-1B0h] BYREF
  __int16 v82; // [rsp+C0h] [rbp-190h]
  _QWORD v83[5]; // [rsp+D0h] [rbp-180h] BYREF
  __int64 v84; // [rsp+F8h] [rbp-158h]
  __int64 v85; // [rsp+100h] [rbp-150h]
  char *v86; // [rsp+110h] [rbp-140h] BYREF
  int v87; // [rsp+118h] [rbp-138h]
  char v88; // [rsp+120h] [rbp-130h] BYREF
  __int64 v89; // [rsp+148h] [rbp-108h]
  __int64 v90; // [rsp+150h] [rbp-100h]
  __int64 v91; // [rsp+160h] [rbp-F0h]
  __int64 v92; // [rsp+168h] [rbp-E8h]
  void *v93; // [rsp+190h] [rbp-C0h]
  void *v94; // [rsp+198h] [rbp-B8h]
  _QWORD v95[12]; // [rsp+1F0h] [rbp-60h] BYREF

  v2 = a2;
  sub_2D46B10((__int64)&v86, a2, a1[1]);
  _BitScanReverse64(&v3, 1LL << *(_BYTE *)(a2 + 3));
  sub_2D44EF0(
    (__int64)v83,
    (__int64)&v86,
    v2,
    *(_QWORD *)(*(_QWORD *)(v2 - 64) + 8LL),
    *(_QWORD *)(v2 - 96),
    63 - (v3 ^ 0x3F),
    *(_DWORD *)(*a1 + 96) >> 3);
  v4 = *(_QWORD *)(a2 - 64);
  v79[0] = "CmpVal_Shifted";
  v5 = v84;
  v78 = 257;
  v80 = 259;
  if ( v83[0] == *(_QWORD *)(v4 + 8) )
  {
    v6 = v4;
  }
  else
  {
    v67 = v83[0];
    v6 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)v91 + 120LL))(v91, 39, v4, v83[0]);
    if ( !v6 )
    {
      v82 = 257;
      v59 = sub_BD2C40(72, 1u);
      v6 = (__int64)v59;
      if ( v59 )
        sub_B515B0((__int64)v59, v4, v67, (__int64)v81, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _DWORD *, __int64, __int64))(*(_QWORD *)v92 + 16LL))(
        v92,
        v6,
        v77,
        v89,
        v90);
      if ( v86 != &v86[16 * v87] )
      {
        v60 = v86;
        v61 = &v86[16 * v87];
        do
        {
          v62 = *((_QWORD *)v60 + 1);
          v63 = *(_DWORD *)v60;
          v60 += 16;
          sub_B99FD0(v6, v63, v62);
        }
        while ( v61 != v60 );
        v2 = a2;
      }
    }
  }
  v7 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v91 + 32LL))(
         v91,
         25,
         v6,
         v5,
         0,
         0);
  if ( !v7 )
  {
    v82 = 257;
    v7 = sub_B504D0(25, v6, v5, (__int64)v81, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v92 + 16LL))(
      v92,
      v7,
      v79,
      v89,
      v90);
    v27 = v86;
    v28 = &v86[16 * v87];
    if ( v86 != v28 )
    {
      do
      {
        v29 = *((_QWORD *)v27 + 1);
        v30 = *(_DWORD *)v27;
        v27 += 16;
        sub_B99FD0(v7, v30, v29);
      }
      while ( v28 != v27 );
    }
  }
  v8 = *(_QWORD *)(v2 - 32);
  v79[0] = "NewVal_Shifted";
  v80 = 259;
  v68 = v84;
  v78 = 257;
  if ( v83[0] == *(_QWORD *)(v8 + 8) )
  {
    v9 = v8;
  }
  else
  {
    v64 = v83[0];
    v9 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)v91 + 120LL))(v91, 39, v8, v83[0]);
    if ( !v9 )
    {
      v82 = 257;
      v54 = sub_BD2C40(72, 1u);
      v9 = (__int64)v54;
      if ( v54 )
        sub_B515B0((__int64)v54, v8, v64, (__int64)v81, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _DWORD *, __int64, __int64))(*(_QWORD *)v92 + 16LL))(
        v92,
        v9,
        v77,
        v89,
        v90);
      if ( v86 != &v86[16 * v87] )
      {
        v65 = v7;
        v55 = v86;
        v56 = &v86[16 * v87];
        do
        {
          v57 = *((_QWORD *)v55 + 1);
          v58 = *(_DWORD *)v55;
          v55 += 16;
          sub_B99FD0(v9, v58, v57);
        }
        while ( v56 != v55 );
        v7 = v65;
      }
    }
  }
  v10 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v91 + 32LL))(
          v91,
          25,
          v9,
          v68,
          0,
          0);
  if ( !v10 )
  {
    v82 = 257;
    v10 = sub_B504D0(25, v9, v68, (__int64)v81, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v92 + 16LL))(
      v92,
      v10,
      v79,
      v89,
      v90);
    if ( v86 != &v86[16 * v87] )
    {
      v71 = v7;
      v23 = v86;
      v24 = &v86[16 * v87];
      do
      {
        v25 = *((_QWORD *)v23 + 1);
        v26 = *(_DWORD *)v23;
        v23 += 16;
        sub_B99FD0(v10, v26, v25);
      }
      while ( v24 != v23 );
      v7 = v71;
    }
  }
  v11 = *a1;
  if ( ((*(_WORD *)(v2 + 2) >> 5) & 7) == 7 )
  {
    v12 = 7;
  }
  else
  {
    v12 = (*(_WORD *)(v2 + 2) >> 2) & 7;
    v13 = (*(_WORD *)(v2 + 2) >> 2) & 7;
    if ( ((*(_WORD *)(v2 + 2) >> 5) & 7) == 4 )
    {
      if ( v13 == 2 )
      {
        v12 = 4;
      }
      else if ( v13 == 5 )
      {
        v12 = 6;
      }
    }
  }
  v14 = (*(__int64 (__fastcall **)(unsigned __int64, char **, __int64, _QWORD, __int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 1088LL))(
          v11,
          &v86,
          v2,
          v83[3],
          v7,
          v10,
          v85,
          v12);
  v15 = v14;
  v16 = v14;
  if ( v83[0] != v83[1] )
    v16 = sub_2D44750((__int64 *)&v86, v14, v83);
  v17 = sub_ACADE0(*(__int64 ***)(v2 + 8));
  v80 = 257;
  v77[0] = 0;
  v69 = v17;
  v18 = (__int64 *)(*(__int64 (__fastcall **)(__int64, __int64, __int64, _DWORD *, __int64))(*(_QWORD *)v91 + 88LL))(
                     v91,
                     v17,
                     v16,
                     v77,
                     1);
  if ( !v18 )
  {
    v82 = 257;
    v49 = sub_BD2C40(104, unk_3F148BC);
    v18 = v49;
    if ( v49 )
    {
      sub_B44260((__int64)v49, *(_QWORD *)(v69 + 8), 65, 2u, 0, 0);
      v18[9] = (__int64)(v18 + 11);
      v18[10] = 0x400000000LL;
      sub_B4FD20((__int64)v18, v69, v16, v77, 1, (__int64)v81);
    }
    (*(void (__fastcall **)(__int64, __int64 *, _QWORD *, __int64, __int64))(*(_QWORD *)v92 + 16LL))(
      v92,
      v18,
      v79,
      v89,
      v90);
    if ( v86 != &v86[16 * v87] )
    {
      v74 = v7;
      v50 = v86;
      v51 = &v86[16 * v87];
      do
      {
        v52 = *((_QWORD *)v50 + 1);
        v53 = *(_DWORD *)v50;
        v50 += 16;
        sub_B99FD0((__int64)v18, v53, v52);
      }
      while ( v51 != v50 );
      v7 = v74;
    }
  }
  v19 = v85;
  v78 = 257;
  v79[0] = "Success";
  v80 = 259;
  v20 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v91 + 16LL))(v91, 28, v15, v85);
  if ( !v20 )
  {
    v82 = 257;
    v72 = sub_B504D0(28, v15, v19, (__int64)v81, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _DWORD *, __int64, __int64))(*(_QWORD *)v92 + 16LL))(
      v92,
      v72,
      v77,
      v89,
      v90);
    v20 = v72;
    if ( v86 != &v86[16 * v87] )
    {
      v73 = v7;
      v44 = v86;
      v45 = &v86[16 * v87];
      v46 = v20;
      do
      {
        v47 = *((_QWORD *)v44 + 1);
        v48 = *(_DWORD *)v44;
        v44 += 16;
        sub_B99FD0(v46, v48, v47);
      }
      while ( v45 != v44 );
      v7 = v73;
      v20 = v46;
    }
  }
  v70 = v20;
  v21 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v91 + 56LL))(
                    v91,
                    32,
                    v7,
                    v20);
  if ( !v21 )
  {
    v82 = 257;
    v21 = sub_BD2C40(72, unk_3F10FD0);
    if ( v21 )
    {
      v36 = *(_QWORD ***)(v7 + 8);
      v37 = *((unsigned __int8 *)v36 + 8);
      if ( (unsigned int)(v37 - 17) > 1 )
      {
        v39 = sub_BCB2A0(*v36);
      }
      else
      {
        BYTE4(v76) = (_BYTE)v37 == 18;
        LODWORD(v76) = *((_DWORD *)v36 + 8);
        v38 = (__int64 *)sub_BCB2A0(*v36);
        v39 = sub_BCE1B0(v38, v76);
      }
      sub_B523C0((__int64)v21, v39, 53, 32, v7, v70, (__int64)v81, 0, 0, 0);
    }
    (*(void (__fastcall **)(__int64, _QWORD *, _QWORD *, __int64, __int64))(*(_QWORD *)v92 + 16LL))(
      v92,
      v21,
      v79,
      v89,
      v90);
    v40 = v86;
    v41 = &v86[16 * v87];
    if ( v86 != v41 )
    {
      do
      {
        v42 = *((_QWORD *)v40 + 1);
        v43 = *(_DWORD *)v40;
        v40 += 16;
        sub_B99FD0((__int64)v21, v43, v42);
      }
      while ( v41 != v40 );
    }
  }
  v80 = 257;
  v77[0] = 1;
  v22 = (*(__int64 (__fastcall **)(__int64, __int64 *, _QWORD *, _DWORD *, __int64))(*(_QWORD *)v91 + 88LL))(
          v91,
          v18,
          v21,
          v77,
          1);
  if ( !v22 )
  {
    v82 = 257;
    v31 = sub_BD2C40(104, unk_3F148BC);
    v22 = (__int64)v31;
    if ( v31 )
    {
      sub_B44260((__int64)v31, v18[1], 65, 2u, 0, 0);
      *(_QWORD *)(v22 + 72) = v22 + 88;
      *(_QWORD *)(v22 + 80) = 0x400000000LL;
      sub_B4FD20(v22, (__int64)v18, (__int64)v21, v77, 1, (__int64)v81);
    }
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v92 + 16LL))(
      v92,
      v22,
      v79,
      v89,
      v90);
    v32 = v86;
    v33 = &v86[16 * v87];
    if ( v86 != v33 )
    {
      do
      {
        v34 = *((_QWORD *)v32 + 1);
        v35 = *(_DWORD *)v32;
        v32 += 16;
        sub_B99FD0(v22, v35, v34);
      }
      while ( v33 != v32 );
    }
  }
  sub_BD84D0(v2, v22);
  sub_B43D60((_QWORD *)v2);
  sub_B32BF0(v95);
  v93 = &unk_49E5698;
  v94 = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  if ( v86 != &v88 )
    _libc_free((unsigned __int64)v86);
}
