// Function: sub_31A1E80
// Address: 0x31a1e80
//
void __fastcall sub_31A1E80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5, char a6)
{
  _QWORD *v6; // r12
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // rax
  unsigned __int64 v10; // rsi
  int v11; // eax
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // r15
  _QWORD *v15; // rax
  __int64 v16; // r9
  __int64 v17; // r13
  unsigned int *v18; // r15
  unsigned int *v19; // rbx
  __int64 v20; // rdx
  unsigned int v21; // esi
  unsigned __int64 v22; // rdi
  int v23; // eax
  _QWORD *v24; // rdi
  char v25; // r13
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned int v28; // eax
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 v32; // r15
  int v33; // eax
  int v34; // eax
  unsigned int v35; // edx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdx
  char v39; // r15
  __int64 v40; // rsi
  __int64 v41; // r12
  __int64 v42; // r9
  _QWORD *v43; // r13
  unsigned int *v44; // r15
  unsigned int *v45; // r12
  __int64 v46; // rdx
  unsigned int v47; // esi
  _BYTE *v48; // rax
  __int64 v49; // rdx
  int v50; // eax
  int v51; // eax
  unsigned int v52; // ecx
  __int64 v53; // rax
  __int64 v54; // rcx
  __int64 v55; // rcx
  __int64 v56; // rax
  __int64 v57; // r12
  _QWORD *v58; // rax
  __int64 v59; // r9
  __int64 v60; // rbx
  unsigned int *v61; // r13
  unsigned int *v62; // r12
  __int64 v63; // rdx
  unsigned int v64; // esi
  __int64 v65; // [rsp-8h] [rbp-238h]
  __int64 v66; // [rsp-8h] [rbp-238h]
  char v68; // [rsp+1Ch] [rbp-214h]
  __int64 v71; // [rsp+38h] [rbp-1F8h]
  __int64 v73; // [rsp+48h] [rbp-1E8h]
  __int64 v75; // [rsp+58h] [rbp-1D8h]
  __int64 v76; // [rsp+60h] [rbp-1D0h]
  __int64 v77; // [rsp+60h] [rbp-1D0h]
  _BYTE *v78; // [rsp+78h] [rbp-1B8h] BYREF
  _BYTE v79[32]; // [rsp+80h] [rbp-1B0h] BYREF
  __int16 v80; // [rsp+A0h] [rbp-190h]
  _BYTE v81[32]; // [rsp+B0h] [rbp-180h] BYREF
  __int16 v82; // [rsp+D0h] [rbp-160h]
  unsigned int *v83; // [rsp+E0h] [rbp-150h] BYREF
  int v84; // [rsp+E8h] [rbp-148h]
  char v85; // [rsp+F0h] [rbp-140h] BYREF
  __int64 v86; // [rsp+118h] [rbp-118h]
  __int64 v87; // [rsp+120h] [rbp-110h]
  __int64 v88; // [rsp+138h] [rbp-F8h]
  void *v89; // [rsp+160h] [rbp-D0h]
  char *v90; // [rsp+170h] [rbp-C0h] BYREF
  __int64 v91; // [rsp+178h] [rbp-B8h]
  _BYTE v92[16]; // [rsp+180h] [rbp-B0h] BYREF
  __int16 v93; // [rsp+190h] [rbp-A0h]
  __int64 v94; // [rsp+1A0h] [rbp-90h]
  __int64 v95; // [rsp+1A8h] [rbp-88h]
  __int64 v96; // [rsp+1B0h] [rbp-80h]
  __int64 v97; // [rsp+1B8h] [rbp-78h]
  void **v98; // [rsp+1C0h] [rbp-70h]
  void **v99; // [rsp+1C8h] [rbp-68h]
  __int64 v100; // [rsp+1D0h] [rbp-60h]
  int v101; // [rsp+1D8h] [rbp-58h]
  __int16 v102; // [rsp+1DCh] [rbp-54h]
  char v103; // [rsp+1DEh] [rbp-52h]
  __int64 v104; // [rsp+1E0h] [rbp-50h]
  __int64 v105; // [rsp+1E8h] [rbp-48h]
  void *v106; // [rsp+1F0h] [rbp-40h] BYREF
  void *v107; // [rsp+1F8h] [rbp-38h] BYREF

  v6 = *(_QWORD **)(a1 + 40);
  v7 = v6[9];
  v76 = *(_QWORD *)(a3 + 8);
  v71 = sub_B2BEC0(v7);
  v90 = "split";
  v93 = 259;
  v73 = sub_AA8550(v6, (__int64 *)(a1 + 24), 0, (__int64)&v90, 0);
  v90 = "loadstoreloop";
  v93 = 259;
  v8 = sub_B2BE50(v7);
  v9 = sub_22077B0(0x50u);
  v75 = v9;
  if ( v9 )
    sub_AA4D50(v9, v8, (__int64)&v90, v7, v73);
  v10 = v6[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6 + 6 == (_QWORD *)v10 )
  {
    v12 = 0;
  }
  else
  {
    if ( !v10 )
      BUG();
    v11 = *(unsigned __int8 *)(v10 - 24);
    v12 = v10 - 24;
    if ( (unsigned int)(v11 - 30) >= 0xB )
      v12 = 0;
  }
  sub_23D0AB0((__int64)&v83, v12, 0, 0, 0);
  v82 = 257;
  v13 = sub_AD64C0(v76, 0, 0);
  v14 = sub_92B530(&v83, 0x20u, v13, (_BYTE *)a3, (__int64)v81);
  v93 = 257;
  v15 = sub_BD2C40(72, 3u);
  v17 = (__int64)v15;
  if ( v15 )
  {
    sub_B4C9A0((__int64)v15, v73, v75, v14, 3u, 0, 0, 0);
    v16 = v65;
  }
  (*(void (__fastcall **)(__int64, __int64, char **, __int64, __int64, __int64))(*(_QWORD *)v88 + 16LL))(
    v88,
    v17,
    &v90,
    v86,
    v87,
    v16);
  v18 = v83;
  v19 = &v83[4 * v84];
  if ( v83 != v19 )
  {
    do
    {
      v20 = *((_QWORD *)v18 + 1);
      v21 = *v18;
      v18 += 4;
      sub_B99FD0(v17, v21, v20);
    }
    while ( v19 != v18 );
  }
  v22 = v6[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v22 == v6 + 6 )
  {
    v24 = 0;
  }
  else
  {
    if ( !v22 )
      BUG();
    v23 = *(unsigned __int8 *)(v22 - 24);
    v24 = (_QWORD *)(v22 - 24);
    if ( (unsigned int)(v23 - 30) >= 0xB )
      v24 = 0;
  }
  sub_B43D60(v24);
  v25 = -1;
  v26 = sub_9208B0(v71, *(_QWORD *)(a4 + 8));
  v91 = v27;
  v90 = (char *)((unsigned __int64)(v26 + 7) >> 3);
  v28 = sub_CA1930(&v90);
  v29 = -(__int64)(v28 | (unsigned __int64)(1LL << a5)) & (v28 | (unsigned __int64)(1LL << a5));
  if ( v29 )
  {
    _BitScanReverse64(&v29, v29);
    v25 = 63 - (v29 ^ 0x3F);
  }
  v30 = sub_AA48A0(v75);
  v103 = 7;
  v97 = v30;
  v98 = &v106;
  v99 = &v107;
  v90 = v92;
  v106 = &unk_49DA100;
  v91 = 0x200000000LL;
  LOWORD(v96) = 0;
  v102 = 512;
  v107 = &unk_49DA0B0;
  v82 = 257;
  v94 = v75;
  v100 = 0;
  v101 = 0;
  v104 = 0;
  v105 = 0;
  v95 = v75 + 48;
  v31 = sub_D5C860((__int64 *)&v90, v76, 0, (__int64)v81);
  v32 = sub_AD64C0(v76, 0, 0);
  v33 = *(_DWORD *)(v31 + 4) & 0x7FFFFFF;
  if ( v33 == *(_DWORD *)(v31 + 72) )
  {
    sub_B48D90(v31);
    v33 = *(_DWORD *)(v31 + 4) & 0x7FFFFFF;
  }
  v34 = (v33 + 1) & 0x7FFFFFF;
  v35 = v34 | *(_DWORD *)(v31 + 4) & 0xF8000000;
  v36 = *(_QWORD *)(v31 - 8) + 32LL * (unsigned int)(v34 - 1);
  *(_DWORD *)(v31 + 4) = v35;
  if ( *(_QWORD *)v36 )
  {
    v37 = *(_QWORD *)(v36 + 8);
    **(_QWORD **)(v36 + 16) = v37;
    if ( v37 )
      *(_QWORD *)(v37 + 16) = *(_QWORD *)(v36 + 16);
  }
  *(_QWORD *)v36 = v32;
  if ( v32 )
  {
    v38 = *(_QWORD *)(v32 + 16);
    *(_QWORD *)(v36 + 8) = v38;
    if ( v38 )
      *(_QWORD *)(v38 + 16) = v36 + 8;
    *(_QWORD *)(v36 + 16) = v32 + 16;
    *(_QWORD *)(v32 + 16) = v36;
  }
  v39 = a6;
  *(_QWORD *)(*(_QWORD *)(v31 - 8) + 32LL * *(unsigned int *)(v31 + 72)
                                   + 8LL * ((*(_DWORD *)(v31 + 4) & 0x7FFFFFFu) - 1)) = v6;
  v80 = 257;
  v40 = *(_QWORD *)(a4 + 8);
  v78 = (_BYTE *)v31;
  v41 = sub_921130((unsigned int **)&v90, v40, a2, &v78, 1, (__int64)v79, 3u);
  v82 = 257;
  v68 = v25;
  v43 = sub_BD2C40(80, unk_3F10A10);
  if ( v43 )
  {
    sub_B4D3C0((__int64)v43, a4, v41, v39, v68, v42, 0, 0);
    v42 = v66;
  }
  (*((void (__fastcall **)(void **, _QWORD *, _BYTE *, __int64, __int64, __int64))*v99 + 2))(
    v99,
    v43,
    v81,
    v95,
    v96,
    v42);
  v44 = (unsigned int *)v90;
  v45 = (unsigned int *)&v90[16 * (unsigned int)v91];
  if ( v90 != (char *)v45 )
  {
    do
    {
      v46 = *((_QWORD *)v44 + 1);
      v47 = *v44;
      v44 += 4;
      sub_B99FD0((__int64)v43, v47, v46);
    }
    while ( v45 != v44 );
  }
  v82 = 257;
  v48 = (_BYTE *)sub_AD64C0(v76, 1, 0);
  v49 = sub_929C50((unsigned int **)&v90, (_BYTE *)v31, v48, (__int64)v81, 0, 0);
  v50 = *(_DWORD *)(v31 + 4) & 0x7FFFFFF;
  if ( v50 == *(_DWORD *)(v31 + 72) )
  {
    v77 = v49;
    sub_B48D90(v31);
    v49 = v77;
    v50 = *(_DWORD *)(v31 + 4) & 0x7FFFFFF;
  }
  v51 = (v50 + 1) & 0x7FFFFFF;
  v52 = v51 | *(_DWORD *)(v31 + 4) & 0xF8000000;
  v53 = *(_QWORD *)(v31 - 8) + 32LL * (unsigned int)(v51 - 1);
  *(_DWORD *)(v31 + 4) = v52;
  if ( *(_QWORD *)v53 )
  {
    v54 = *(_QWORD *)(v53 + 8);
    **(_QWORD **)(v53 + 16) = v54;
    if ( v54 )
      *(_QWORD *)(v54 + 16) = *(_QWORD *)(v53 + 16);
  }
  *(_QWORD *)v53 = v49;
  if ( v49 )
  {
    v55 = *(_QWORD *)(v49 + 16);
    *(_QWORD *)(v53 + 8) = v55;
    if ( v55 )
      *(_QWORD *)(v55 + 16) = v53 + 8;
    *(_QWORD *)(v53 + 16) = v49 + 16;
    *(_QWORD *)(v49 + 16) = v53;
  }
  *(_QWORD *)(*(_QWORD *)(v31 - 8) + 32LL * *(unsigned int *)(v31 + 72)
                                   + 8LL * ((*(_DWORD *)(v31 + 4) & 0x7FFFFFFu) - 1)) = v75;
  v80 = 257;
  v56 = sub_92B530((unsigned int **)&v90, 0x24u, v49, (_BYTE *)a3, (__int64)v79);
  v82 = 257;
  v57 = v56;
  v58 = sub_BD2C40(72, 3u);
  v60 = (__int64)v58;
  if ( v58 )
    sub_B4C9A0((__int64)v58, v75, v73, v57, 3u, v59, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v99 + 2))(v99, v60, v81, v95, v96);
  v61 = (unsigned int *)v90;
  v62 = (unsigned int *)&v90[16 * (unsigned int)v91];
  if ( v90 != (char *)v62 )
  {
    do
    {
      v63 = *((_QWORD *)v61 + 1);
      v64 = *v61;
      v61 += 4;
      sub_B99FD0(v60, v64, v63);
    }
    while ( v62 != v61 );
  }
  nullsub_61();
  v106 = &unk_49DA100;
  nullsub_63();
  if ( v90 != v92 )
    _libc_free((unsigned __int64)v90);
  nullsub_61();
  v89 = &unk_49DA100;
  nullsub_63();
  if ( v83 != (unsigned int *)&v85 )
    _libc_free((unsigned __int64)v83);
}
