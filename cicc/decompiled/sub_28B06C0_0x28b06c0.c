// Function: sub_28B06C0
// Address: 0x28b06c0
//
__int64 __fastcall sub_28B06C0(_QWORD *a1, __int64 a2, unsigned int a3, __m128i a4, __m128i a5, __m128i a6)
{
  __int64 v8; // r13
  __int64 v9; // r14
  char v10; // r15
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  unsigned __int64 v13; // r8
  __int64 v14; // rax
  unsigned __int64 v15; // r8
  unsigned __int64 v16; // r15
  __int64 v17; // rdi
  int v18; // ecx
  __int64 v19; // rsi
  int v20; // ecx
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // r11
  _BYTE *v24; // r14
  __int64 v25; // rcx
  __int64 *v26; // rax
  __int64 *v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // r9
  _QWORD *v30; // rdi
  __int64 *v31; // rax
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // r11
  unsigned int v35; // r12d
  __int64 v37; // rax
  int v38; // eax
  __int64 v39; // rax
  __int64 v40; // rdi
  int v41; // eax
  bool v42; // al
  __int64 v43; // rax
  unsigned __int64 v44; // rax
  __int16 v45; // ax
  unsigned int v46; // ecx
  unsigned int v47; // r15d
  __int64 v48; // rsi
  unsigned __int16 v49; // ax
  __int64 v50; // r11
  unsigned __int8 *v51; // rdi
  unsigned __int8 *v52; // rax
  unsigned __int8 v53; // al
  __int64 v54; // rax
  int v55; // ecx
  __int64 v56; // rsi
  int v57; // ecx
  unsigned int v58; // edx
  __int64 *v59; // rax
  __int64 v60; // r8
  __int64 v61; // r13
  __int64 v62; // r8
  __int64 v63; // r9
  unsigned __int8 *v64; // rax
  int v65; // r8d
  int v66; // eax
  int v67; // edi
  int v68; // [rsp+10h] [rbp-380h]
  __int64 v69; // [rsp+10h] [rbp-380h]
  unsigned __int8 v70; // [rsp+10h] [rbp-380h]
  __int64 v71; // [rsp+18h] [rbp-378h]
  unsigned __int8 *v72; // [rsp+18h] [rbp-378h]
  __int64 v73; // [rsp+18h] [rbp-378h]
  __int64 v74; // [rsp+18h] [rbp-378h]
  __int64 v75; // [rsp+20h] [rbp-370h]
  _QWORD *v76; // [rsp+28h] [rbp-368h]
  __int64 v77; // [rsp+28h] [rbp-368h]
  __int64 v78; // [rsp+28h] [rbp-368h]
  char v79; // [rsp+30h] [rbp-360h]
  unsigned __int8 v80; // [rsp+30h] [rbp-360h]
  __int64 v81; // [rsp+30h] [rbp-360h]
  __int64 v82; // [rsp+30h] [rbp-360h]
  __int64 v83; // [rsp+40h] [rbp-350h]
  _QWORD v85[6]; // [rsp+50h] [rbp-340h] BYREF
  __m128i v86; // [rsp+80h] [rbp-310h] BYREF
  _QWORD *v87; // [rsp+B0h] [rbp-2E0h] BYREF
  _QWORD *v88; // [rsp+B8h] [rbp-2D8h]
  __int64 v89; // [rsp+C0h] [rbp-2D0h]
  __int64 v90; // [rsp+C8h] [rbp-2C8h]
  __int64 v91; // [rsp+D0h] [rbp-2C0h] BYREF
  unsigned int v92; // [rsp+D8h] [rbp-2B8h]
  _QWORD v93[2]; // [rsp+210h] [rbp-180h] BYREF
  char v94; // [rsp+220h] [rbp-170h]
  _BYTE *v95; // [rsp+228h] [rbp-168h]
  __int64 v96; // [rsp+230h] [rbp-160h]
  _BYTE v97[128]; // [rsp+238h] [rbp-158h] BYREF
  __int16 v98; // [rsp+2B8h] [rbp-D8h]
  void *v99; // [rsp+2C0h] [rbp-D0h]
  __int64 v100; // [rsp+2C8h] [rbp-C8h]
  __int64 v101; // [rsp+2D0h] [rbp-C0h]
  __int64 v102; // [rsp+2D8h] [rbp-B8h] BYREF
  unsigned int v103; // [rsp+2E0h] [rbp-B0h]
  char v104; // [rsp+358h] [rbp-38h] BYREF

  v8 = sub_B43CC0(a2);
  v75 = a3;
  v83 = *(_QWORD *)(a2 + 32 * (a3 - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v76 = (_QWORD *)(a2 + 72);
  v9 = sub_A748A0((_QWORD *)(a2 + 72), a3);
  if ( !v9 )
  {
    v37 = *(_QWORD *)(a2 - 32);
    if ( v37 )
    {
      if ( !*(_BYTE *)v37 && *(_QWORD *)(v37 + 24) == *(_QWORD *)(a2 + 80) )
      {
        v87 = *(_QWORD **)(v37 + 120);
        v9 = sub_A748A0(&v87, a3);
      }
    }
  }
  v10 = sub_AE5020(v8, v9);
  v11 = (_QWORD *)sub_9208B0(v8, v9);
  v88 = v12;
  v87 = v11;
  v79 = (char)v12;
  v13 = (1LL << v10) + (((unsigned __int64)v11 + 7) >> 3) - 1;
  v14 = 0xBFFFFFFFFFFFFFFELL;
  v15 = v13 >> v10 << v10;
  v16 = v15;
  if ( v15 <= 0x3FFFFFFFFFFFFFFBLL )
  {
    v14 = v15;
    if ( (_BYTE)v12 )
      v14 = v15 | 0x4000000000000000LL;
  }
  v85[1] = v14;
  memset(&v85[2], 0, 32);
  v17 = a1[5];
  v85[0] = v83;
  v18 = *(_DWORD *)(v17 + 56);
  v19 = *(_QWORD *)(v17 + 40);
  if ( !v18 )
    return 0;
  v20 = v18 - 1;
  v21 = v20 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v22 = (__int64 *)(v19 + 16LL * v21);
  v23 = *v22;
  if ( a2 != *v22 )
  {
    v38 = 1;
    while ( v23 != -4096 )
    {
      v65 = v38 + 1;
      v21 = v20 & (v38 + v21);
      v22 = (__int64 *)(v19 + 16LL * v21);
      v23 = *v22;
      if ( a2 == *v22 )
        goto LABEL_7;
      v38 = v65;
    }
    return 0;
  }
LABEL_7:
  v24 = (_BYTE *)v22[1];
  if ( !v24 )
    return 0;
  v89 = 0;
  v90 = 1;
  v25 = a1[7];
  v87 = (_QWORD *)a1[1];
  v88 = v87;
  v26 = &v91;
  do
  {
    *v26 = -4;
    v26 += 5;
    *(v26 - 4) = -3;
    *(v26 - 3) = -4;
    *(v26 - 2) = -3;
  }
  while ( v26 != v93 );
  v93[0] = v25;
  v95 = v97;
  v96 = 0x400000000LL;
  v98 = 256;
  v93[1] = 0;
  v94 = 0;
  v100 = 0;
  v101 = 1;
  v99 = &unk_49DDBE8;
  v27 = &v102;
  do
  {
    *v27 = -4096;
    v27 += 2;
  }
  while ( v27 != (__int64 *)&v104 );
  v28 = sub_103E0E0((_QWORD *)v17);
  v29 = *v28;
  v30 = v28;
  v31 = (__int64 *)(v24 - 64);
  if ( *v24 == 26 )
    v31 = (__int64 *)(v24 - 32);
  v32 = *v31;
  v33 = (*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD *))(v29 + 24))(v30, *v31, v85);
  if ( *(_BYTE *)v33 != 27 )
    goto LABEL_17;
  v34 = *(_QWORD *)(v33 + 72);
  if ( !v34 )
    goto LABEL_17;
  if ( *(_BYTE *)v34 != 85 )
    goto LABEL_17;
  v39 = *(_QWORD *)(v34 - 32);
  if ( !v39
    || *(_BYTE *)v39
    || *(_QWORD *)(v39 + 24) != *(_QWORD *)(v34 + 80)
    || (*(_BYTE *)(v39 + 33) & 0x20) == 0
    || ((*(_DWORD *)(v39 + 36) - 238) & 0xFFFFFFFD) != 0 )
  {
    goto LABEL_17;
  }
  v40 = *(_QWORD *)(v34 + 32 * (3LL - (*(_DWORD *)(v34 + 4) & 0x7FFFFFF)));
  if ( *(_DWORD *)(v40 + 32) <= 0x40u )
  {
    v42 = *(_QWORD *)(v40 + 24) == 0;
  }
  else
  {
    v68 = *(_DWORD *)(v40 + 32);
    v71 = v34;
    v41 = sub_C444A0(v40 + 24);
    v34 = v71;
    v42 = v68 == v41;
  }
  if ( !v42 )
    goto LABEL_17;
  v69 = v34;
  v72 = sub_BD3990((unsigned __int8 *)v83, v32);
  if ( v72 != sub_BD3990(*(unsigned __int8 **)(v69 - 32LL * (*(_DWORD *)(v69 + 4) & 0x7FFFFFF)), v32) )
    goto LABEL_17;
  v43 = *(_QWORD *)(v69 + 32 * (2LL - (*(_DWORD *)(v69 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v43 != 17 )
    goto LABEL_17;
  v44 = *(_DWORD *)(v43 + 32) <= 0x40u ? *(_QWORD *)(v43 + 24) : **(_QWORD **)(v43 + 24);
  if ( v44 < v16 )
    goto LABEL_17;
  if ( v79 == 1 )
    goto LABEL_17;
  v73 = v69;
  v45 = sub_A74840(v76, a3);
  v80 = v45;
  v35 = HIBYTE(v45);
  v46 = (unsigned __int8)v45;
  BYTE1(v46) = HIBYTE(v45);
  v47 = v46;
  if ( !HIBYTE(v45) )
    goto LABEL_17;
  v48 = 1;
  v77 = v69;
  v49 = sub_A74840((_QWORD *)(v69 + 72), 1);
  v50 = v69;
  if ( !HIBYTE(v49) || v80 > (unsigned __int8)v49 )
  {
    v70 = v80;
    v81 = v73;
    v74 = a1[3];
    v51 = *(unsigned __int8 **)(v77 + 32 * (1LL - (*(_DWORD *)(v77 + 4) & 0x7FFFFFF)));
    v78 = a1[2];
    v52 = sub_BD3990(v51, 1);
    v48 = v47;
    v53 = sub_F518D0(v52, v47, v8, a2, v78, v74);
    v50 = v81;
    if ( v53 < v70 )
      goto LABEL_17;
  }
  v82 = v50;
  if ( *(_QWORD *)(v83 + 8) != *((_QWORD *)sub_BD3990(
                                             *(unsigned __int8 **)(v50 + 32 * (1LL - (*(_DWORD *)(v50 + 4) & 0x7FFFFFF))),
                                             v48)
                               + 1) )
    goto LABEL_17;
  v54 = a1[5];
  v55 = *(_DWORD *)(v54 + 56);
  v56 = *(_QWORD *)(v54 + 40);
  if ( v55 )
  {
    v57 = v55 - 1;
    v58 = v57 & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
    v59 = (__int64 *)(v56 + 16LL * v58);
    v60 = *v59;
    if ( v82 == *v59 )
    {
LABEL_53:
      v61 = v59[1];
      goto LABEL_54;
    }
    v66 = 1;
    while ( v60 != -4096 )
    {
      v67 = v66 + 1;
      v58 = v57 & (v66 + v58);
      v59 = (__int64 *)(v56 + 16LL * v58);
      v60 = *v59;
      if ( v82 == *v59 )
        goto LABEL_53;
      v66 = v67;
    }
  }
  v61 = 0;
LABEL_54:
  sub_D671D0(&v86, v82);
  if ( !sub_28A97D0((_QWORD *)a1[5], &v87, v61, (__int64)v24, v62, v63, a4, a5, a6) )
  {
    sub_F57040((_BYTE *)a2, v82);
    v64 = sub_28B06A0(v82, v82);
    sub_AC2B30(a2 + 32 * (v75 - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), (__int64)v64);
    goto LABEL_18;
  }
LABEL_17:
  v35 = 0;
LABEL_18:
  v99 = &unk_49DDBE8;
  if ( (v101 & 1) == 0 )
    sub_C7D6A0(v102, 16LL * v103, 8);
  nullsub_184();
  if ( v95 != v97 )
    _libc_free((unsigned __int64)v95);
  if ( (v90 & 1) == 0 )
    sub_C7D6A0(v91, 40LL * v92, 8);
  return v35;
}
