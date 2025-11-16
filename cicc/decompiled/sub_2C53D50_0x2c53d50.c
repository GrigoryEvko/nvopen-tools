// Function: sub_2C53D50
// Address: 0x2c53d50
//
__int64 __fastcall sub_2C53D50(__int64 a1, __int64 a2)
{
  __int64 v4; // rcx
  int v5; // edx
  _DWORD *v6; // rax
  __int64 v7; // rsi
  _DWORD *v8; // rdi
  __int64 v9; // rcx
  _DWORD *v10; // rcx
  __int64 v11; // rcx
  __int64 v12; // r15
  __int64 v13; // rsi
  __int64 v14; // r13
  __int64 v15; // r14
  unsigned int v16; // r13d
  unsigned __int8 v17; // si
  unsigned __int64 v18; // rax
  char v19; // al
  __int64 v20; // rax
  __int64 v21; // rax
  int v22; // edx
  __int64 v23; // rax
  int v24; // edx
  bool v25; // cc
  __int64 v26; // rsi
  __int64 (__fastcall *v27)(__int64, unsigned __int64, __int64); // rax
  __int64 v28; // rdx
  _BYTE *v29; // r15
  unsigned __int64 v30; // rbx
  __int64 v31; // rdx
  unsigned int v32; // esi
  _QWORD *v33; // rax
  __int64 v34; // r15
  __int64 v35; // r14
  _BYTE *v36; // r14
  unsigned __int64 v37; // rbx
  __int64 v38; // rdx
  unsigned int v39; // esi
  __int64 v40; // r14
  __int64 i; // rbx
  __int64 v43; // rdx
  _BYTE *v44; // r15
  unsigned __int64 v45; // rbx
  __int64 v46; // rdx
  unsigned int v47; // esi
  __int64 v48; // [rsp+18h] [rbp-148h]
  int v49; // [rsp+20h] [rbp-140h]
  int v50; // [rsp+28h] [rbp-138h]
  __int64 v51; // [rsp+28h] [rbp-138h]
  unsigned __int16 v52; // [rsp+34h] [rbp-12Ch]
  char v53; // [rsp+34h] [rbp-12Ch]
  __int64 v54; // [rsp+38h] [rbp-128h]
  __int64 v55; // [rsp+38h] [rbp-128h]
  _BYTE v56[32]; // [rsp+40h] [rbp-120h] BYREF
  __int16 v57; // [rsp+60h] [rbp-100h]
  _BYTE v58[32]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v59; // [rsp+90h] [rbp-D0h]
  _BYTE *v60; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v61; // [rsp+A8h] [rbp-B8h]
  _BYTE v62[32]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v63; // [rsp+D0h] [rbp-90h]
  __int64 v64; // [rsp+D8h] [rbp-88h]
  __int64 v65; // [rsp+E0h] [rbp-80h]
  __int64 *v66; // [rsp+E8h] [rbp-78h]
  void **v67; // [rsp+F0h] [rbp-70h]
  void **v68; // [rsp+F8h] [rbp-68h]
  __int64 v69; // [rsp+100h] [rbp-60h]
  int v70; // [rsp+108h] [rbp-58h]
  __int16 v71; // [rsp+10Ch] [rbp-54h]
  char v72; // [rsp+10Eh] [rbp-52h]
  __int64 v73; // [rsp+110h] [rbp-50h]
  __int64 v74; // [rsp+118h] [rbp-48h]
  void *v75; // [rsp+120h] [rbp-40h] BYREF
  void *v76; // [rsp+128h] [rbp-38h] BYREF

  if ( !(unsigned __int8)sub_B4F540(a2) )
    return 0;
  v4 = 4LL * *(unsigned int *)(a2 + 80);
  v5 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a2 - 64) + 8LL) + 32LL);
  v6 = *(_DWORD **)(a2 + 72);
  v7 = v4 >> 2;
  v8 = &v6[(unsigned __int64)v4 / 4];
  v9 = v4 >> 4;
  if ( v9 )
  {
    v10 = &v6[4 * v9];
    while ( *v6 < v5 )
    {
      if ( v6[1] >= v5 )
      {
        v11 = 32LL * (v8 != v6 + 1) - 64;
        goto LABEL_10;
      }
      if ( v6[2] >= v5 )
      {
        v11 = 32LL * (v8 != v6 + 2) - 64;
        goto LABEL_10;
      }
      if ( v6[3] >= v5 )
      {
        v11 = 32LL * (v8 != v6 + 3) - 64;
        goto LABEL_10;
      }
      v6 += 4;
      if ( v10 == v6 )
      {
        v7 = v8 - v6;
        goto LABEL_49;
      }
    }
    goto LABEL_9;
  }
LABEL_49:
  if ( v7 != 2 )
  {
    if ( v7 != 3 )
    {
      v11 = -64;
      if ( v7 != 1 )
        goto LABEL_10;
LABEL_52:
      v11 = -64;
      if ( *v6 < v5 )
        goto LABEL_10;
      goto LABEL_53;
    }
    if ( *v6 >= v5 )
    {
LABEL_9:
      v11 = 32LL * (v6 != v8) - 64;
      goto LABEL_10;
    }
    ++v6;
  }
  if ( *v6 < v5 )
  {
    ++v6;
    goto LABEL_52;
  }
LABEL_53:
  v11 = 32LL * (v8 != v6) - 64;
LABEL_10:
  v12 = *(_QWORD *)(a2 + v11);
  v13 = *(_QWORD *)(a1 + 152);
  if ( *(_BYTE *)v12 != 61 )
    v12 = 0;
  if ( !(unsigned __int8)sub_2C4D8C0(v12, v13) )
    return 0;
  v14 = *(_QWORD *)(a2 + 8);
  v54 = v14;
  v15 = (__int64)sub_BD3990(*(unsigned __int8 **)(v12 - 32), v13);
  v52 = *(_WORD *)(v12 + 2);
  v16 = sub_D31180(v15, v14, 0, *(_QWORD *)(a1 + 184), v12, *(_QWORD *)(a1 + 176), *(__int64 **)(a1 + 160), 0);
  if ( !(_BYTE)v16 )
    return 0;
  v17 = sub_BD5420((unsigned __int8 *)v15, *(_QWORD *)(a1 + 184));
  _BitScanReverse64(&v18, 1LL << (v52 >> 1));
  v25 = v17 <= (unsigned __int8)(63 - (v18 ^ 0x3F));
  v19 = 63 - (v18 ^ 0x3F);
  if ( !v25 )
    v19 = v17;
  v53 = v19;
  v20 = *(_QWORD *)(*(_QWORD *)(v12 - 32) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v20 + 8) - 17 <= 1 )
    v20 = **(_QWORD **)(v20 + 16);
  v49 = *(_DWORD *)(v20 + 8) >> 8;
  v21 = sub_DFD4A0(*(__int64 **)(a1 + 152));
  v50 = v22;
  v48 = v21;
  v23 = sub_DFD4A0(*(__int64 **)(a1 + 152));
  v25 = v50 < v24;
  if ( v50 == v24 )
    v25 = v48 < v23;
  if ( v25 || v24 )
    return 0;
  v66 = (__int64 *)sub_BD5C60(v12);
  v67 = &v75;
  v68 = &v76;
  v60 = v62;
  v75 = &unk_49DA100;
  v61 = 0x200000000LL;
  v71 = 512;
  LOWORD(v65) = 0;
  v76 = &unk_49DA0B0;
  v69 = 0;
  v70 = 0;
  v72 = 7;
  v73 = 0;
  v74 = 0;
  v63 = 0;
  v64 = 0;
  sub_D5F1F0((__int64)&v60, v12);
  v57 = 257;
  v26 = sub_BCE3C0(v66, v49);
  if ( v26 == *(_QWORD *)(v15 + 8) )
    goto LABEL_31;
  if ( *(_BYTE *)v15 > 0x15u )
  {
    v59 = 257;
    v15 = sub_B52190(v15, v26, (__int64)v58, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v68 + 2))(v68, v15, v56, v64, v65);
    v43 = 16LL * (unsigned int)v61;
    v44 = &v60[v43];
    if ( v60 == &v60[v43] )
      goto LABEL_31;
    v51 = a1;
    v45 = (unsigned __int64)v60;
    do
    {
      v46 = *(_QWORD *)(v45 + 8);
      v47 = *(_DWORD *)v45;
      v45 += 16LL;
      sub_B99FD0(v15, v47, v46);
    }
    while ( v44 != (_BYTE *)v45 );
  }
  else
  {
    v27 = (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))*((_QWORD *)*v67 + 18);
    if ( v27 == sub_B32D70 )
      v15 = sub_ADB060(v15, v26);
    else
      v15 = v27((__int64)v67, v15, v26);
    if ( *(_BYTE *)v15 <= 0x1Cu )
      goto LABEL_31;
    (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v68 + 2))(v68, v15, v56, v64, v65);
    v28 = 16LL * (unsigned int)v61;
    v29 = &v60[v28];
    if ( v60 == &v60[v28] )
      goto LABEL_31;
    v51 = a1;
    v30 = (unsigned __int64)v60;
    do
    {
      v31 = *(_QWORD *)(v30 + 8);
      v32 = *(_DWORD *)v30;
      v30 += 16LL;
      sub_B99FD0(v15, v32, v31);
    }
    while ( v29 != (_BYTE *)v30 );
  }
  a1 = v51;
LABEL_31:
  v59 = 257;
  v57 = 257;
  v33 = sub_BD2C40(80, 1u);
  v34 = (__int64)v33;
  if ( v33 )
    sub_B4D190((__int64)v33, v54, v15, (__int64)v58, 0, v53, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v68 + 2))(v68, v34, v56, v64, v65);
  v35 = 16LL * (unsigned int)v61;
  if ( v60 != &v60[v35] )
  {
    v55 = a1;
    v36 = &v60[v35];
    v37 = (unsigned __int64)v60;
    do
    {
      v38 = *(_QWORD *)(v37 + 8);
      v39 = *(_DWORD *)v37;
      v37 += 16LL;
      sub_B99FD0(v34, v39, v38);
    }
    while ( v36 != (_BYTE *)v37 );
    a1 = v55;
  }
  v40 = a1 + 200;
  sub_BD84D0(a2, v34);
  if ( *(_BYTE *)v34 > 0x1Cu )
  {
    sub_BD6B90((unsigned __int8 *)v34, (unsigned __int8 *)a2);
    for ( i = *(_QWORD *)(v34 + 16); i; i = *(_QWORD *)(i + 8) )
      sub_F15FC0(v40, *(_QWORD *)(i + 24));
    if ( *(_BYTE *)v34 > 0x1Cu )
      sub_F15FC0(v40, v34);
  }
  if ( *(_BYTE *)a2 > 0x1Cu )
    sub_F15FC0(v40, a2);
  nullsub_61();
  v75 = &unk_49DA100;
  nullsub_63();
  if ( v60 != v62 )
    _libc_free((unsigned __int64)v60);
  return v16;
}
