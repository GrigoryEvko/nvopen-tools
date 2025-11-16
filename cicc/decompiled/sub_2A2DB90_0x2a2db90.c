// Function: sub_2A2DB90
// Address: 0x2a2db90
//
__int64 __fastcall sub_2A2DB90(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rsi
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // r12
  unsigned __int64 v6; // rax
  int v7; // ecx
  _QWORD *v8; // rdx
  __int64 v9; // rax
  char v10; // al
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rax
  char v15; // bl
  _QWORD *v16; // rax
  __int64 v17; // r12
  unsigned __int64 v18; // r15
  _BYTE *v19; // rbx
  __int64 v20; // rdx
  unsigned int v21; // esi
  __int64 v22; // rax
  char v23; // al
  _QWORD *v24; // rax
  __int64 v25; // r9
  __int64 v26; // rbx
  unsigned __int64 v27; // r15
  _BYTE *v28; // r14
  __int64 v29; // rdx
  unsigned int v30; // esi
  unsigned __int64 v32; // rsi
  unsigned __int8 *v33; // [rsp+20h] [rbp-140h]
  char v34; // [rsp+20h] [rbp-140h]
  __int64 v35; // [rsp+30h] [rbp-130h]
  __int64 v36; // [rsp+38h] [rbp-128h]
  _BYTE v37[32]; // [rsp+40h] [rbp-120h] BYREF
  __int16 v38; // [rsp+60h] [rbp-100h]
  _QWORD v39[4]; // [rsp+70h] [rbp-F0h] BYREF
  __int16 v40; // [rsp+90h] [rbp-D0h]
  _BYTE *v41; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v42; // [rsp+A8h] [rbp-B8h]
  _BYTE v43[32]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v44; // [rsp+D0h] [rbp-90h]
  __int64 v45; // [rsp+D8h] [rbp-88h]
  __int64 v46; // [rsp+E0h] [rbp-80h]
  __int64 v47; // [rsp+E8h] [rbp-78h]
  void **v48; // [rsp+F0h] [rbp-70h]
  void **v49; // [rsp+F8h] [rbp-68h]
  __int64 v50; // [rsp+100h] [rbp-60h]
  int v51; // [rsp+108h] [rbp-58h]
  __int16 v52; // [rsp+10Ch] [rbp-54h]
  char v53; // [rsp+10Eh] [rbp-52h]
  __int64 v54; // [rsp+110h] [rbp-50h]
  __int64 v55; // [rsp+118h] [rbp-48h]
  void *v56; // [rsp+120h] [rbp-40h] BYREF
  void *v57; // [rsp+128h] [rbp-38h] BYREF

  v47 = sub_BD5C60(a1);
  v48 = &v56;
  v49 = &v57;
  v41 = v43;
  v56 = &unk_49DA100;
  v42 = 0x200000000LL;
  v50 = 0;
  v57 = &unk_49DA0B0;
  v1 = *(_QWORD *)(a1 + 40);
  v51 = 0;
  v44 = v1;
  v52 = 512;
  v53 = 7;
  v54 = 0;
  v55 = 0;
  v45 = a1 + 24;
  LOWORD(v46) = 0;
  v2 = *(_QWORD *)sub_B46C60(a1);
  v39[0] = v2;
  if ( v2 && (sub_B96E90((__int64)v39, v2, 1), (v5 = v39[0]) != 0) )
  {
    v6 = (unsigned __int64)v41;
    v7 = v42;
    v8 = &v41[16 * (unsigned int)v42];
    if ( v41 != (_BYTE *)v8 )
    {
      while ( *(_DWORD *)v6 )
      {
        v6 += 16LL;
        if ( v8 == (_QWORD *)v6 )
          goto LABEL_23;
      }
      *(_QWORD *)(v6 + 8) = v39[0];
      goto LABEL_8;
    }
LABEL_23:
    if ( (unsigned int)v42 >= (unsigned __int64)HIDWORD(v42) )
    {
      v32 = (unsigned int)v42 + 1LL;
      if ( HIDWORD(v42) < v32 )
      {
        sub_C8D5F0((__int64)&v41, v43, v32, 0x10u, v3, v4);
        v8 = &v41[16 * (unsigned int)v42];
      }
      *v8 = 0;
      v8[1] = v5;
      v5 = v39[0];
      LODWORD(v42) = v42 + 1;
    }
    else
    {
      if ( v8 )
      {
        *(_DWORD *)v8 = 0;
        v8[1] = v5;
        v7 = v42;
        v5 = v39[0];
      }
      LODWORD(v42) = v7 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v41, 0);
    v5 = v39[0];
  }
  if ( v5 )
LABEL_8:
    sub_B91220((__int64)v39, v5);
  v9 = sub_B43CB0(a1);
  v10 = sub_B2D610(v9, 72);
  v11 = *(_QWORD *)(a1 - 32);
  LOBYTE(v52) = v10;
  v12 = *(_QWORD *)(a1 - 64);
  v38 = 257;
  v13 = *(_QWORD *)(v11 + 8);
  v33 = (unsigned __int8 *)v11;
  v36 = v12;
  v14 = sub_AA4E30(v44);
  v15 = sub_AE5020(v14, v13);
  v40 = 257;
  v16 = sub_BD2C40(80, 1u);
  v17 = (__int64)v16;
  if ( v16 )
    sub_B4D190((__int64)v16, v13, v36, (__int64)v39, 0, v15, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v49 + 2))(v49, v17, v37, v45, v46);
  v18 = (unsigned __int64)v41;
  v19 = &v41[16 * (unsigned int)v42];
  if ( v41 != v19 )
  {
    do
    {
      v20 = *(_QWORD *)(v18 + 8);
      v21 = *(_DWORD *)v18;
      v18 += 16LL;
      sub_B99FD0(v17, v21, v20);
    }
    while ( v19 != (_BYTE *)v18 );
  }
  v35 = sub_2A2C8D0((*(_WORD *)(a1 + 2) >> 4) & 0x1F, (__int64)&v41, v17, v33);
  v22 = sub_AA4E30(v44);
  v23 = sub_AE5020(v22, *(_QWORD *)(v35 + 8));
  v40 = 257;
  v34 = v23;
  v24 = sub_BD2C40(80, unk_3F10A10);
  v25 = v35;
  v26 = (__int64)v24;
  if ( v24 )
    sub_B4D3C0((__int64)v24, v35, v36, 0, v34, v35, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64, __int64))*v49 + 2))(
    v49,
    v26,
    v39,
    v45,
    v46,
    v25);
  v27 = (unsigned __int64)v41;
  v28 = &v41[16 * (unsigned int)v42];
  if ( v41 != v28 )
  {
    do
    {
      v29 = *(_QWORD *)(v27 + 8);
      v30 = *(_DWORD *)v27;
      v27 += 16LL;
      sub_B99FD0(v26, v30, v29);
    }
    while ( v28 != (_BYTE *)v27 );
  }
  sub_BD84D0(a1, v17);
  sub_B43D60((_QWORD *)a1);
  nullsub_61();
  v56 = &unk_49DA100;
  nullsub_63();
  if ( v41 != v43 )
    _libc_free((unsigned __int64)v41);
  return 1;
}
