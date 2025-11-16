// Function: sub_E551A0
// Address: 0xe551a0
//
__int64 __fastcall sub_E551A0(__int64 a1, __int64 *a2, __int64 a3, __int64 *a4, __int64 *a5)
{
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 (*v18)(void); // rdx
  char v19; // al
  __int64 v20; // rdx
  bool v21; // al
  int v22; // eax
  __int64 v25; // [rsp+10h] [rbp-50h] BYREF
  __int64 v26; // [rsp+18h] [rbp-48h] BYREF
  __int64 v27; // [rsp+20h] [rbp-40h] BYREF
  __int64 v28[7]; // [rsp+28h] [rbp-38h] BYREF

  v6 = *a2;
  *a2 = 0;
  v7 = *a4;
  *a4 = 0;
  v8 = *a5;
  *a5 = 0;
  v9 = sub_22077B0(752);
  v10 = v9;
  if ( v9 )
  {
    sub_E98A20(v9, a1);
    v11 = *(_QWORD *)(a1 + 152);
    *(_QWORD *)(v10 + 296) = v6;
    *(_QWORD *)v10 = off_49E14D8;
    *(_QWORD *)(v10 + 312) = v11;
    *(_QWORD *)(v10 + 304) = v6;
    *(_QWORD *)(v10 + 320) = a3;
    v12 = 0;
    if ( v8 )
    {
      sub_106DB90(&v25, v8, v10 + 696);
      v12 = v25;
    }
    v28[0] = v8;
    v27 = v7;
    v26 = v12;
    v25 = 0;
    v13 = sub_22077B0(376);
    v14 = v13;
    if ( v13 )
      sub_E5B9D0(v13, a1, v28, &v27, &v26);
    v15 = v26;
    *(_QWORD *)(v10 + 328) = v14;
    if ( v15 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v15 + 8LL))(v15);
    if ( v27 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v27 + 8LL))(v27);
    if ( v28[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v28[0] + 8LL))(v28[0]);
    if ( v25 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v25 + 8LL))(v25);
    *(_QWORD *)(v10 + 336) = v10 + 360;
    *(_QWORD *)(v10 + 488) = v10 + 512;
    *(_QWORD *)(v10 + 344) = 0;
    *(_QWORD *)(v10 + 352) = 128;
    *(_QWORD *)(v10 + 640) = &unk_49DD288;
    *(_QWORD *)(v10 + 688) = v10 + 488;
    *(_QWORD *)(v10 + 496) = 0;
    *(_QWORD *)(v10 + 504) = 128;
    *(_DWORD *)(v10 + 648) = 2;
    *(_BYTE *)(v10 + 680) = 0;
    *(_DWORD *)(v10 + 684) = 1;
    *(_QWORD *)(v10 + 672) = 0;
    *(_QWORD *)(v10 + 664) = 0;
    *(_QWORD *)(v10 + 656) = 0;
    sub_CB5980(v10 + 640, 0, 0, 0);
    *(_DWORD *)(v10 + 704) = 0;
    *(_BYTE *)(v10 + 736) = 0;
    *(_QWORD *)(v10 + 728) = 0;
    *(_QWORD *)(v10 + 696) = &unk_49DD308;
    v16 = *(_QWORD *)(v10 + 328);
    *(_QWORD *)(v10 + 720) = 0;
    *(_QWORD *)(v10 + 712) = 0;
    *(_QWORD *)(v10 + 740) = 0;
    v17 = *(_QWORD *)(v16 + 8);
    if ( v17 )
    {
      v18 = *(__int64 (**)(void))(*(_QWORD *)v17 + 16LL);
      v19 = 0;
      if ( v18 != sub_E4C920 )
        v19 = v18();
      *(_BYTE *)(v10 + 277) = v19;
    }
    v20 = *(_QWORD *)(a1 + 2368);
    *(_BYTE *)(a1 + 1908) = 1;
    if ( v20 )
    {
      v21 = (*(_BYTE *)(v20 + 1) & 8) != 0;
      *(_BYTE *)(v10 + 745) = v21;
      if ( v21 )
        *(_QWORD *)(*(_QWORD *)(v10 + 320) + 8LL) = v10 + 640;
      *(_BYTE *)(v10 + 746) = (*(_BYTE *)(v20 + 1) & 4) != 0;
      v22 = *(_DWORD *)(v20 + 24);
      switch ( v22 )
      {
        case 1:
          *(_BYTE *)(v10 + 747) = 1;
          break;
        case 2:
          *(_BYTE *)(v10 + 747) = *(_BYTE *)(*(_QWORD *)(a1 + 152) + 350LL);
          break;
        case 0:
          *(_BYTE *)(v10 + 747) = 0;
          break;
      }
    }
  }
  else
  {
    if ( v8 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL))(v8);
    if ( v7 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
    if ( v6 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
  }
  return v10;
}
