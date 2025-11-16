// Function: sub_2F89A40
// Address: 0x2f89a40
//
__int64 __fastcall sub_2F89A40(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 (*v7)(); // rax
  __int64 v8; // rdi
  __int64 (*v9)(); // rax
  __int64 v11; // r12
  __int64 *v12; // rax
  __int64 v13; // rsi
  _QWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned __int64 v17; // r8
  __int64 v18; // r9
  char v19; // al
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  void *v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  _QWORD *v29; // rbx
  _QWORD *v30; // r12
  void (__fastcall *v31)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v32; // rax
  __int64 v33; // [rsp+0h] [rbp-360h]
  __int64 v34; // [rsp+10h] [rbp-350h] BYREF
  _QWORD *v35; // [rsp+18h] [rbp-348h]
  __int64 v36; // [rsp+20h] [rbp-340h]
  unsigned __int64 *v37; // [rsp+28h] [rbp-338h]
  _QWORD v38[2]; // [rsp+30h] [rbp-330h] BYREF
  __int64 v39; // [rsp+40h] [rbp-320h] BYREF
  _BYTE *v40; // [rsp+48h] [rbp-318h]
  __int64 v41; // [rsp+50h] [rbp-310h]
  int v42; // [rsp+58h] [rbp-308h]
  char v43; // [rsp+5Ch] [rbp-304h]
  _BYTE v44[16]; // [rsp+60h] [rbp-300h] BYREF
  unsigned __int64 v45[2]; // [rsp+70h] [rbp-2F0h] BYREF
  _BYTE v46[512]; // [rsp+80h] [rbp-2E0h] BYREF
  __int64 v47; // [rsp+280h] [rbp-E0h]
  __int64 v48; // [rsp+288h] [rbp-D8h]
  __int64 v49; // [rsp+290h] [rbp-D0h]
  __int64 v50; // [rsp+298h] [rbp-C8h]
  char v51; // [rsp+2A0h] [rbp-C0h]
  __int64 v52; // [rsp+2A8h] [rbp-B8h]
  char *v53; // [rsp+2B0h] [rbp-B0h]
  __int64 v54; // [rsp+2B8h] [rbp-A8h]
  int v55; // [rsp+2C0h] [rbp-A0h]
  char v56; // [rsp+2C4h] [rbp-9Ch]
  char v57; // [rsp+2C8h] [rbp-98h] BYREF
  __int16 v58; // [rsp+308h] [rbp-58h]
  _QWORD *v59; // [rsp+310h] [rbp-50h]
  _QWORD *v60; // [rsp+318h] [rbp-48h]
  __int64 v61; // [rsp+320h] [rbp-40h]

  if ( !(unsigned __int8)sub_B2D610(a3, 55) || sub_B2FC80(a3) )
  {
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  else
  {
    v7 = *(__int64 (**)())(*(_QWORD *)*a2 + 16LL);
    if ( v7 == sub_23CE270 )
      BUG();
    v8 = ((__int64 (__fastcall *)(_QWORD, __int64))v7)(*a2, a3);
    v9 = *(__int64 (**)())(*(_QWORD *)v8 + 144LL);
    if ( v9 == sub_2C8F680 || (v33 = ((__int64 (__fastcall *)(__int64))v9)(v8)) == 0 )
      sub_C64ED0("TargetLowering instance is required", 1u);
    v11 = sub_B2BEC0(a3);
    v49 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
    v53 = &v57;
    v45[0] = (unsigned __int64)v46;
    v58 = 0;
    v45[1] = 0x1000000000LL;
    v34 = a3;
    v35 = (_QWORD *)v33;
    v36 = v11;
    v47 = 0;
    v48 = 0;
    v50 = 0;
    v51 = 1;
    v52 = 0;
    v54 = 8;
    v55 = 0;
    v56 = 1;
    v59 = 0;
    v60 = 0;
    v61 = 0;
    v37 = v45;
    v38[0] = sub_BC1CD0(a4, &unk_4F881D0, a3) + 8;
    v12 = (__int64 *)sub_B2BE50(a3);
    v38[1] = sub_BCE3C0(v12, *(_DWORD *)(v11 + 4));
    v13 = sub_B2BE50(a3);
    v39 = sub_AE4420(v11, v13, 0);
    v14 = (_QWORD *)sub_B2BE50(a3);
    v41 = 0;
    v40 = (_BYTE *)sub_BCB2D0(v14);
    v19 = sub_2F86AD0(&v34, v13, v15, v16, v17, v18);
    v24 = (void *)(a1 + 32);
    if ( v19 )
    {
      v40 = v44;
      v35 = v38;
      v38[0] = &unk_4F81450;
      v36 = 0x100000002LL;
      LODWORD(v37) = 0;
      BYTE4(v37) = 1;
      v39 = 0;
      v41 = 2;
      v42 = 0;
      v43 = 1;
      v34 = 1;
      sub_C8CF70(a1, v24, 2, (__int64)v38, (__int64)&v34);
      v24 = (void *)(a1 + 80);
      sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v44, (__int64)&v39);
      if ( !v43 )
        _libc_free((unsigned __int64)v40);
      if ( !BYTE4(v37) )
        _libc_free((unsigned __int64)v35);
    }
    else
    {
      *(_QWORD *)(a1 + 8) = v24;
      *(_QWORD *)(a1 + 16) = 0x100000002LL;
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 56) = a1 + 80;
      *(_QWORD *)(a1 + 64) = 2;
      *(_DWORD *)(a1 + 72) = 0;
      *(_BYTE *)(a1 + 76) = 1;
      *(_DWORD *)(a1 + 24) = 0;
      *(_BYTE *)(a1 + 28) = 1;
      *(_QWORD *)(a1 + 32) = &qword_4F82400;
      *(_QWORD *)a1 = 1;
    }
    sub_FFCE90((__int64)v45, (__int64)v24, v20, v21, v22, v23);
    sub_FFD870((__int64)v45, (__int64)v24, v25, v26, v27, v28);
    sub_FFBC40((__int64)v45, (__int64)v24);
    v29 = v60;
    v30 = v59;
    if ( v60 != v59 )
    {
      do
      {
        v31 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v30[7];
        *v30 = &unk_49E5048;
        if ( v31 )
          v31(v30 + 5, v30 + 5, 3);
        *v30 = &unk_49DB368;
        v32 = v30[3];
        if ( v32 != 0 && v32 != -4096 && v32 != -8192 )
          sub_BD60C0(v30 + 1);
        v30 += 9;
      }
      while ( v29 != v30 );
      v30 = v59;
    }
    if ( v30 )
      j_j___libc_free_0((unsigned __int64)v30);
    if ( !v56 )
      _libc_free((unsigned __int64)v53);
    if ( (_BYTE *)v45[0] != v46 )
      _libc_free(v45[0]);
  }
  return a1;
}
