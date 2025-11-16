// Function: sub_2F0F9F0
// Address: 0x2f0f9f0
//
void __fastcall sub_2F0F9F0(__int64 a1, __int64 a2)
{
  char v2; // al
  _BOOL8 v3; // rcx
  char v4; // al
  _BOOL8 v5; // rcx
  char v6; // al
  _BOOL8 v7; // rcx
  char v8; // al
  _BOOL8 v9; // rcx
  char v10; // al
  _BOOL8 v11; // rcx
  char v12; // al
  _BOOL8 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rax
  const char *v16; // rax
  __int64 v17; // rdx
  void (__fastcall *v18)(__int64, __int64 **); // rcx
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rax
  char v22; // [rsp+17h] [rbp-129h] BYREF
  __int64 v23; // [rsp+18h] [rbp-128h] BYREF
  _QWORD v24[2]; // [rsp+20h] [rbp-120h] BYREF
  void *v25; // [rsp+30h] [rbp-110h] BYREF
  __int64 v26; // [rsp+38h] [rbp-108h]
  __int64 v27; // [rsp+40h] [rbp-100h]
  __int64 v28; // [rsp+48h] [rbp-F8h]
  __int64 v29; // [rsp+50h] [rbp-F0h]
  __int64 v30; // [rsp+58h] [rbp-E8h]
  __int64 **v31; // [rsp+60h] [rbp-E0h]
  __int64 *v32; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v33; // [rsp+78h] [rbp-C8h]
  __int64 v34; // [rsp+80h] [rbp-C0h] BYREF
  char v35; // [rsp+88h] [rbp-B8h] BYREF
  __int128 v36; // [rsp+90h] [rbp-B0h]

  LOBYTE(v32) = 0;
  sub_2F07B20(a1, (__int64)"isFrameAddressTaken", (_BYTE *)a2, &v32, 0);
  LOBYTE(v32) = 0;
  sub_2F07B20(a1, (__int64)"isReturnAddressTaken", (_BYTE *)(a2 + 1), &v32, 0);
  LOBYTE(v32) = 0;
  sub_2F07B20(a1, (__int64)"hasStackMap", (_BYTE *)(a2 + 2), &v32, 0);
  LOBYTE(v32) = 0;
  sub_2F07B20(a1, (__int64)"hasPatchPoint", (_BYTE *)(a2 + 3), &v32, 0);
  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v3 = 0;
  if ( v2 )
    v3 = *(_QWORD *)(a2 + 8) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, void **, __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "stackSize",
         0,
         v3,
         &v25,
         &v32) )
  {
    sub_2F07BD0(a1, (_QWORD *)(a2 + 8));
    (*(void (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v32);
  }
  else if ( (_BYTE)v25 )
  {
    *(_QWORD *)(a2 + 8) = 0;
  }
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v5 = 0;
  if ( v4 )
    v5 = *(_DWORD *)(a2 + 16) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "offsetAdjustment",
         0,
         v5,
         &v22,
         &v23) )
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    {
      v30 = 0x100000000LL;
      v32 = (__int64 *)&v35;
      v25 = &unk_49DD288;
      v33 = 0;
      v34 = 128;
      v26 = 2;
      v27 = 0;
      v28 = 0;
      v29 = 0;
      v31 = &v32;
      sub_CB5980((__int64)&v25, 0, 0, 0);
      v19 = sub_CB0A70(a1);
      sub_CB2E20((int *)(a2 + 16), v19, (__int64)&v25);
      v20 = v31[1];
      v24[0] = *v31;
      v21 = *(_QWORD *)a1;
      v24[1] = v20;
      (*(void (__fastcall **)(__int64, _QWORD *, _QWORD))(v21 + 216))(a1, v24, 0);
      v25 = &unk_49DD388;
      sub_CB5840((__int64)&v25);
      if ( v32 != (__int64 *)&v35 )
        _libc_free((unsigned __int64)v32);
    }
    else
    {
      v14 = *(_QWORD *)a1;
      v25 = 0;
      v26 = 0;
      (*(void (__fastcall **)(__int64, void **, _QWORD))(v14 + 216))(a1, &v25, 0);
      v15 = sub_CB0A70(a1);
      v16 = sub_CB2E30((__int64)v25, v26, v15, (_DWORD *)(a2 + 16));
      if ( v17 )
      {
        v18 = *(void (__fastcall **)(__int64, __int64 **))(*(_QWORD *)a1 + 248LL);
        LOWORD(v36) = 261;
        v32 = (__int64 *)v16;
        v33 = v17;
        v18(a1, &v32);
      }
    }
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v23);
  }
  else if ( v22 )
  {
    *(_DWORD *)(a2 + 16) = 0;
  }
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v7 = 0;
  if ( v6 )
    v7 = *(_DWORD *)(a2 + 20) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, void **, __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "maxAlignment",
         0,
         v7,
         &v25,
         &v32) )
  {
    sub_2F07DB0(a1, (unsigned int *)(a2 + 20));
    (*(void (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v32);
  }
  else if ( (_BYTE)v25 )
  {
    *(_DWORD *)(a2 + 20) = 0;
  }
  LOBYTE(v32) = 0;
  sub_2F07B20(a1, (__int64)"adjustsStack", (_BYTE *)(a2 + 24), &v32, 0);
  LOBYTE(v32) = 0;
  sub_2F07B20(a1, (__int64)"hasCalls", (_BYTE *)(a2 + 25), &v32, 0);
  v32 = &v34;
  v33 = 0;
  LOBYTE(v34) = 0;
  v36 = 0;
  sub_2F0ECF0(a1, (__int64)"stackProtector", a2 + 32, (__int64)&v32, 0);
  if ( v32 != &v34 )
    j_j___libc_free_0((unsigned __int64)v32);
  v32 = &v34;
  v33 = 0;
  LOBYTE(v34) = 0;
  v36 = 0;
  sub_2F0ECF0(a1, (__int64)"functionContext", a2 + 80, (__int64)&v32, 0);
  if ( v32 != &v34 )
    j_j___libc_free_0((unsigned __int64)v32);
  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v9 = 0;
  if ( v8 )
    v9 = *(_DWORD *)(a2 + 128) == -1;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, void **, __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "maxCallFrameSize",
         0,
         v9,
         &v25,
         &v32) )
  {
    sub_2F07DB0(a1, (unsigned int *)(a2 + 128));
    (*(void (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v32);
  }
  else if ( (_BYTE)v25 )
  {
    *(_DWORD *)(a2 + 128) = -1;
  }
  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v11 = 0;
  if ( v10 )
    v11 = *(_DWORD *)(a2 + 132) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, void **, __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "cvBytesOfCalleeSavedRegisters",
         0,
         v11,
         &v25,
         &v32) )
  {
    sub_2F07DB0(a1, (unsigned int *)(a2 + 132));
    (*(void (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v32);
  }
  else if ( (_BYTE)v25 )
  {
    *(_DWORD *)(a2 + 132) = 0;
  }
  LOBYTE(v32) = 0;
  sub_2F07B20(a1, (__int64)"hasOpaqueSPAdjustment", (_BYTE *)(a2 + 136), &v32, 0);
  LOBYTE(v32) = 0;
  sub_2F07B20(a1, (__int64)"hasVAStart", (_BYTE *)(a2 + 137), &v32, 0);
  LOBYTE(v32) = 0;
  sub_2F07B20(a1, (__int64)"hasMustTailInVarArgFunc", (_BYTE *)(a2 + 138), &v32, 0);
  LOBYTE(v32) = 0;
  sub_2F07B20(a1, (__int64)"hasTailCall", (_BYTE *)(a2 + 139), &v32, 0);
  LOBYTE(v32) = 0;
  sub_2F07B20(a1, (__int64)"isCalleeSavedInfoValid", (_BYTE *)(a2 + 140), &v32, 0);
  v12 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v13 = 0;
  if ( v12 )
    v13 = *(_DWORD *)(a2 + 144) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, void **, __int64 **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "localFrameSize",
         0,
         v13,
         &v25,
         &v32) )
  {
    sub_2F07DB0(a1, (unsigned int *)(a2 + 144));
    (*(void (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 128LL))(a1, v32);
  }
  else if ( (_BYTE)v25 )
  {
    *(_DWORD *)(a2 + 144) = 0;
  }
  v32 = &v34;
  v33 = 0;
  LOBYTE(v34) = 0;
  v36 = 0;
  sub_2F0ECF0(a1, (__int64)"savePoint", a2 + 152, (__int64)&v32, 0);
  if ( v32 != &v34 )
    j_j___libc_free_0((unsigned __int64)v32);
  v32 = &v34;
  v33 = 0;
  LOBYTE(v34) = 0;
  v36 = 0;
  sub_2F0ECF0(a1, (__int64)"restorePoint", a2 + 200, (__int64)&v32, 0);
  if ( v32 != &v34 )
    j_j___libc_free_0((unsigned __int64)v32);
}
