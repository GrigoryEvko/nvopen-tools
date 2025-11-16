// Function: sub_29C7E40
// Address: 0x29c7e40
//
__int64 __fastcall sub_29C7E40(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v5; // rdx
  __int64 v7; // rsi
  _BYTE v8[16]; // [rsp+0h] [rbp-30h] BYREF
  void (__fastcall *v9)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-20h]

  v5 = (__int64)(a3 + 3);
  if ( *(_DWORD *)(a2 + 24) == 1 )
  {
    v7 = a3[4];
    v9 = 0;
    sub_29C2F90(a3, v7, v5, "ModuleDebugify: ", 0x10u, (__int64)v8);
    if ( v9 )
      v9(v8, v8, 3);
  }
  else
  {
    sub_29C70F0((__int64)a3, a3[4], v5, *(_QWORD *)(a2 + 16), "ModuleDebugify (original debuginfo)", 0x23u);
  }
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = &unk_4F82408;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
