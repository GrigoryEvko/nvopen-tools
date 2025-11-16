// Function: sub_29C7F50
// Address: 0x29c7f50
//
__int64 __fastcall sub_29C7F50(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdx
  __int64 v5; // rsi
  unsigned int v6; // r12d
  _BYTE v8[16]; // [rsp+0h] [rbp-30h] BYREF
  void (__fastcall *v9)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-20h]

  v3 = (__int64)(a2 + 3);
  v5 = a2[4];
  if ( *(_DWORD *)(a1 + 200) != 1 )
    return (unsigned int)sub_29C70F0(
                           (__int64)a2,
                           v5,
                           v3,
                           *(_QWORD *)(a1 + 192),
                           "ModuleDebugify (original debuginfo)",
                           0x23u);
  v9 = 0;
  v6 = sub_29C2F90(a2, v5, v3, "ModuleDebugify: ", 0x10u, (__int64)v8);
  if ( !v9 )
    return v6;
  v9(v8, v8, 3);
  return v6;
}
