// Function: sub_29CB0C0
// Address: 0x29cb0c0
//
__int64 __fastcall sub_29CB0C0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rdx
  __int64 v4; // rsi

  v3 = (__int64)(a2 + 3);
  v4 = a2[4];
  if ( *(_DWORD *)(a1 + 224) == 1 )
    return sub_29C57C0(
             a2,
             v4,
             v3,
             *(_QWORD *)(a1 + 176),
             *(_QWORD *)(a1 + 184),
             *(_BYTE *)(a1 + 228),
             "CheckModuleDebugify",
             0x13u,
             *(_DWORD **)(a1 + 208));
  else
    return sub_29C8000(
             (__int64)a2,
             v4,
             v3,
             *(_QWORD *)(a1 + 216),
             "CheckModuleDebugify (original debuginfo)",
             0x28u,
             *(unsigned __int8 **)(a1 + 176),
             *(_QWORD *)(a1 + 184),
             *(unsigned __int8 **)(a1 + 192),
             *(_QWORD *)(a1 + 200));
}
