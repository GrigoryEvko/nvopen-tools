// Function: sub_29CB160
// Address: 0x29cb160
//
__int64 __fastcall sub_29CB160(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r10
  __int64 v3; // rdx

  v2 = *(_QWORD **)(a2 + 40);
  v3 = *(_QWORD *)(a2 + 64);
  if ( *(_DWORD *)(a1 + 224) == 1 )
    return sub_29C57C0(
             v2,
             a2 + 56,
             v3,
             *(_QWORD *)(a1 + 176),
             *(_QWORD *)(a1 + 184),
             *(_BYTE *)(a1 + 228),
             "CheckFunctionDebugify",
             0x15u,
             *(_DWORD **)(a1 + 208));
  else
    return sub_29C8000(
             (__int64)v2,
             a2 + 56,
             v3,
             *(_QWORD *)(a1 + 216),
             "CheckFunctionDebugify (original debuginfo)",
             0x2Au,
             *(unsigned __int8 **)(a1 + 176),
             *(_QWORD *)(a1 + 184),
             *(unsigned __int8 **)(a1 + 192),
             *(_QWORD *)(a1 + 200));
}
