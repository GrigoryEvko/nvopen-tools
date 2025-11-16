// Function: sub_29CB210
// Address: 0x29cb210
//
__int64 __fastcall sub_29CB210(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // rdx
  __int64 v6; // r10

  v5 = (__int64)(a3 + 3);
  v6 = a3[4];
  if ( *(_DWORD *)(a2 + 48) == 1 )
    sub_29C57C0(
      a3,
      v6,
      v5,
      *(_QWORD *)a2,
      *(_QWORD *)(a2 + 8),
      *(_BYTE *)(a2 + 52),
      "CheckModuleDebugify",
      0x13u,
      *(_DWORD **)(a2 + 32));
  else
    sub_29C8000(
      (__int64)a3,
      v6,
      v5,
      *(_QWORD *)(a2 + 40),
      "CheckModuleDebugify (original debuginfo)",
      0x28u,
      *(unsigned __int8 **)a2,
      *(_QWORD *)(a2 + 8),
      *(unsigned __int8 **)(a2 + 16),
      *(_QWORD *)(a2 + 24));
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
