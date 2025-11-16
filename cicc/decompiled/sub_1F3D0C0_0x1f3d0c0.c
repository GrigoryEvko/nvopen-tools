// Function: sub_1F3D0C0
// Address: 0x1f3d0c0
//
bool __fastcall sub_1F3D0C0(__int64 a1, __int64 a2)
{
  _DWORD *v2; // rax
  __int64 v3; // rdx
  bool result; // al
  __int64 v5; // [rsp+8h] [rbp-18h] BYREF

  v5 = sub_1560340((_QWORD *)(a2 + 112), -1, "no-jump-tables", 0xEu);
  v2 = (_DWORD *)sub_155D8B0(&v5);
  if ( v3 == 4 && *v2 == 1702195828 )
    return 0;
  result = 1;
  if ( (*(_BYTE *)(a1 + 2871) & 0xFB) != 0 )
    return (*(_BYTE *)(a1 + 2870) & 0xFB) == 0;
  return result;
}
