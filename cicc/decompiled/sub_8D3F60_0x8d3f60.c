// Function: sub_8D3F60
// Address: 0x8d3f60
//
_BOOL8 __fastcall sub_8D3F60(__int64 a1)
{
  __int64 v1; // rax

  v1 = sub_8D2290(a1);
  return *(_BYTE *)(v1 + 140) == 14 && !*(_BYTE *)(v1 + 160) && *(_DWORD *)(*(_QWORD *)(v1 + 168) + 28LL) == -2;
}
