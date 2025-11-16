// Function: sub_CEA640
// Address: 0xcea640
//
_BOOL8 __fastcall sub_CEA640(__int64 a1)
{
  __int64 v2; // rax

  return *(_BYTE *)a1 == 85
      && (v2 = *(_QWORD *)(a1 - 32)) != 0
      && !*(_BYTE *)v2
      && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80)
      && (*(_BYTE *)(v2 + 33) & 0x20) != 0
      && sub_CEA4B0(*(_DWORD *)(v2 + 36));
}
