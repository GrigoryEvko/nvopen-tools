// Function: sub_CEA680
// Address: 0xcea680
//
_BOOL8 __fastcall sub_CEA680(__int64 a1)
{
  _BOOL8 result; // rax
  __int64 v2; // r8
  const char *v3; // rax
  unsigned __int64 v4; // rdx

  result = 0;
  if ( *(_BYTE *)a1 == 85 )
  {
    v2 = *(_QWORD *)(a1 - 32);
    if ( v2 )
    {
      if ( !*(_BYTE *)v2 && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80) && (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
      {
        v3 = sub_BD5D20(v2);
        return v4 > 0xE
            && *(_QWORD *)v3 == 0x76766E2E6D766C6CLL
            && *((_DWORD *)v3 + 2) == 1970482797
            && *((_WORD *)v3 + 6) == 25708
            && v3[14] == 46;
      }
    }
  }
  return result;
}
