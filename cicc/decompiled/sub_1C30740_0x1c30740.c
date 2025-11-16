// Function: sub_1C30740
// Address: 0x1c30740
//
_BOOL8 __fastcall sub_1C30740(__int64 a1)
{
  _BOOL8 result; // rax
  __int64 v2; // rdi
  const char *v3; // rax
  unsigned __int64 v4; // rdx

  result = 0;
  if ( *(_BYTE *)(a1 + 16) == 78 )
  {
    v2 = *(_QWORD *)(a1 - 24);
    if ( !*(_BYTE *)(v2 + 16) && (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
    {
      v3 = sub_1649960(v2);
      return v4 > 0xE
          && *(_QWORD *)v3 == 0x76766E2E6D766C6CLL
          && *((_DWORD *)v3 + 2) == 1970482797
          && *((_WORD *)v3 + 6) == 25708
          && v3[14] == 46;
    }
  }
  return result;
}
