// Function: sub_CEA4B0
// Address: 0xcea4b0
//
_BOOL8 __fastcall sub_CEA4B0(int a1)
{
  char *v1; // rax
  unsigned __int64 v2; // rdx

  v1 = sub_B60B70(a1);
  if ( v2 <= 0xC )
    goto LABEL_14;
  if ( *(_QWORD *)v1 == 0x76766E2E6D766C6CLL && *((_DWORD *)v1 + 2) == 1702112877 && v1[12] == 120
    || v2 != 13
    && (*(_QWORD *)v1 == 0x76766E2E6D766C6CLL && *((_DWORD *)v1 + 2) == 1819553389 && *((_WORD *)v1 + 6) == 13412
     || *(_QWORD *)v1 == 0x76766E2E6D766C6CLL && *((_DWORD *)v1 + 2) == 1970482797 && *((_WORD *)v1 + 6) == 25708)
    || *(_QWORD *)v1 == 0x76766E2E6D766C6CLL && *((_DWORD *)v1 + 2) == 2020879981 && v1[12] == 113
    || *(_QWORD *)v1 == 0x76766E2E6D766C6CLL && *((_DWORD *)v1 + 2) == 1970482797 && v1[12] == 113 )
  {
    return 1;
  }
  if ( v2 <= 0x10 )
  {
LABEL_14:
    if ( v2 <= 0xD )
      return 0;
  }
  else if ( !(*(_QWORD *)v1 ^ 0x76766E2E6D766C6CLL | *((_QWORD *)v1 + 1) ^ 0x6570797473692E6DLL) && v1[16] == 112 )
  {
    return 1;
  }
  if ( *(_QWORD *)v1 != 0x76766E2E6D766C6CLL || *((_DWORD *)v1 + 2) != 1970482797 )
    return 0;
  return *((_WORD *)v1 + 6) == 29811;
}
