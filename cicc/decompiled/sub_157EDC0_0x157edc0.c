// Function: sub_157EDC0
// Address: 0x157edc0
//
__int64 __fastcall sub_157EDC0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdi
  char v3; // dl
  __int64 v4; // r8
  __int64 v6; // rcx

  v1 = *(_QWORD *)(a1 + 48);
  v2 = a1 + 40;
  if ( v1 == v2 )
    return 0;
  while ( 1 )
  {
    if ( !v1 )
      BUG();
    v3 = *(_BYTE *)(v1 - 8);
    v4 = v1 - 24;
    if ( v3 != 77 )
    {
      if ( v3 != 78 )
        break;
      v6 = *(_QWORD *)(v1 - 48);
      if ( *(_BYTE *)(v6 + 16)
        || (*(_BYTE *)(v6 + 33) & 0x20) == 0
        || (unsigned int)(*(_DWORD *)(v6 + 36) - 35) > 3
        && ((*(_BYTE *)(v6 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v6 + 36) - 116) > 1) )
      {
        break;
      }
    }
    v1 = *(_QWORD *)(v1 + 8);
    if ( v2 == v1 )
      return 0;
  }
  return v4;
}
