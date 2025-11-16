// Function: sub_157ED60
// Address: 0x157ed60
//
__int64 __fastcall sub_157ED60(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdi
  char v3; // dl
  __int64 v5; // rdx

  v1 = *(_QWORD *)(a1 + 48);
  v2 = a1 + 40;
  if ( v1 == v2 )
    return 0;
  while ( 1 )
  {
    if ( !v1 )
      BUG();
    v3 = *(_BYTE *)(v1 - 8);
    if ( v3 != 77 )
    {
      if ( v3 != 78 )
        break;
      v5 = *(_QWORD *)(v1 - 48);
      if ( *(_BYTE *)(v5 + 16) || (*(_BYTE *)(v5 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v5 + 36) - 35) > 3 )
        break;
    }
    v1 = *(_QWORD *)(v1 + 8);
    if ( v2 == v1 )
      return 0;
  }
  return v1 - 24;
}
