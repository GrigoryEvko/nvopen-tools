// Function: sub_30A7DC0
// Address: 0x30a7dc0
//
__int64 __fastcall sub_30A7DC0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdi
  __int64 v3; // rdx

  v1 = *(_QWORD *)(a1 + 56);
  v2 = a1 + 48;
  if ( v1 == v2 )
    return 0;
  while ( 1 )
  {
    if ( !v1 )
      BUG();
    if ( *(_BYTE *)(v1 - 24) == 85 )
    {
      v3 = *(_QWORD *)(v1 - 56);
      if ( v3 )
      {
        if ( !*(_BYTE *)v3
          && *(_QWORD *)(v3 + 24) == *(_QWORD *)(v1 + 56)
          && (*(_BYTE *)(v3 + 33) & 0x20) != 0
          && *(_DWORD *)(v3 + 36) == 198 )
        {
          break;
        }
      }
    }
    v1 = *(_QWORD *)(v1 + 8);
    if ( v2 == v1 )
      return 0;
  }
  return v1 - 24;
}
