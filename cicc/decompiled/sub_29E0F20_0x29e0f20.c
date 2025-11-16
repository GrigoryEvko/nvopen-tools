// Function: sub_29E0F20
// Address: 0x29e0f20
//
__int64 __fastcall sub_29E0F20(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx

  v1 = sub_AA4FF0(a1);
  if ( v1 == a1 + 48 )
    return 0;
  while ( 1 )
  {
    if ( !v1 )
      BUG();
    if ( *(_BYTE *)(v1 - 24) == 85 )
    {
      v2 = *(_QWORD *)(v1 - 56);
      if ( v2 )
      {
        if ( !*(_BYTE *)v2
          && *(_QWORD *)(v2 + 24) == *(_QWORD *)(v1 + 56)
          && (*(_BYTE *)(v2 + 33) & 0x20) != 0
          && *(_DWORD *)(v2 + 36) == 143 )
        {
          break;
        }
      }
    }
    v1 = *(_QWORD *)(v1 + 8);
    if ( v1 == a1 + 48 )
      return 0;
  }
  return v1 - 24;
}
