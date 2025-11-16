// Function: sub_879360
// Address: 0x879360
//
__int64 __fastcall sub_879360(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // r8d
  char v3; // dl
  __int64 v5; // rax

  v1 = *(_QWORD *)(a1 + 8);
  v2 = 0;
  if ( !v1 )
    return v2;
  v3 = *(_BYTE *)(v1 + 80);
  if ( v3 == 17 )
  {
    v5 = *(_QWORD *)(v1 + 88);
    if ( v5 )
    {
      do
      {
        if ( *(_BYTE *)(v5 + 80) != 10 || (*(_BYTE *)(*(_QWORD *)(v5 + 88) + 194LL) & 6) == 0 )
          return 1;
        v5 = *(_QWORD *)(v5 + 8);
      }
      while ( v5 );
      return 0;
    }
    return v2;
  }
  v2 = 1;
  if ( v3 != 10 )
    return v2;
  return (*(_BYTE *)(*(_QWORD *)(v1 + 88) + 194LL) & 6) == 0;
}
