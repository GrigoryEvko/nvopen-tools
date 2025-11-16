// Function: sub_30A7E20
// Address: 0x30a7e20
//
unsigned __int64 __fastcall sub_30A7E20(unsigned __int64 a1)
{
  unsigned __int64 v1; // r8
  unsigned __int64 v2; // rax
  __int64 v4; // rdx

  v1 = a1;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 40) + 56LL) != a1 + 24 )
  {
    do
    {
      v2 = *(_QWORD *)(v1 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v2 )
        break;
      v1 = v2 - 24;
      if ( *(_BYTE *)(v2 - 24) == 85 )
      {
        v4 = *(_QWORD *)(v2 - 56);
        if ( v4 )
        {
          if ( !*(_BYTE *)v4
            && *(_QWORD *)(v4 + 24) == *(_QWORD *)(v2 + 56)
            && (*(_BYTE *)(v4 + 33) & 0x20) != 0
            && *(_DWORD *)(v4 + 36) == 199 )
          {
            return v2 - 24;
          }
        }
      }
    }
    while ( *(_QWORD *)(*(_QWORD *)(v2 + 16) + 56LL) != v2 );
  }
  return 0;
}
