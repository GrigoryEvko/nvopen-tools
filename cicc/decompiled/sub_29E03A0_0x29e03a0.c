// Function: sub_29E03A0
// Address: 0x29e03a0
//
__int64 __fastcall sub_29E03A0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx

  while ( a1 )
  {
    v1 = *(_QWORD *)(a1 + 24);
    if ( *(_BYTE *)v1 == 85 )
    {
      v2 = *(_QWORD *)(v1 - 32);
      if ( v2 )
      {
        if ( !*(_BYTE *)v2
          && *(_QWORD *)(v2 + 24) == *(_QWORD *)(v1 + 80)
          && (*(_BYTE *)(v2 + 33) & 0x20) != 0
          && (unsigned int)(*(_DWORD *)(v2 + 36) - 210) <= 1 )
        {
          return 1;
        }
      }
    }
    a1 = *(_QWORD *)(a1 + 8);
  }
  return 0;
}
