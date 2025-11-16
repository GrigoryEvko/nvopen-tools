// Function: sub_1B7CA20
// Address: 0x1b7ca20
//
__int64 __fastcall sub_1B7CA20(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 v2; // r8
  __int64 v4; // rax
  int v5; // eax

  v1 = *(_BYTE *)(a1 + 16);
  v2 = 0;
  if ( v1 <= 0x17u )
    return v2;
  if ( (unsigned __int8)(v1 - 54) > 1u )
  {
    if ( v1 == 78 )
    {
      v4 = *(_QWORD *)(a1 - 24);
      if ( !*(_BYTE *)(v4 + 16) )
      {
        v5 = *(_DWORD *)(v4 + 36);
        if ( v5 )
        {
          if ( v5 == 4085 || v5 == 4057 )
          {
            return *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
          }
          else if ( v5 == 4503 || v5 == 4492 )
          {
            return *(_QWORD *)(a1 + 24 * (2LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
          }
        }
      }
    }
    return v2;
  }
  return *(_QWORD *)(a1 - 24);
}
