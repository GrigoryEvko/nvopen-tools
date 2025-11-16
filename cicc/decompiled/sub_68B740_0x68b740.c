// Function: sub_68B740
// Address: 0x68b740
//
__int64 __fastcall sub_68B740(__int64 a1)
{
  char v1; // al
  __int64 v2; // r8
  __int64 v3; // rax
  __int64 v4; // rdx

  v1 = *(_BYTE *)(a1 + 16);
  v2 = a1 + 144;
  if ( v1 != 2 )
  {
    v2 = 0;
    if ( v1 == 1 )
    {
      v3 = *(_QWORD *)(a1 + 144);
      if ( *(_BYTE *)(v3 + 24) == 5 )
      {
        v4 = *(_QWORD *)(v3 + 56);
        if ( (*(_BYTE *)(v4 + 48) & 0xFB) == 2 )
        {
          v2 = *(_QWORD *)(v4 + 56);
          if ( *(_BYTE *)(v2 + 173) != 10 )
            return 0;
        }
      }
    }
  }
  return v2;
}
