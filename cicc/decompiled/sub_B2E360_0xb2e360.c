// Function: sub_B2E360
// Address: 0xb2e360
//
__int64 __fastcall sub_B2E360(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // rax

  LOBYTE(a5) = (((*(_BYTE *)(a1 + 32) & 0xF) + 9) & 0xFu) > 1
            && (*(_BYTE *)(a1 + 32) & 0xFu) - 2 > 1
            && (*(_BYTE *)(a1 + 32) & 0xF) != 1;
  if ( (_BYTE)a5 )
    return 0;
  v5 = *(_QWORD *)(a1 + 16);
  if ( v5 )
  {
    while ( **(_BYTE **)(v5 + 24) == 4 )
    {
      v5 = *(_QWORD *)(v5 + 8);
      if ( !v5 )
        return 1;
    }
    return a5;
  }
  return 1;
}
