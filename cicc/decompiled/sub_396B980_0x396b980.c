// Function: sub_396B980
// Address: 0x396b980
//
char __fastcall sub_396B980(__int64 a1, __int64 a2)
{
  char result; // al
  __int64 v3; // rdi
  __int64 v4; // rax

  result = 1;
  if ( *(_QWORD *)(a1 + 408) == *(_QWORD *)(a1 + 416) )
  {
    result = *(_BYTE *)(a1 + 523);
    if ( !result )
    {
      result = *(_BYTE *)(a2 + 1744);
      if ( !result )
      {
        v3 = *(_QWORD *)a1;
        if ( (*(_BYTE *)(v3 + 18) & 8) != 0 )
        {
          v4 = sub_15E38F0(v3);
          return (unsigned int)sub_14DD7D0(v4) == 0;
        }
      }
    }
  }
  return result;
}
