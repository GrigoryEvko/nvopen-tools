// Function: sub_AC2D20
// Address: 0xac2d20
//
__int64 __fastcall sub_AC2D20(__int64 a1)
{
  __int64 v1; // rbx
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 16);
  if ( !v1 )
    return 0;
  while ( (unsigned __int8)(**(_BYTE **)(v1 + 24) - 4) <= 0x11u )
  {
    result = sub_AC2D20();
    if ( !(_BYTE)result )
    {
      v1 = *(_QWORD *)(v1 + 8);
      if ( v1 )
        continue;
    }
    return result;
  }
  return 1;
}
