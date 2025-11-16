// Function: sub_27A2D20
// Address: 0x27a2d20
//
__int64 __fastcall sub_27A2D20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 result; // rax

  v4 = a2;
  v5 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v6 = *(_QWORD *)(a2 - 8);
    v4 = v6 + v5;
  }
  else
  {
    v6 = a2 - v5;
  }
  if ( v6 == v4 )
    return 1;
  while ( 1 )
  {
    if ( **(_BYTE **)v6 > 0x1Cu )
    {
      result = sub_B19720(*(_QWORD *)(a1 + 216), *(_QWORD *)(*(_QWORD *)v6 + 40LL), a3);
      if ( !(_BYTE)result )
        break;
    }
    v6 += 32;
    if ( v4 == v6 )
      return 1;
  }
  return result;
}
