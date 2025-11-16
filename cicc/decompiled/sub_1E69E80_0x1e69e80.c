// Function: sub_1E69E80
// Address: 0x1e69e80
//
__int64 __fastcall sub_1E69E80(__int64 a1, int a2)
{
  __int64 result; // rax

  if ( a2 < 0 )
    result = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    result = *(_QWORD *)(*(_QWORD *)(a1 + 272) + 8LL * (unsigned int)a2);
  if ( result )
  {
    if ( (*(_BYTE *)(result + 3) & 0x10) != 0 )
    {
      result = *(_QWORD *)(result + 32);
      if ( !result )
        return result;
      while ( (*(_BYTE *)(result + 3) & 0x10) != 0 )
      {
        result = *(_QWORD *)(result + 32);
        if ( !result )
          return result;
      }
    }
LABEL_5:
    *(_BYTE *)(result + 3) &= ~0x40u;
    while ( 1 )
    {
      result = *(_QWORD *)(result + 32);
      if ( !result )
        break;
      if ( (*(_BYTE *)(result + 3) & 0x10) == 0 )
        goto LABEL_5;
    }
  }
  return result;
}
