// Function: sub_1E69C40
// Address: 0x1e69c40
//
__int64 __fastcall sub_1E69C40(__int64 a1, int a2, int a3)
{
  __int64 result; // rax
  __int64 v6; // rdi

  while ( 1 )
  {
    if ( a2 >= 0 )
    {
      result = *(_QWORD *)(a1 + 272);
      v6 = *(_QWORD *)(result + 8LL * (unsigned int)a2);
      if ( !v6 )
        return result;
    }
    else
    {
      result = *(_QWORD *)(a1 + 24);
      v6 = *(_QWORD *)(result + 16LL * (a2 & 0x7FFFFFFF) + 8);
      if ( !v6 )
        return result;
    }
    if ( (*(_BYTE *)(v6 + 3) & 0x10) != 0 )
      break;
LABEL_4:
    sub_1E310D0(v6, a3);
  }
  v6 = *(_QWORD *)(v6 + 32);
  result = v6;
  if ( v6 )
  {
    while ( (*(_BYTE *)(result + 3) & 0x10) != 0 )
    {
      result = *(_QWORD *)(result + 32);
      if ( !result )
        return result;
    }
    do
    {
      if ( (*(_BYTE *)(v6 + 3) & 0x10) == 0 )
        break;
      v6 = *(_QWORD *)(v6 + 32);
    }
    while ( v6 );
    goto LABEL_4;
  }
  return result;
}
