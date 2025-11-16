// Function: sub_D0E890
// Address: 0xd0e890
//
__int64 __fastcall sub_D0E890(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 result; // rax
  __int64 v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // rsi

  if ( (unsigned int)sub_B46E30(a1) == 1 )
    return 0;
  result = *(_QWORD *)(a2 + 16);
  do
  {
    if ( !result )
      BUG();
    v6 = *(_QWORD *)(result + 24);
    result = *(_QWORD *)(result + 8);
  }
  while ( (unsigned __int8)(*(_BYTE *)v6 - 30) > 0xAu );
  if ( !result )
  {
LABEL_15:
    if ( !(_BYTE)a3 )
      return a3;
    return 0;
  }
  while ( 1 )
  {
    v7 = *(_QWORD *)(result + 24);
    if ( (unsigned __int8)(*(_BYTE *)v7 - 30) <= 0xAu )
      break;
    result = *(_QWORD *)(result + 8);
    if ( !result )
      goto LABEL_15;
  }
  if ( !(_BYTE)a3 )
    return 1;
  v8 = *(_QWORD *)(v6 + 40);
  while ( v8 == *(_QWORD *)(v7 + 40) )
  {
    result = *(_QWORD *)(result + 8);
    if ( !result )
      return 0;
    while ( 1 )
    {
      v7 = *(_QWORD *)(result + 24);
      if ( (unsigned __int8)(*(_BYTE *)v7 - 30) <= 0xAu )
        break;
      result = *(_QWORD *)(result + 8);
      if ( !result )
        return result;
    }
  }
  return a3;
}
