// Function: sub_8D5B70
// Address: 0x8d5b70
//
_QWORD *__fastcall sub_8D5B70(__int64 a1, __int64 a2, int a3)
{
  __int64 i; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD *result; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx

  if ( dword_4F077C4 != 2 )
    return 0;
  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  while ( *(_BYTE *)(a2 + 140) == 12 )
    a2 = *(_QWORD *)(a2 + 160);
  if ( a3 )
  {
    if ( a2 != i )
    {
      if ( !dword_4F07588 || (v5 = *(_QWORD *)(i + 32), *(_QWORD *)(a2 + 32) != v5) || !v5 )
      {
        if ( sub_8D23B0(i) && sub_8D3A70(i) )
          sub_8AD220(i, 0);
      }
    }
  }
  if ( (unsigned __int8)(*(_BYTE *)(a2 + 140) - 9) > 2u )
    return 0;
  if ( (unsigned __int8)(*(_BYTE *)(i + 140) - 9) > 2u )
    return 0;
  v6 = *(_QWORD *)(i + 168);
  v7 = *(_QWORD *)(v6 + 152);
  if ( !v7 )
    return 0;
  if ( (*(_BYTE *)(v7 + 29) & 0x20) != 0 )
    return 0;
  v8 = *(_QWORD *)(*(_QWORD *)(a2 + 168) + 152LL);
  if ( !v8 )
    return 0;
  if ( (*(_BYTE *)(v8 + 29) & 0x20) != 0 )
    return 0;
  result = *(_QWORD **)v6;
  if ( !*(_QWORD *)v6 )
    return 0;
  while ( 1 )
  {
    v10 = result[5];
    if ( v10 == a2 )
      break;
    if ( v10 )
    {
      if ( dword_4F07588 )
      {
        v11 = *(_QWORD *)(v10 + 32);
        if ( *(_QWORD *)(a2 + 32) == v11 )
        {
          if ( v11 )
            break;
        }
      }
    }
    result = (_QWORD *)*result;
    if ( !result )
      return 0;
  }
  return result;
}
