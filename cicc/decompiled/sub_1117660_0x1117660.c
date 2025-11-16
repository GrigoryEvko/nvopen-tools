// Function: sub_1117660
// Address: 0x1117660
//
__int64 __fastcall sub_1117660(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v7; // rbx
  __int64 v8; // rax

  v4 = *(_QWORD *)(a2 + 40);
  if ( !v4 )
    return 0;
  if ( *(_QWORD *)(a3 + 40) == v4 && a4 != v4 )
  {
    v7 = *(_QWORD *)(a2 + 16);
    if ( !v7 )
      return 1;
    while ( 1 )
    {
      v8 = *(_QWORD *)(v7 + 24);
      if ( a3 != v8 && !(unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 80), a4, *(_QWORD *)(v8 + 40)) )
        break;
      v7 = *(_QWORD *)(v7 + 8);
      if ( !v7 )
        return 1;
    }
  }
  return 0;
}
