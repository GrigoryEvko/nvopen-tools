// Function: sub_22DB930
// Address: 0x22db930
//
__int64 __fastcall sub_22DB930(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // r15
  __int64 result; // rax

  v6 = *(_QWORD *)(a2 + 16);
  if ( !v6 )
    return 1;
  while ( 1 )
  {
    v7 = *(_QWORD *)(v6 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v7 - 30) <= 0xAu )
      break;
    v6 = *(_QWORD *)(v6 + 8);
    if ( !v6 )
      return 1;
  }
LABEL_5:
  v8 = *(_QWORD *)(v7 + 40);
  if ( !(unsigned __int8)sub_B19720(*(_QWORD *)(a1 + 8), a3, v8)
    || (result = sub_B19720(*(_QWORD *)(a1 + 8), a4, v8), (_BYTE)result) )
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        return 1;
      v7 = *(_QWORD *)(v6 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v7 - 30) <= 0xAu )
        goto LABEL_5;
    }
  }
  return result;
}
