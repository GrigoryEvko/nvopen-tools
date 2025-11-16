// Function: sub_1443AC0
// Address: 0x1443ac0
//
__int64 __fastcall sub_1443AC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 result; // rax
  __int64 v11; // r15
  __int64 v12; // rcx
  __int64 v13; // r8

  v6 = *(_QWORD *)(a2 + 8);
  if ( !v6 )
    return 1;
  while ( 1 )
  {
    v7 = sub_1648700(v6);
    if ( (unsigned __int8)(*(_BYTE *)(v7 + 16) - 25) <= 9u )
      break;
    v6 = *(_QWORD *)(v6 + 8);
    if ( !v6 )
      return 1;
  }
LABEL_7:
  v11 = *(_QWORD *)(v7 + 40);
  if ( !(unsigned __int8)sub_15CC8F0(*(_QWORD *)(a1 + 8), a3, v11, v8, v9)
    || (result = sub_15CC8F0(*(_QWORD *)(a1 + 8), a4, v11, v12, v13), (_BYTE)result) )
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v6 )
        return 1;
      v7 = sub_1648700(v6);
      v8 = *(unsigned __int8 *)(v7 + 16);
      if ( (unsigned __int8)(v8 - 25) <= 9u )
        goto LABEL_7;
    }
  }
  return result;
}
