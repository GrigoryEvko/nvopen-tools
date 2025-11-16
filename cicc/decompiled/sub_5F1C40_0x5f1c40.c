// Function: sub_5F1C40
// Address: 0x5f1c40
//
_BOOL8 __fastcall sub_5F1C40(__int64 a1)
{
  __int64 i; // rbx
  __int64 j; // r12
  __int64 *v3; // rax
  __int64 v4; // r13
  _BOOL8 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  char v8; // al

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  for ( j = sub_73D790(i); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v3 = *(__int64 **)(i + 168);
  v4 = v3[5];
  if ( !v4 )
  {
    v7 = *v3;
    if ( !v7 || (*(_BYTE *)(v7 + 35) & 1) == 0 )
      return 1;
    v4 = *(_QWORD *)(v7 + 8);
    if ( (unsigned int)sub_8D32E0(v4) )
      v4 = sub_8D46C0(v4);
    while ( 1 )
    {
      v8 = *(_BYTE *)(v4 + 140);
      if ( v8 != 12 )
        break;
      v4 = *(_QWORD *)(v4 + 160);
    }
    if ( !v8 )
      return 1;
  }
  result = 0;
  if ( j != v4 )
  {
    if ( !dword_4F07588 || (v6 = *(_QWORD *)(j + 32), *(_QWORD *)(v4 + 32) != v6) || !v6 )
    {
      if ( (unsigned __int8)(*(_BYTE *)(j + 140) - 9) > 2u )
        return (unsigned int)sub_8D2600(j) == 0;
      if ( !HIDWORD(qword_4D0495C) )
        return sub_8D5CE0(v4, j) == 0;
      return 1;
    }
  }
  return result;
}
