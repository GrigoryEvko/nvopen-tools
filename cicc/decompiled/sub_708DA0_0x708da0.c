// Function: sub_708DA0
// Address: 0x708da0
//
__int64 __fastcall sub_708DA0(__int64 a1)
{
  __int64 result; // rax
  __int64 i; // rbx
  __int64 j; // rbx
  __int64 k; // rbx

  result = sub_708EB0(*(_QWORD *)(a1 + 104));
  for ( i = *(_QWORD *)(a1 + 144); i; i = *(_QWORD *)(i + 112) )
  {
    while ( 1 )
    {
      if ( (*(_DWORD *)(i + 192) & 0x8000400) == 0 )
      {
        result = sub_736A10(i);
        if ( (_DWORD)result )
          break;
      }
      i = *(_QWORD *)(i + 112);
      if ( !i )
        goto LABEL_7;
    }
    result = sub_708CD0(i, 0xBu);
  }
LABEL_7:
  for ( j = *(_QWORD *)(a1 + 112); j; j = *(_QWORD *)(j + 112) )
  {
    if ( (*(_BYTE *)(j + 170) & 0x60) == 0 && *(_BYTE *)(j + 177) != 5 )
    {
      result = sub_736A30(j);
      if ( (_DWORD)result )
        result = sub_708CD0(j, 7u);
    }
  }
  for ( k = *(_QWORD *)(a1 + 168); k; k = *(_QWORD *)(k + 112) )
  {
    if ( (*(_BYTE *)(k + 124) & 1) == 0 )
      result = sub_708DA0(*(_QWORD *)(k + 128));
  }
  if ( !*(_BYTE *)(a1 + 28) )
    return sub_76C540(a1, sub_708EB0);
  return result;
}
