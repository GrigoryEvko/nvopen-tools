// Function: sub_5E7390
// Address: 0x5e7390
//
__int64 __fastcall sub_5E7390(__int64 a1, _DWORD *a2)
{
  char v2; // al
  __int64 result; // rax
  __int64 j; // rbx
  _QWORD *i; // rbx

  if ( *(char *)(a1 + 192) >= 0 )
  {
    v2 = *(_BYTE *)(a1 + 88) & 0x8F;
    *(_BYTE *)(a1 + 88) = v2 | 0x20;
    if ( *(_DWORD *)(a1 + 160) )
    {
      *(_BYTE *)(a1 + 172) = 0;
      *(_BYTE *)(a1 + 88) = v2 | 0x24;
      sub_7604D0(a1, 11);
    }
    else
    {
      *(_BYTE *)(a1 + 172) = 1;
    }
  }
  sub_5E71C0(*(_QWORD *)(a1 + 152), a2);
  result = (unsigned int)*(unsigned __int8 *)(a1 + 174) - 1;
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 174) - 1) <= 1u )
  {
    for ( i = *(_QWORD **)(a1 + 176); i; i = (_QWORD *)*i )
      result = sub_5E7390(i[1], a2);
  }
  for ( j = *(_QWORD *)(a1 + 112); j; j = *(_QWORD *)(j + 112) )
  {
    if ( *(_QWORD *)(j + 272) != a1 )
      break;
    result = sub_5E7390(j, a2);
  }
  return result;
}
