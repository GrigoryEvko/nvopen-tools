// Function: sub_7E45A0
// Address: 0x7e45a0
//
_BYTE *__fastcall sub_7E45A0(__int64 *a1)
{
  _BYTE *v1; // r12
  __int64 i; // rbx
  __int64 v3; // rbx

  v1 = a1;
  for ( i = *a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  sub_7E3EE0(i);
  while ( 1 )
  {
    v3 = sub_7E05E0(*(_QWORD *)(i + 160), 0);
    v1 = sub_73DE50((__int64)v1, v3);
    if ( !(unsigned int)sub_8D3A70(*(_QWORD *)(v3 + 120)) )
      break;
    i = *(_QWORD *)(v3 + 120);
  }
  return v1;
}
