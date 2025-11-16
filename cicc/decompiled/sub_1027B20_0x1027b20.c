// Function: sub_1027B20
// Address: 0x1027b20
//
__int64 __fastcall sub_1027B20(__int64 a1, __int64 a2)
{
  __int64 v2; // r12

  v2 = *(_QWORD *)(a1 + 176);
  if ( *(_BYTE *)(v2 + 280) )
    return sub_FF0A60(*(_QWORD *)(a1 + 176), a2);
  sub_FF9360(*(_QWORD **)(a1 + 176), *(_QWORD *)(v2 + 288), *(_QWORD *)(v2 + 296), *(__int64 **)(v2 + 304), 0, 0);
  *(_BYTE *)(v2 + 280) = 1;
  return sub_FF0A60(v2, a2);
}
