// Function: sub_5EAA30
// Address: 0x5eaa30
//
__int64 __fastcall sub_5EAA30(__int64 a1, int a2)
{
  __int64 v2; // rbx
  __int64 result; // rax

  v2 = 0;
  if ( (*(_BYTE *)(a1 + 194) & 0x40) != 0 )
    v2 = *(_QWORD *)(a1 + 232);
  if ( !a2 )
    return sub_73BC00(*(_QWORD *)(v2 + 152), *(_QWORD *)(a1 + 152));
  *(_BYTE *)(***(_QWORD ***)(*(_QWORD *)(v2 + 152) + 168LL) + 32LL) &= ~4u;
  sub_73BC00(*(_QWORD *)(v2 + 152), *(_QWORD *)(a1 + 152));
  result = ***(_QWORD ***)(*(_QWORD *)(v2 + 152) + 168LL);
  *(_BYTE *)(result + 32) |= 4u;
  return result;
}
