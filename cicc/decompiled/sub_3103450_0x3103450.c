// Function: sub_3103450
// Address: 0x3103450
//
__int64 __fastcall sub_3103450(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 *v3; // rbx
  __int64 *i; // r13
  unsigned __int8 v5; // al
  bool v6; // zf

  v2 = sub_98CE00(**(_QWORD **)(a2 + 32)) ^ 1;
  *(_BYTE *)(a1 + 41) = v2;
  *(_BYTE *)(a1 + 40) = v2;
  v3 = *(__int64 **)(a2 + 40);
  for ( i = (__int64 *)(*(_QWORD *)(a2 + 32) + 8LL); v3 != i; ++i )
  {
    v5 = sub_98CE00(*i) ^ 1;
    v6 = (v5 | *(_BYTE *)(a1 + 40)) == 0;
    *(_BYTE *)(a1 + 40) |= v5;
    if ( !v6 )
      break;
  }
  return sub_3103310(a1, a2);
}
