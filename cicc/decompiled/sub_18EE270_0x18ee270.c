// Function: sub_18EE270
// Address: 0x18ee270
//
__int64 __fastcall sub_18EE270(__int64 a1, __int64 *a2)
{
  __int64 v2; // r12
  __int64 *v3; // rbx
  __int64 *v4; // r14

  v2 = sub_15A0680(*(_QWORD *)a1, 0, 0);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v3 = *(__int64 **)(a1 - 8);
    v4 = &v3[3 * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)];
  }
  else
  {
    v4 = (__int64 *)a1;
    v3 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  }
  if ( v4 == v3 )
    return 1;
  while ( (unsigned int)sub_13F3450(a2, 0x27u, *v3, v2, a1) == 1 )
  {
    v3 += 3;
    if ( v4 == v3 )
      return 1;
  }
  return 0;
}
