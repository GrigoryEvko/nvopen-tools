// Function: sub_14B5110
// Address: 0x14b5110
//
char __fastcall sub_14B5110(__int64 *a1, __int64 a2)
{
  char v2; // al

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 52 )
  {
    if ( !(unsigned __int8)sub_14B35A0(a1, *(_QWORD *)(a2 - 48)) || !(unsigned __int8)sub_14A9710(*(_QWORD *)(a2 - 24)) )
    {
      if ( (unsigned __int8)sub_14B35A0(a1, *(_QWORD *)(a2 - 24)) )
        return sub_14A9710(*(_QWORD *)(a2 - 48));
      return 0;
    }
    return 1;
  }
  if ( v2 != 5 || *(_WORD *)(a2 + 18) != 28 )
    return 0;
  if ( (unsigned __int8)sub_14B4CC0(a1, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)))
    && sub_14A9880(*(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)))) )
  {
    return 1;
  }
  if ( !(unsigned __int8)sub_14B4CC0(a1, *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)))) )
    return 0;
  return sub_14A9880(*(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
}
