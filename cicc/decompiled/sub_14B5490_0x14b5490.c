// Function: sub_14B5490
// Address: 0x14b5490
//
char __fastcall sub_14B5490(_QWORD *a1, __int64 a2)
{
  char v2; // al

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 52 )
  {
    if ( !(unsigned __int8)sub_14B39F0(a1, *(_QWORD *)(a2 - 48)) || !(unsigned __int8)sub_14A9710(*(_QWORD *)(a2 - 24)) )
    {
      if ( (unsigned __int8)sub_14B39F0(a1, *(_QWORD *)(a2 - 24)) )
        return sub_14A9710(*(_QWORD *)(a2 - 48));
      return 0;
    }
    return 1;
  }
  if ( v2 != 5 || *(_WORD *)(a2 + 18) != 28 )
    return 0;
  if ( (unsigned __int8)sub_14B5220(a1, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)))
    && sub_14A9880(*(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)))) )
  {
    return 1;
  }
  if ( !(unsigned __int8)sub_14B5220(a1, *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)))) )
    return 0;
  return sub_14A9880(*(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
}
