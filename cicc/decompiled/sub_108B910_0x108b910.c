// Function: sub_108B910
// Address: 0x108b910
//
__int64 __fastcall sub_108B910(__int64 a1)
{
  _QWORD *v2; // rax

  if ( *(_QWORD *)a1 )
    return *(_QWORD *)(*(_QWORD *)a1 + 8LL);
  if ( (*(_BYTE *)(a1 + 9) & 0x70) == 0x20 && *(char *)(a1 + 8) >= 0 )
  {
    *(_BYTE *)(a1 + 8) |= 8u;
    v2 = sub_E807D0(*(_QWORD *)(a1 + 24));
    *(_QWORD *)a1 = v2;
    if ( v2 )
      return v2[1];
  }
  return sub_EA1870(a1);
}
