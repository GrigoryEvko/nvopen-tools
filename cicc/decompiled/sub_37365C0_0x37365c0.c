// Function: sub_37365C0
// Address: 0x37365c0
//
bool __fastcall sub_37365C0(_QWORD *a1)
{
  unsigned int v1; // eax
  bool result; // al
  __int64 v3; // rdi

  v1 = *(_DWORD *)(a1[10] + 36LL);
  if ( v1 == 1 )
    return 1;
  if ( v1 > 1 )
  {
    if ( v1 - 2 > 1 )
      BUG();
    return 0;
  }
  if ( *(_DWORD *)(a1[26] + 6224LL) != 1 )
    return 0;
  result = !sub_3736590(a1)
        && *(_DWORD *)(a1[10] + 32LL) != 3
        && (v3 = a1[26], *(_DWORD *)(v3 + 3764) != 2)
        && (unsigned __int16)sub_3220AA0(v3) <= 4u;
  return result;
}
