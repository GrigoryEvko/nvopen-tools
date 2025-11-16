// Function: sub_13A3940
// Address: 0x13a3940
//
bool __fastcall sub_13A3940(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 *v3; // r12
  __int64 v4; // r12
  unsigned int v6; // ebx
  __int64 v7; // r13

  v2 = *(_DWORD *)(a1 + 8);
  v3 = *(__int64 **)a1;
  if ( v2 <= 0x40 )
  {
    v4 = (__int64)((_QWORD)v3 << (64 - (unsigned __int8)v2)) >> (64 - (unsigned __int8)v2);
    return a2 > v4;
  }
  v6 = v2 + 1;
  v7 = v3[(v2 - 1) >> 6] & (1LL << ((unsigned __int8)v2 - 1));
  if ( v7 )
  {
    if ( v6 - (unsigned int)sub_16A5810() > 0x40 )
      return v7 != 0;
LABEL_8:
    v4 = *v3;
    return a2 > v4;
  }
  if ( v6 - (unsigned int)sub_16A57B0(a1) <= 0x40 )
    goto LABEL_8;
  return v7 != 0;
}
