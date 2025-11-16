// Function: sub_1110F00
// Address: 0x1110f00
//
bool __fastcall sub_1110F00(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _BYTE *v3; // r12
  unsigned int v4; // r13d
  __int64 v5; // rax
  __int64 v6; // rdx
  _BYTE *v7; // rax

  v2 = *(_QWORD *)(a2 - 64);
  if ( !v2 )
    return 0;
  **(_QWORD **)a1 = v2;
  v3 = *(_BYTE **)(a2 - 32);
  if ( !v3 )
    BUG();
  if ( *v3 != 17 )
  {
    v6 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v3 + 1) + 8LL) - 17;
    if ( (unsigned int)v6 > 1 )
      return 0;
    if ( *v3 > 0x15u )
      return 0;
    v7 = sub_AD7630(*(_QWORD *)(a2 - 32), 0, v6);
    v3 = v7;
    if ( !v7 || *v7 != 17 )
      return 0;
  }
  v4 = *((_DWORD *)v3 + 8);
  if ( v4 > 0x40 )
  {
    if ( v4 - (unsigned int)sub_C444A0((__int64)(v3 + 24)) > 0x40 )
      return 0;
    v5 = **((_QWORD **)v3 + 3);
  }
  else
  {
    v5 = *((_QWORD *)v3 + 3);
  }
  return *(_QWORD *)(a1 + 8) == v5;
}
