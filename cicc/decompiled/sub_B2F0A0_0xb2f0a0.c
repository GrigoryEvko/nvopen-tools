// Function: sub_B2F0A0
// Address: 0xb2f0a0
//
__int64 __fastcall sub_B2F0A0(__int64 a1, char a2)
{
  unsigned int v2; // r12d
  __int64 v4; // rax
  int v5; // eax

  if ( *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) != 14 )
    return 0;
  v2 = sub_B2D640(*(_QWORD *)(a1 + 24), *(_DWORD *)(a1 + 32), 43);
  if ( !(_BYTE)v2 || !a2 && !(unsigned __int8)sub_B2D640(*(_QWORD *)(a1 + 24), *(_DWORD *)(a1 + 32), 40) )
  {
    if ( !sub_B2BD50(a1) )
      return 0;
    v4 = *(_QWORD *)(a1 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
      v4 = **(_QWORD **)(v4 + 16);
    LOBYTE(v5) = sub_B2F070(*(_QWORD *)(a1 + 24), *(_DWORD *)(v4 + 8) >> 8);
    return v5 ^ 1u;
  }
  return v2;
}
