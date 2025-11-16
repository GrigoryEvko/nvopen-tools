// Function: sub_2E501D0
// Address: 0x2e501d0
//
bool __fastcall sub_2E501D0(__int64 a1)
{
  int v1; // eax
  unsigned __int16 v2; // cx
  int v4; // eax
  __int64 v5; // rax

  v2 = *(_WORD *)(a1 + 68);
  LOBYTE(v1) = v2 == 68;
  if ( v2 <= 0x2Du )
    v1 |= (0x28000017D4FFuLL >> v2) & 1;
  if ( (_BYTE)v1 )
    return 0;
  if ( (unsigned __int8)sub_2E50190(a1, 20, 1)
    || (unsigned __int8)sub_2E50190(a1, 7, 1)
    || (unsigned __int8)sub_2E50190(a1, 9, 1)
    || (unsigned __int8)sub_2E50190(a1, 21, 1) && (*(_BYTE *)(a1 + 45) & 0x40) == 0
    || (unsigned __int8)sub_2E8B090(a1) )
  {
    return 0;
  }
  if ( (unsigned int)*(unsigned __int16 *)(a1 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 8) != 0
    || ((v4 = *(_DWORD *)(a1 + 44), (v4 & 4) != 0) || (v4 & 8) == 0
      ? (v5 = (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) >> 19) & 1LL)
      : (LOBYTE(v5) = sub_2E88A90(a1, 0x80000, 1)),
        (_BYTE)v5) )
  {
    if ( !(unsigned __int8)sub_2E8AED0(a1) )
      return 0;
  }
  return *(_WORD *)(a1 + 68) != 29;
}
