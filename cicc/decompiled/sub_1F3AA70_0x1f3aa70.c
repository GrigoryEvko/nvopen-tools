// Function: sub_1F3AA70
// Address: 0x1f3aa70
//
char __fastcall sub_1F3AA70(__int64 a1, __int64 a2)
{
  __int16 v2; // ax
  __int64 v3; // rax
  __int16 v4; // ax

  v2 = *(_WORD *)(a2 + 46);
  if ( (v2 & 4) == 0 && (v2 & 8) != 0 )
  {
    LOBYTE(v3) = sub_1E15D00(a2, 8u, 1);
    if ( !(_BYTE)v3 )
      return v3;
  }
  else
  {
    v3 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) >> 3) & 1LL;
    if ( (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) & 8LL) == 0 )
      return v3;
  }
  v4 = *(_WORD *)(a2 + 46);
  if ( (v4 & 4) != 0 || (v4 & 8) == 0 )
    return (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) >> 4) & 1LL;
  else
    LOBYTE(v3) = sub_1E15D00(a2, 0x10u, 1);
  return v3;
}
