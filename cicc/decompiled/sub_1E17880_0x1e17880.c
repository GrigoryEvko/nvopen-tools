// Function: sub_1E17880
// Address: 0x1e17880
//
char __fastcall sub_1E17880(__int64 a1)
{
  __int16 v1; // ax
  __int64 v2; // rax

  v1 = *(_WORD *)(a1 + 46);
  if ( (v1 & 4) == 0 && (v1 & 8) != 0 )
    LOBYTE(v2) = sub_1E15D00(a1, 0x100000u, 1);
  else
    v2 = (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) >> 20) & 1LL;
  if ( !(_BYTE)v2 && **(_WORD **)(a1 + 16) == 1 )
    return *(_QWORD *)(*(_QWORD *)(a1 + 32) + 64LL) & 1LL;
  return v2;
}
