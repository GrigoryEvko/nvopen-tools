// Function: sub_1E17D90
// Address: 0x1e17d90
//
char __fastcall sub_1E17D90(__int64 a1)
{
  __int64 v1; // rax
  __int16 v2; // dx
  __int16 v4; // ax
  __int64 v5; // rax

  v1 = *(_QWORD *)(a1 + 16);
  if ( *(_WORD *)v1 == 1 && (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 0x10) != 0 )
    return 1;
  v2 = *(_WORD *)(a1 + 46);
  if ( (v2 & 4) == 0 && (v2 & 8) != 0 )
  {
    if ( !sub_1E15D00(a1, 0x20000u, 1) )
      goto LABEL_8;
    return 1;
  }
  if ( (*(_QWORD *)(v1 + 8) & 0x20000LL) != 0 )
    return 1;
LABEL_8:
  v4 = *(_WORD *)(a1 + 46);
  if ( (v4 & 4) != 0 || (v4 & 8) == 0 )
    v5 = (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) >> 4) & 1LL;
  else
    LOBYTE(v5) = sub_1E15D00(a1, 0x10u, 1);
  if ( (_BYTE)v5 )
    return 1;
  return sub_1E17880(a1);
}
