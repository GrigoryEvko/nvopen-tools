// Function: sub_1A1E260
// Address: 0x1a1e260
//
char __fastcall sub_1A1E260(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // r13
  unsigned int v6; // r14d

  v3 = *(_QWORD *)(*(_QWORD *)sub_1649C60(*(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)))) + 24LL);
  if ( *(_BYTE *)(v3 + 8) == 13 && !byte_4FB3D80 )
  {
    LOBYTE(v4) = sub_1A1E0D0(v3, *(_QWORD *)a1);
    if ( (_BYTE)v4 )
      goto LABEL_9;
  }
  LOBYTE(v4) = 3 * (2 - *(_BYTE *)(a2 + 20));
  v5 = *(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( *(_BYTE *)(v5 + 16) != 13 )
    goto LABEL_5;
  v6 = *(_DWORD *)(v5 + 32);
  if ( v6 <= 0x40 )
  {
    v4 = *(_QWORD *)(v5 + 24);
    if ( !v4 )
      return v4;
    goto LABEL_5;
  }
  LODWORD(v4) = sub_16A57B0(v5 + 24);
  if ( v6 - (unsigned int)v4 > 0x40 )
  {
LABEL_5:
    if ( *(_BYTE *)(a1 + 344) )
      return v4;
    goto LABEL_9;
  }
  v4 = **(_QWORD **)(v5 + 24);
  if ( v4 && !*(_BYTE *)(a1 + 344) )
LABEL_9:
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(a1 + 8) & 3LL | a2 | 4;
  return v4;
}
