// Function: sub_7AB730
// Address: 0x7ab730
//
__int64 __fastcall sub_7AB730(__int64 a1, _DWORD *a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax

  *a2 = 1;
  if ( dword_4F04C44 == -1 )
    return 0;
  v2 = *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C44 + 512);
  if ( !v2 )
    return 0;
  v3 = *(_QWORD *)(v2 + 88);
  if ( (*(_BYTE *)(v3 + 89) & 4) != 0 )
    goto LABEL_6;
LABEL_12:
  if ( !*(_QWORD *)(*(_QWORD *)(v3 + 168) + 168LL) )
    return 0;
  while ( 1 )
  {
    v4 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v3 + 96LL) + 72LL);
    v5 = *(_QWORD *)(*(_QWORD *)(v4 + 88) + 152LL);
    if ( !v5 )
      v5 = v4;
    if ( v5 == a1 )
      return 1;
    if ( (*(_BYTE *)(v5 + 81) & 0x10) == 0 )
      return 0;
    v6 = **(_QWORD **)(v5 + 64);
    *a2 = 0;
    v3 = *(_QWORD *)(v6 + 88);
    if ( (*(_BYTE *)(v3 + 89) & 4) == 0 )
      goto LABEL_12;
LABEL_6:
    while ( !*(_QWORD *)(*(_QWORD *)(v3 + 168) + 168LL) )
    {
      v3 = *(_QWORD *)(*(_QWORD *)(v3 + 40) + 32LL);
      *a2 = 0;
      if ( (*(_BYTE *)(v3 + 89) & 4) == 0 )
        goto LABEL_12;
    }
  }
}
