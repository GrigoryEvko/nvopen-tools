// Function: sub_15F2E00
// Address: 0x15f2e00
//
__int64 __fastcall sub_15F2E00(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v4; // rax
  __int64 v5; // rsi

  v2 = *(_QWORD *)(a1 + 8);
  if ( !v2 )
    return 0;
  while ( 1 )
  {
    v4 = sub_1648700(v2);
    if ( *(_BYTE *)(v4 + 16) == 77 )
      break;
    if ( a2 != *(_QWORD *)(v4 + 40) )
      return 1;
LABEL_6:
    v2 = *(_QWORD *)(v2 + 8);
    if ( !v2 )
      return 0;
  }
  if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
    v5 = *(_QWORD *)(v4 - 8);
  else
    v5 = v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
  if ( a2 == *(_QWORD *)(v5
                       + 0xFFFFFFFD55555558LL * (unsigned int)((v2 - v5) >> 3)
                       + 24LL * *(unsigned int *)(v4 + 56)
                       + 8) )
    goto LABEL_6;
  return 1;
}
