// Function: sub_2B1F720
// Address: 0x2b1f720
//
char __fastcall sub_2B1F720(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r14
  __int64 v4; // r12
  char result; // al
  int v7; // esi
  int v8; // eax
  unsigned int v9; // eax

  if ( a3 <= 1 )
    return 0;
  v3 = a2;
  v4 = a2;
  if ( (_BYTE)qword_5010508 && *(_BYTE *)(a2 + 8) == 17 )
    v3 = **(_QWORD **)(a2 + 16);
  result = sub_BCBCB0(v3);
  if ( !result || (*(_BYTE *)(v3 + 8) & 0xFD) == 4 )
  {
    if ( *(_BYTE *)(a2 + 8) != 17 )
      return 0;
    result = 1;
    if ( (a3 & (a3 - 1)) == 0 )
      return result;
    goto LABEL_9;
  }
  if ( (a3 & (a3 - 1)) != 0 )
  {
    v8 = *(unsigned __int8 *)(a2 + 8);
    if ( (_BYTE)v8 != 17 )
    {
      v7 = a3;
      if ( (unsigned int)(v8 - 17) > 1 )
        goto LABEL_16;
      goto LABEL_10;
    }
LABEL_9:
    v7 = a3 * *(_DWORD *)(a2 + 32);
LABEL_10:
    v4 = **(_QWORD **)(v4 + 16);
LABEL_16:
    sub_BCDA70((__int64 *)v4, v7);
    v9 = sub_DFDB60(a1);
    if ( v9 && a3 > v9 && ((a3 / v9) & (a3 / v9 - 1)) == 0 )
      return a3 % v9 == 0;
    return 0;
  }
  return result;
}
