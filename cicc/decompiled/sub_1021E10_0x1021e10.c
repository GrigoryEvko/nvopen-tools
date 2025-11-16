// Function: sub_1021E10
// Address: 0x1021e10
//
__int64 __fastcall sub_1021E10(__int64 a1, __int64 a2)
{
  _BYTE **v2; // r13
  __int64 v4; // rax
  _BYTE **v5; // rbx
  _BYTE *v6; // rsi
  _QWORD *v7; // rax
  _QWORD *v8; // rdx

  v2 = (_BYTE **)a1;
  v4 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v5 = *(_BYTE ***)(a1 - 8);
    v2 = &v5[v4];
  }
  else
  {
    v5 = (_BYTE **)(a1 - v4 * 8);
  }
  if ( v5 == v2 )
    return 1;
  while ( 1 )
  {
    v6 = *v5;
    if ( **v5 <= 0x1Cu )
      v6 = 0;
    if ( !*(_BYTE *)(a2 + 28) )
      break;
    v7 = *(_QWORD **)(a2 + 8);
    v8 = &v7[*(unsigned int *)(a2 + 20)];
    if ( v7 == v8 )
      return 0;
    while ( v6 != (_BYTE *)*v7 )
    {
      if ( v8 == ++v7 )
        return 0;
    }
LABEL_11:
    v5 += 4;
    if ( v2 == v5 )
      return 1;
  }
  if ( sub_C8CA60(a2, (__int64)v6) )
    goto LABEL_11;
  return 0;
}
