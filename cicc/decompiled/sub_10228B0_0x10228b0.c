// Function: sub_10228B0
// Address: 0x10228b0
//
__int64 __fastcall sub_10228B0(__int64 a1, __int64 a2, unsigned int a3)
{
  _BYTE **v4; // r14
  __int64 v6; // rax
  _BYTE **v7; // rbx
  unsigned int v8; // r13d
  _BYTE *v9; // rsi
  _QWORD *v10; // rax
  _QWORD *v11; // rdx

  v4 = (_BYTE **)a1;
  v6 = 4LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v7 = *(_BYTE ***)(a1 - 8);
    v4 = &v7[v6];
  }
  else
  {
    v7 = (_BYTE **)(a1 - v6 * 8);
  }
  v8 = 0;
  if ( v4 == v7 )
    return 0;
  while ( 1 )
  {
    v9 = *v7;
    if ( **v7 <= 0x1Cu )
      v9 = 0;
    if ( !*(_BYTE *)(a2 + 28) )
      break;
    v10 = *(_QWORD **)(a2 + 8);
    v11 = &v10[*(unsigned int *)(a2 + 20)];
    if ( v10 != v11 )
    {
      while ( v9 != (_BYTE *)*v10 )
      {
        if ( v11 == ++v10 )
          goto LABEL_12;
      }
      goto LABEL_11;
    }
LABEL_12:
    if ( v8 > a3 )
      return 1;
LABEL_13:
    v7 += 4;
    if ( v4 == v7 )
      return 0;
  }
  if ( sub_C8CA60(a2, (__int64)v9) )
  {
LABEL_11:
    ++v8;
    goto LABEL_12;
  }
  if ( v8 <= a3 )
    goto LABEL_13;
  return 1;
}
