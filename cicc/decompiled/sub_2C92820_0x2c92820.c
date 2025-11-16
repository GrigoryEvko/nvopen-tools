// Function: sub_2C92820
// Address: 0x2c92820
//
__int64 __fastcall sub_2C92820(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int16 v5; // ax
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  _QWORD *v11; // rbx
  __int64 v12; // r12

  v5 = *(_WORD *)(a1 + 24);
  v6 = a1;
  if ( v5 == 15 )
  {
LABEL_2:
    v7 = *(_QWORD *)(v6 - 8);
    a5 = 0;
    if ( *(_BYTE *)v7 == 85 )
    {
      v9 = *(_QWORD *)(v7 - 32);
      a5 = 1;
      if ( v9 )
      {
        if ( !*(_BYTE *)v9 && *(_QWORD *)(v9 + 24) == *(_QWORD *)(v7 + 80) )
          LOBYTE(a5) = (*(_BYTE *)(v9 + 33) & 0x20) == 0;
      }
    }
    return a5;
  }
  while ( 1 )
  {
    if ( !v5 )
      return 0;
    LOBYTE(a5) = v5 == 14 || (unsigned __int16)(v5 - 2) <= 2u;
    if ( (_BYTE)a5 )
    {
      v6 = *(_QWORD *)(v6 + 32);
      goto LABEL_6;
    }
    if ( v5 != 7 )
      break;
    if ( (unsigned __int8)sub_2C92820(*(_QWORD *)(v6 + 32)) )
      return 1;
    v6 = *(_QWORD *)(v6 + 40);
LABEL_6:
    v5 = *(_WORD *)(v6 + 24);
    if ( v5 == 15 )
      goto LABEL_2;
  }
  if ( (unsigned __int16)(v5 - 5) <= 1u || (unsigned __int16)(v5 - 8) <= 5u )
  {
    v10 = *(_QWORD *)(v6 + 40);
    if ( (_DWORD)v10 )
    {
      v11 = *(_QWORD **)(v6 + 32);
      v12 = (__int64)&v11[(unsigned int)(v10 - 1) + 1];
      while ( 1 )
      {
        a5 = sub_2C92820(*v11);
        if ( (_BYTE)a5 )
          break;
        if ( ++v11 == (_QWORD *)v12 )
          return a5;
      }
      return 1;
    }
  }
  return a5;
}
