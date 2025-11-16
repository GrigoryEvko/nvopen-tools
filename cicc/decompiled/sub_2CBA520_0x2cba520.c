// Function: sub_2CBA520
// Address: 0x2cba520
//
__int64 __fastcall sub_2CBA520(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rdx
  int v6; // r12d
  unsigned __int64 v7; // r15
  _QWORD *v8; // rax
  _QWORD *v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v13; // rdx
  __int64 v14; // rcx
  _QWORD v15[7]; // [rsp+18h] [rbp-38h] BYREF

  v15[0] = *(_QWORD *)(a2 + 120);
  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a2, a2, a3, a4);
    v4 = *(_QWORD *)(a2 + 96);
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
      sub_B2C6D0(a2, a2, v13, v14);
    v5 = *(_QWORD *)(a2 + 96);
  }
  else
  {
    v4 = *(_QWORD *)(a2 + 96);
    v5 = v4;
  }
  v6 = 2;
  v7 = v5 + 40LL * *(_QWORD *)(a2 + 104);
  if ( v4 == v7 )
    return 0;
  while ( 1 )
  {
    if ( *(_BYTE *)(*(_QWORD *)(v4 + 8) + 8LL) == 14 && !(unsigned __int8)sub_A74710(v15, v6, 81) )
    {
      v8 = *(_QWORD **)(a1 + 72);
      if ( !v8 )
        return 1;
      v9 = (_QWORD *)(a1 + 64);
      do
      {
        while ( 1 )
        {
          v10 = v8[2];
          v11 = v8[3];
          if ( v8[4] >= v4 )
            break;
          v8 = (_QWORD *)v8[3];
          if ( !v11 )
            goto LABEL_13;
        }
        v9 = v8;
        v8 = (_QWORD *)v8[2];
      }
      while ( v10 );
LABEL_13:
      if ( (_QWORD *)(a1 + 64) == v9 || v9[4] > v4 )
        return 1;
    }
    v4 += 40LL;
    ++v6;
    if ( v4 == v7 )
      return 0;
  }
}
