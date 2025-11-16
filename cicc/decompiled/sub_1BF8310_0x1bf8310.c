// Function: sub_1BF8310
// Address: 0x1bf8310
//
__int64 __fastcall sub_1BF8310(__int64 a1, char a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v4; // rax
  int v6; // eax
  int v7; // r8d
  __int64 v8; // rsi
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // rax
  int v13; // eax
  int v14; // r10d

  v3 = 0;
  v4 = *(_QWORD *)(a1 + 48);
  if ( !v4 )
    goto LABEL_15;
  if ( *(_BYTE *)(v4 - 8) == 77 )
  {
    while ( 1 )
    {
      v4 = *(_QWORD *)(v4 + 8);
      ++v3;
      if ( !v4 )
        break;
      if ( *(_BYTE *)(v4 - 8) != 77 )
        goto LABEL_5;
    }
LABEL_15:
    BUG();
  }
LABEL_5:
  if ( !a2 )
    return v3;
  v6 = *(_DWORD *)(a3 + 24);
  if ( !v6 )
LABEL_16:
    BUG();
  v7 = v6 - 1;
  v8 = *(_QWORD *)(a3 + 8);
  v9 = (v6 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v10 = (__int64 *)(v8 + 16LL * v9);
  v11 = *v10;
  if ( a1 != *v10 )
  {
    v13 = 1;
    while ( v11 != -8 )
    {
      v14 = v13 + 1;
      v9 = v7 & (v13 + v9);
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( a1 == *v10 )
        goto LABEL_9;
      v13 = v14;
    }
    goto LABEL_16;
  }
LABEL_9:
  v12 = *(_QWORD *)v10[1];
  if ( !v12 )
    return v3;
  return (unsigned int)sub_1BF8310(**(_QWORD **)(v12 + 32), 1, a3) + v3;
}
