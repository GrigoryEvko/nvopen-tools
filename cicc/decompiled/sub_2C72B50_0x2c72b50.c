// Function: sub_2C72B50
// Address: 0x2c72b50
//
__int64 __fastcall sub_2C72B50(__int64 a1, char a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v4; // rax
  int v5; // eax
  __int64 v6; // rsi
  int v7; // r8d
  unsigned int v8; // ecx
  __int64 *v9; // rax
  __int64 v10; // r9
  __int64 v11; // rax
  int v13; // eax
  int v14; // r10d

  v3 = 0;
  v4 = *(_QWORD *)(a1 + 56);
  if ( !v4 )
    goto LABEL_16;
  if ( *(_BYTE *)(v4 - 24) == 84 )
  {
    while ( 1 )
    {
      v4 = *(_QWORD *)(v4 + 8);
      ++v3;
      if ( !v4 )
        break;
      if ( *(_BYTE *)(v4 - 24) != 84 )
        goto LABEL_5;
    }
LABEL_16:
    BUG();
  }
LABEL_5:
  if ( a2 )
  {
    v5 = *(_DWORD *)(a3 + 24);
    v6 = *(_QWORD *)(a3 + 8);
    if ( v5 )
    {
      v7 = v5 - 1;
      v8 = (v5 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v9 = (__int64 *)(v6 + 16LL * v8);
      v10 = *v9;
      if ( a1 == *v9 )
      {
LABEL_8:
        v11 = *(_QWORD *)v9[1];
        if ( v11 )
          v3 += sub_2C72B50(**(_QWORD **)(v11 + 32), 1);
        return v3;
      }
      v13 = 1;
      while ( v10 != -4096 )
      {
        v14 = v13 + 1;
        v8 = v7 & (v13 + v8);
        v9 = (__int64 *)(v6 + 16LL * v8);
        v10 = *v9;
        if ( a1 == *v9 )
          goto LABEL_8;
        v13 = v14;
      }
    }
    BUG();
  }
  return v3;
}
