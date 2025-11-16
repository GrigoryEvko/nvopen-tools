// Function: sub_1B298A0
// Address: 0x1b298a0
//
char __fastcall sub_1B298A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  int v8; // edi
  unsigned int v9; // r9d
  __int64 *v10; // rdx
  __int64 v11; // r10
  __int64 *v12; // rax
  __int64 v13; // r9
  unsigned int v14; // r8d
  __int64 *v15; // rdx
  __int64 v16; // r10
  int v18; // edx
  int v19; // edx
  int v20; // r11d
  int v21; // r11d

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a3 + 40);
  if ( v3 != v4 )
  {
    v5 = *(_QWORD *)(a1 + 32);
    v6 = *(_QWORD *)(v5 + 32);
    v7 = *(unsigned int *)(v5 + 48);
    if ( !(_DWORD)v7 )
      goto LABEL_20;
    v8 = v7 - 1;
    v9 = (v7 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v3 == *v10 )
    {
LABEL_4:
      v12 = (__int64 *)(v6 + 16 * v7);
      if ( v10 != v12 )
      {
        v13 = v10[1];
        goto LABEL_6;
      }
    }
    else
    {
      v18 = 1;
      while ( v11 != -8 )
      {
        v21 = v18 + 1;
        v9 = v8 & (v18 + v9);
        v10 = (__int64 *)(v6 + 16LL * v9);
        v11 = *v10;
        if ( v3 == *v10 )
          goto LABEL_4;
        v18 = v21;
      }
      v12 = (__int64 *)(v6 + 16 * v7);
    }
    v13 = 0;
LABEL_6:
    v14 = v8 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v15 = (__int64 *)(v6 + 16LL * v14);
    v16 = *v15;
    if ( v4 == *v15 )
    {
LABEL_7:
      if ( v15 != v12 )
        return *(_DWORD *)(v13 + 48) < *(_DWORD *)(v15[1] + 48);
    }
    else
    {
      v19 = 1;
      while ( v16 != -8 )
      {
        v20 = v19 + 1;
        v14 = v8 & (v19 + v14);
        v15 = (__int64 *)(v6 + 16LL * v14);
        v16 = *v15;
        if ( v4 == *v15 )
          goto LABEL_7;
        v19 = v20;
      }
    }
LABEL_20:
    BUG();
  }
  return sub_1B29560(a1, a2, a3);
}
