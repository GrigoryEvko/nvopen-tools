// Function: sub_1AFC4C0
// Address: 0x1afc4c0
//
bool __fastcall sub_1AFC4C0(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rsi
  int v8; // edi
  unsigned int v9; // r9d
  __int64 *v10; // rdx
  __int64 v11; // r10
  __int64 *v12; // rax
  unsigned int v13; // r9d
  unsigned int v14; // r8d
  __int64 *v15; // rdx
  __int64 v16; // r10
  int v18; // edx
  int v19; // r11d
  int v20; // edx
  int v21; // r11d

  v5 = **a1;
  v6 = *(unsigned int *)(v5 + 48);
  if ( !(_DWORD)v6 )
    goto LABEL_16;
  v7 = *(_QWORD *)(v5 + 32);
  v8 = v6 - 1;
  v9 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( a2 != *v10 )
  {
    v20 = 1;
    while ( v11 != -8 )
    {
      v21 = v20 + 1;
      v9 = v8 & (v20 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        goto LABEL_3;
      v20 = v21;
    }
LABEL_16:
    BUG();
  }
LABEL_3:
  v12 = (__int64 *)(v7 + 16 * v6);
  if ( v10 == v12 )
    BUG();
  v13 = *(_DWORD *)(v10[1] + 16);
  v14 = v8 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v15 = (__int64 *)(v7 + 16LL * v14);
  v16 = *v15;
  if ( a3 != *v15 )
  {
    v18 = 1;
    while ( v16 != -8 )
    {
      v19 = v18 + 1;
      v14 = v8 & (v18 + v14);
      v15 = (__int64 *)(v7 + 16LL * v14);
      v16 = *v15;
      if ( a3 == *v15 )
        goto LABEL_5;
      v18 = v19;
    }
LABEL_17:
    BUG();
  }
LABEL_5:
  if ( v15 == v12 )
    goto LABEL_17;
  return v13 > *(_DWORD *)(v15[1] + 16);
}
