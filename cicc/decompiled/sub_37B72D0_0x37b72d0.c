// Function: sub_37B72D0
// Address: 0x37b72d0
//
bool __fastcall sub_37B72D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rcx
  __int64 v5; // rsi
  int v6; // edi
  unsigned int v7; // r9d
  __int64 *v8; // rax
  __int64 v9; // r10
  unsigned int v10; // r9d
  unsigned int v11; // r8d
  __int64 *v12; // rax
  __int64 v13; // r10
  unsigned int v14; // eax
  int v16; // eax
  int v17; // eax
  int v18; // r11d
  int v19; // r11d

  v4 = *(unsigned int *)(*(_QWORD *)a1 + 688LL);
  v5 = *(_QWORD *)(*(_QWORD *)a1 + 672LL);
  if ( !(_DWORD)v4 )
    return 0;
  v6 = v4 - 1;
  v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( a2 == *v8 )
  {
LABEL_3:
    v10 = *((_DWORD *)v8 + 2);
  }
  else
  {
    v16 = 1;
    while ( v9 != -4096 )
    {
      v18 = v16 + 1;
      v7 = v6 & (v16 + v7);
      v8 = (__int64 *)(v5 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      v16 = v18;
    }
    v10 = *(_DWORD *)(v5 + 16LL * (unsigned int)v4 + 8);
  }
  v11 = v6 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v12 = (__int64 *)(v5 + 16LL * v11);
  v13 = *v12;
  if ( a3 == *v12 )
  {
LABEL_5:
    v14 = *((_DWORD *)v12 + 2);
  }
  else
  {
    v17 = 1;
    while ( v13 != -4096 )
    {
      v19 = v17 + 1;
      v11 = v6 & (v17 + v11);
      v12 = (__int64 *)(v5 + 16LL * v11);
      v13 = *v12;
      if ( a3 == *v12 )
        goto LABEL_5;
      v17 = v19;
    }
    v14 = *(_DWORD *)(v5 + 16 * v4 + 8);
  }
  return v14 > v10;
}
