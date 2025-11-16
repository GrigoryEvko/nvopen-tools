// Function: sub_29AAD40
// Address: 0x29aad40
//
__int64 __fastcall sub_29AAD40(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // eax
  __int64 v4; // r9
  int v6; // edx
  unsigned int v7; // eax
  __int64 v8; // r10
  int v10; // r8d
  __int64 v11; // rax
  __int64 v12; // r8
  unsigned int v13; // edx
  __int64 *v14; // rdi
  __int64 v15; // r9
  int v16; // eax
  __int64 v17; // rsi
  int v18; // edx
  unsigned int v19; // eax
  __int64 v20; // rdi
  int v21; // r8d
  int v22; // edi
  int v23; // r11d

  v3 = *(_DWORD *)(a1 + 200);
  v4 = *(_QWORD *)(a1 + 184);
  if ( v3 )
  {
    v6 = v3 - 1;
    v7 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = *(_QWORD *)(v4 + 8LL * v7);
    if ( a2 == v8 )
      return 1;
    v10 = 1;
    while ( v8 != -4096 )
    {
      v7 = v6 & (v10 + v7);
      v8 = *(_QWORD *)(v4 + 8LL * v7);
      if ( a2 == v8 )
        return 1;
      ++v10;
    }
  }
  v11 = *(unsigned int *)(a1 + 168);
  v12 = *(_QWORD *)(a1 + 152);
  if ( (_DWORD)v11 )
  {
    v13 = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v14 = (__int64 *)(v12 + 40LL * v13);
    v15 = *v14;
    if ( a2 == *v14 )
    {
LABEL_8:
      if ( v14 != (__int64 *)(v12 + 40 * v11) )
      {
        v16 = *((_DWORD *)v14 + 8);
        v17 = v14[2];
        if ( v16 )
        {
          v18 = v16 - 1;
          v19 = (v16 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
          v20 = *(_QWORD *)(v17 + 8LL * v19);
          if ( a3 == v20 )
            return 1;
          v21 = 1;
          while ( v20 != -4096 )
          {
            v19 = v18 & (v21 + v19);
            v20 = *(_QWORD *)(v17 + 8LL * v19);
            if ( a3 == v20 )
              return 1;
            ++v21;
          }
        }
      }
    }
    else
    {
      v22 = 1;
      while ( v15 != -4096 )
      {
        v23 = v22 + 1;
        v13 = (v11 - 1) & (v22 + v13);
        v14 = (__int64 *)(v12 + 40LL * v13);
        v15 = *v14;
        if ( a2 == *v14 )
          goto LABEL_8;
        v22 = v23;
      }
    }
  }
  return 0;
}
