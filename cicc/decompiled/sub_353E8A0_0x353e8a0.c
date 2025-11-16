// Function: sub_353E8A0
// Address: 0x353e8a0
//
__int64 __fastcall sub_353E8A0(__int64 **a1, __int64 *a2)
{
  __int64 v2; // rcx
  __int64 v3; // rax
  __int64 v4; // rsi
  int v5; // eax
  int v6; // eax
  unsigned int v7; // r9d
  __int64 v8; // r8
  __int64 *v9; // rax
  int v10; // esi
  int v11; // esi
  __int64 v12; // r9
  unsigned int v13; // edx
  __int64 *v14; // rdi
  __int64 v15; // r8
  int v17; // r10d
  int v18; // edi
  int v19; // r10d

  v2 = *a2;
  v3 = **a1;
  v4 = *(_QWORD *)(v3 + 8);
  v5 = *(_DWORD *)(v3 + 24);
  if ( !v5 )
    return 0;
  v6 = v5 - 1;
  v7 = v6 & (((unsigned int)v2 >> 4) ^ ((unsigned int)v2 >> 9));
  v8 = *(_QWORD *)(v4 + 8LL * v7);
  if ( v2 != v8 )
  {
    v17 = 1;
    while ( v8 != -4096 )
    {
      v7 = v6 & (v17 + v7);
      v8 = *(_QWORD *)(v4 + 8LL * v7);
      if ( v2 == v8 )
        goto LABEL_3;
      ++v17;
    }
    return 0;
  }
LABEL_3:
  v9 = a1[1];
  v10 = *((_DWORD *)v9 + 6);
  if ( v10 )
  {
    v11 = v10 - 1;
    v12 = v9[1];
    v13 = v11 & (((unsigned int)v2 >> 4) ^ ((unsigned int)v2 >> 9));
    v14 = (__int64 *)(v12 + 8LL * v13);
    v15 = *v14;
    if ( v2 == *v14 )
    {
LABEL_5:
      *v14 = -8192;
      --*((_DWORD *)v9 + 4);
      ++*((_DWORD *)v9 + 5);
    }
    else
    {
      v18 = 1;
      while ( v15 != -4096 )
      {
        v19 = v18 + 1;
        v13 = v11 & (v18 + v13);
        v14 = (__int64 *)(v12 + 8LL * v13);
        v15 = *v14;
        if ( v2 == *v14 )
          goto LABEL_5;
        v18 = v19;
      }
    }
  }
  return 1;
}
