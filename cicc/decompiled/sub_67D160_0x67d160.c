// Function: sub_67D160
// Address: 0x67d160
//
__int64 __fastcall sub_67D160(__int64 a1, int a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v5; // rbx
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  int v9; // edx
  __int64 v10; // r15
  __int16 v11; // ax
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  int v15; // r12d

  v5 = qword_4CFFD90;
  if ( *(_QWORD *)qword_4CFFD90 != a1 )
  {
    v6 = *(_QWORD **)(qword_4CFFD90 + 16);
    if ( !v6 )
    {
LABEL_15:
      v3 = qword_4CFFD88;
      *(_QWORD *)(qword_4CFFD88 + 16) = qword_4CFFD90;
      *(_QWORD *)(v5 + 8) = v3;
      BUG();
    }
    while ( *v6 != a1 )
    {
      v6 = (_QWORD *)v6[2];
      if ( !v6 )
        goto LABEL_15;
    }
    v7 = qword_4CFFD88;
    qword_4CFFD90 = (__int64)v6;
    *(_QWORD *)(qword_4CFFD88 + 16) = v5;
    *(_QWORD *)(v5 + 8) = v7;
    v8 = v6[1];
    v5 = (__int64)v6;
    *(_QWORD *)(v8 + 16) = 0;
    qword_4CFFD88 = v8;
    v6[1] = 0;
  }
  v9 = *(__int16 *)(v5 + 24);
  v10 = *(_QWORD *)(v5 + 152);
  v11 = *(_WORD *)(v5 + 24);
  if ( v11 > 9 )
  {
    v13 = 0;
    v14 = *(unsigned int *)(v5 + 48) / 5uLL;
    while ( 1 )
    {
      v15 = v13;
      if ( *(unsigned int *)(v5 + 4 * v13 + 28) < v14 )
        break;
      ++v13;
      v14 += *(unsigned int *)(v5 + 48) / 5uLL;
      if ( v13 == 5 )
      {
        v15 = 5;
        break;
      }
    }
    v10 += 100;
    memmove((void *)(v5 + 4LL * (v15 + 7)), (const void *)(v5 + 4LL * (v15 + 7) + 4), 4LL * (unsigned int)(9 - v15));
    memmove((void *)(v5 + 8LL * (v15 + 9)), (const void *)(v5 + 8LL * (v15 + 9) + 8), 8LL * (unsigned int)(9 - v15));
    *(_DWORD *)(v5 + 64) = a2;
    *(_QWORD *)(v5 + 144) = a3;
    *(_QWORD *)(v5 + 152) = v10;
  }
  else
  {
    *(_DWORD *)(v5 + 4LL * *(__int16 *)(v5 + 24) + 28) = a2;
    *(_QWORD *)(v5 + 8LL * v9 + 72) = a3;
    *(_WORD *)(v5 + 24) = v11 + 1;
  }
  return (unsigned int)(a2 + v10);
}
