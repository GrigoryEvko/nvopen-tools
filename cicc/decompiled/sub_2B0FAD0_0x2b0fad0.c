// Function: sub_2B0FAD0
// Address: 0x2b0fad0
//
bool __fastcall sub_2B0FAD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r11
  int v6; // r9d
  int v7; // ebx
  __int64 *v8; // r10
  unsigned int v9; // eax
  __int64 *v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rdx
  int v14; // ecx
  int v15; // r12d

  if ( a2 == a1 )
    return 0;
  v4 = *(unsigned int *)(a3 + 24);
  v5 = *(_QWORD *)(a3 + 8);
  v6 = v4;
  v7 = v4 - 1;
  v8 = (__int64 *)(v5 + 8 * v4);
  while ( 1 )
  {
    v12 = *(_QWORD *)(a1 + 24);
    if ( !v6 )
      return a2 != a1;
    v9 = v7 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    v10 = (__int64 *)(v5 + 8LL * v9);
    v11 = *v10;
    if ( v12 != *v10 )
    {
      v14 = 1;
      while ( v11 != -4096 )
      {
        v15 = v14 + 1;
        v9 = v7 & (v14 + v9);
        v10 = (__int64 *)(v5 + 8LL * v9);
        v11 = *v10;
        if ( v12 == *v10 )
          goto LABEL_4;
        v14 = v15;
      }
      return a2 != a1;
    }
LABEL_4:
    if ( v8 != v10 )
    {
      a1 = *(_QWORD *)(a1 + 8);
      if ( a2 != a1 )
        continue;
    }
    return a2 != a1;
  }
}
