// Function: sub_1E41F20
// Address: 0x1e41f20
//
__int64 __fastcall sub_1E41F20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rdx
  __int64 *v8; // rdi
  __int64 v9; // rax
  __int64 *v10; // rbx
  __int64 *v11; // rax
  __int64 v12; // rsi
  __int64 *v13; // r13
  __int64 v14; // r12
  __int64 *v15; // rax
  unsigned int v16; // eax
  __int64 v17; // rdx

  v7 = *(_QWORD *)(a1 + 2352);
  v8 = *(__int64 **)(a1 + 2360);
  if ( v8 == (__int64 *)v7 )
    v9 = *(unsigned int *)(a1 + 2372);
  else
    v9 = *(unsigned int *)(a1 + 2368);
  v10 = &v8[v9];
  if ( v8 == v10 )
  {
LABEL_7:
    v14 = a1 + 2344;
  }
  else
  {
    v11 = v8;
    while ( 1 )
    {
      v12 = *v11;
      v13 = v11;
      if ( (unsigned __int64)*v11 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v10 == ++v11 )
        goto LABEL_7;
    }
    v14 = a1 + 2344;
    if ( v11 != v10 )
    {
      do
      {
        sub_1E0A0F0(*(_QWORD *)(a1 + 32), v12, v7, a4, a5, a6);
        v15 = v13 + 1;
        if ( v13 + 1 == v10 )
          break;
        while ( 1 )
        {
          v12 = *v15;
          v13 = v15;
          if ( (unsigned __int64)*v15 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v10 == ++v15 )
            goto LABEL_12;
        }
      }
      while ( v15 != v10 );
LABEL_12:
      v8 = *(__int64 **)(a1 + 2360);
      v7 = *(_QWORD *)(a1 + 2352);
    }
  }
  ++*(_QWORD *)(a1 + 2344);
  if ( v8 != (__int64 *)v7 )
  {
    v16 = 4 * (*(_DWORD *)(a1 + 2372) - *(_DWORD *)(a1 + 2376));
    v17 = *(unsigned int *)(a1 + 2368);
    if ( v16 < 0x20 )
      v16 = 32;
    if ( (unsigned int)v17 > v16 )
    {
      sub_16CC920(v14);
      return sub_1F03420(a1);
    }
    memset(v8, -1, 8 * v17);
  }
  *(_QWORD *)(a1 + 2372) = 0;
  return sub_1F03420(a1);
}
