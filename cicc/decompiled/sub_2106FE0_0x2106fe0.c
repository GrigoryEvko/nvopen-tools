// Function: sub_2106FE0
// Address: 0x2106fe0
//
_QWORD *__fastcall sub_2106FE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v8; // rbx
  int v9; // eax
  __int64 v10; // rax
  unsigned __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  unsigned int v15; // ecx
  __int64 *v16; // rdx
  __int64 v17; // r9
  int v19; // edx
  int v20; // r10d

  v8 = **(_QWORD **)a2;
  v9 = *(_DWORD *)(a1 + 60);
  *(_DWORD *)(a1 + 56) = 0;
  if ( v9 )
  {
    v10 = 0;
  }
  else
  {
    sub_16CD150(a1 + 48, (const void *)(a1 + 64), 1u, 8, a5, a6);
    v10 = 8LL * *(unsigned int *)(a1 + 56);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 48) + v10) = v8;
  v11 = 0;
  ++*(_DWORD *)(a1 + 56);
  v12 = *(unsigned int *)(a2 + 48);
  if ( (_DWORD)v12 )
  {
    v13 = *(_QWORD *)(a2 + 32);
    v14 = **(_QWORD **)(a1 + 48);
    v15 = (v12 - 1) & (((unsigned int)**(_QWORD **)(a1 + 48) >> 9) ^ ((unsigned int)v14 >> 4));
    v16 = (__int64 *)(v13 + 16LL * v15);
    v17 = *v16;
    if ( v14 == *v16 )
    {
LABEL_5:
      if ( v16 != (__int64 *)(v13 + 16 * v12) )
      {
        v11 = (unsigned __int64 *)v16[1];
        return sub_21064C0((_QWORD *)a1, a2, v11);
      }
    }
    else
    {
      v19 = 1;
      while ( v17 != -8 )
      {
        v20 = v19 + 1;
        v15 = (v12 - 1) & (v19 + v15);
        v16 = (__int64 *)(v13 + 16LL * v15);
        v17 = *v16;
        if ( v14 == *v16 )
          goto LABEL_5;
        v19 = v20;
      }
    }
    v11 = 0;
  }
  return sub_21064C0((_QWORD *)a1, a2, v11);
}
