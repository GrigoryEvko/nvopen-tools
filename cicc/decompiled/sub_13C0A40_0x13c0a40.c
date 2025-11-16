// Function: sub_13C0A40
// Address: 0x13c0a40
//
_QWORD *__fastcall sub_13C0A40(__int64 a1, __int64 a2)
{
  __int64 v4; // rbx
  int v5; // eax
  __int64 v6; // rax
  unsigned __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdi
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // r9
  int v15; // edx
  int v16; // r10d

  v4 = **(_QWORD **)a2;
  v5 = *(_DWORD *)(a1 + 60);
  *(_DWORD *)(a1 + 56) = 0;
  if ( v5 )
  {
    v6 = 0;
  }
  else
  {
    sub_16CD150(a1 + 48, a1 + 64, 1, 8);
    v6 = 8LL * *(unsigned int *)(a1 + 56);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 48) + v6) = v4;
  v7 = 0;
  ++*(_DWORD *)(a1 + 56);
  v8 = *(unsigned int *)(a2 + 48);
  if ( (_DWORD)v8 )
  {
    v9 = *(_QWORD *)(a2 + 32);
    v10 = **(_QWORD **)(a1 + 48);
    v11 = (v8 - 1) & (((unsigned int)**(_QWORD **)(a1 + 48) >> 9) ^ ((unsigned int)v10 >> 4));
    v12 = (__int64 *)(v9 + 16LL * v11);
    v13 = *v12;
    if ( v10 == *v12 )
    {
LABEL_5:
      if ( v12 != (__int64 *)(v9 + 16 * v8) )
      {
        v7 = (unsigned __int64 *)v12[1];
        return sub_13BFEC0((_QWORD *)a1, a2, v7);
      }
    }
    else
    {
      v15 = 1;
      while ( v13 != -8 )
      {
        v16 = v15 + 1;
        v11 = (v8 - 1) & (v15 + v11);
        v12 = (__int64 *)(v9 + 16LL * v11);
        v13 = *v12;
        if ( v10 == *v12 )
          goto LABEL_5;
        v15 = v16;
      }
    }
    v7 = 0;
  }
  return sub_13BFEC0((_QWORD *)a1, a2, v7);
}
