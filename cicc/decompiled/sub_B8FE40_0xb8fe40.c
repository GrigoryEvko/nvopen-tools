// Function: sub_B8FE40
// Address: 0xb8fe40
//
__int64 __fastcall sub_B8FE40(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  __int64 v7; // rsi
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rdi
  int v14; // ecx
  unsigned int v15; // edx
  __int64 *v16; // r8
  __int64 v17; // r9
  int v18; // r8d
  int v19; // r10d

  v6 = *a1;
  v7 = *a2;
  if ( *(_BYTE *)(v6 + 28) )
  {
    v8 = *(_QWORD **)(v6 + 8);
    v9 = &v8[*(unsigned int *)(v6 + 20)];
    if ( v8 == v9 )
      goto LABEL_8;
    while ( v7 != *v8 )
    {
      if ( v9 == ++v8 )
        goto LABEL_8;
    }
    return 0;
  }
  if ( sub_C8CA60(v6, v7, a3, a4) )
    return 0;
LABEL_8:
  v11 = a1[1];
  v12 = *(_DWORD *)(v11 + 24);
  v13 = *(_QWORD *)(v11 + 8);
  if ( v12 )
  {
    v14 = v12 - 1;
    v15 = (v12 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
    v16 = (__int64 *)(v13 + 8LL * v15);
    v17 = *v16;
    if ( *v16 == *a2 )
    {
LABEL_10:
      *v16 = -8192;
      --*(_DWORD *)(v11 + 16);
      ++*(_DWORD *)(v11 + 20);
    }
    else
    {
      v18 = 1;
      while ( v17 != -4096 )
      {
        v19 = v18 + 1;
        v15 = v14 & (v18 + v15);
        v16 = (__int64 *)(v13 + 8LL * v15);
        v17 = *v16;
        if ( *a2 == *v16 )
          goto LABEL_10;
        v18 = v19;
      }
    }
  }
  return 1;
}
