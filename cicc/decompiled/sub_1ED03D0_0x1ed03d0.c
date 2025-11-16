// Function: sub_1ED03D0
// Address: 0x1ed03d0
//
__int64 __fastcall sub_1ED03D0(__int64 a1, unsigned __int64 *a2, __int64 **a3)
{
  int v4; // r13d
  unsigned __int64 v6; // r12
  __int64 v7; // r15
  int v8; // r13d
  int v9; // eax
  unsigned __int64 v10; // rsi
  int v11; // r10d
  __int64 *v12; // r9
  unsigned int i; // ecx
  __int64 *v14; // rdi
  __int64 v15; // rax
  float *v16; // rdx
  float *v17; // rax
  float *v18; // r8
  unsigned int v19; // ecx
  unsigned __int64 v20[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *a2;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = v4 - 1;
  v20[0] = sub_1ECD5B0(
             *(_QWORD **)(*a2 + 32),
             *(_QWORD *)(*a2 + 32) + 4LL * (unsigned int)(*(_DWORD *)(*a2 + 28) * *(_DWORD *)(*a2 + 24)));
  v9 = sub_1ECC960((_DWORD *)(v6 + 24), (int *)(v6 + 28), v20);
  v10 = *a2;
  v11 = 1;
  v12 = 0;
  for ( i = v8 & v9; ; i = v8 & v19 )
  {
    v14 = (__int64 *)(v7 + 8LL * i);
    v15 = *v14;
    if ( v10 <= 1 )
      break;
    if ( !v15 )
      goto LABEL_15;
    if ( v15 == 1 )
      goto LABEL_20;
    if ( *(_QWORD *)(v10 + 24) == *(_QWORD *)(v15 + 24) )
    {
      v16 = *(float **)(v15 + 32);
      v17 = *(float **)(v10 + 32);
      v18 = &v17[*(_DWORD *)(v10 + 28) * *(_DWORD *)(v10 + 24)];
      if ( v17 == v18 )
        goto LABEL_14;
      while ( *v17 == *v16 )
      {
        ++v17;
        ++v16;
        if ( v18 == v17 )
          goto LABEL_14;
      }
    }
LABEL_12:
    v19 = v11 + i;
    ++v11;
  }
  if ( v15 == v10 )
  {
LABEL_14:
    *a3 = v14;
    return 1;
  }
  if ( v15 )
  {
    if ( v15 != 1 )
      goto LABEL_12;
LABEL_20:
    if ( !v12 )
      v12 = (__int64 *)(v7 + 8LL * i);
    goto LABEL_12;
  }
LABEL_15:
  if ( !v12 )
    v12 = (__int64 *)(v7 + 8LL * i);
  *a3 = v12;
  return 0;
}
