// Function: sub_1ED01F0
// Address: 0x1ed01f0
//
__int64 __fastcall sub_1ED01F0(__int64 a1, unsigned __int64 *a2, __int64 **a3)
{
  int v4; // r12d
  unsigned __int64 v6; // r13
  __int64 v7; // r15
  int v8; // r12d
  int v9; // eax
  unsigned __int64 v10; // rsi
  int v11; // r10d
  __int64 *v12; // r9
  unsigned int i; // ecx
  __int64 *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r8
  float *v17; // rdx
  float *v18; // rax
  float *v19; // r8
  unsigned int v20; // ecx
  unsigned __int64 v21[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *a2;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = v4 - 1;
  v21[0] = sub_1ECD5B0(*(_QWORD **)(*a2 + 32), *(_QWORD *)(*a2 + 32) + 4LL * *(unsigned int *)(*a2 + 24));
  v9 = sub_18FDAA0((int *)(v6 + 24), (__int64 *)v21);
  v10 = *a2;
  v11 = 1;
  v12 = 0;
  for ( i = v8 & v9; ; i = v8 & v20 )
  {
    v14 = (__int64 *)(v7 + 8LL * i);
    v15 = *v14;
    if ( v10 <= 1 )
      break;
    if ( !v15 )
      goto LABEL_15;
    if ( v15 == 1 )
      goto LABEL_20;
    v16 = *(unsigned int *)(v10 + 24);
    if ( (_DWORD)v16 == *(_DWORD *)(v15 + 24) )
    {
      v17 = *(float **)(v15 + 32);
      v18 = *(float **)(v10 + 32);
      v19 = &v18[v16];
      if ( v18 == v19 )
        goto LABEL_14;
      while ( *v18 == *v17 )
      {
        ++v18;
        ++v17;
        if ( v19 == v18 )
          goto LABEL_14;
      }
    }
LABEL_12:
    v20 = v11 + i;
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
