// Function: sub_1FD4C00
// Address: 0x1fd4c00
//
__int64 __fastcall sub_1FD4C00(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdi
  unsigned int v7; // esi
  __int64 *v8; // rdx
  __int64 v9; // r9
  int v11; // edx
  unsigned int v12; // esi
  __int64 v13; // rdi
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r10
  int v17; // r10d
  int v18; // r11d
  __int64 *v19; // r9
  int v20; // eax
  int v21; // edx
  __int64 v22; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v23[5]; // [rsp+18h] [rbp-28h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 40);
  v22 = a2;
  v5 = *(unsigned int *)(v4 + 232);
  if ( (_DWORD)v5 )
  {
    v6 = *(_QWORD *)(v4 + 216);
    v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( v2 == *v8 )
    {
LABEL_3:
      if ( v8 != (__int64 *)(v6 + 16 * v5) )
        return *((unsigned int *)v8 + 2);
    }
    else
    {
      v11 = 1;
      while ( v9 != -8 )
      {
        v17 = v11 + 1;
        v7 = (v5 - 1) & (v11 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( v2 == *v8 )
          goto LABEL_3;
        v11 = v17;
      }
    }
  }
  v12 = *(_DWORD *)(a1 + 32);
  if ( !v12 )
  {
    ++*(_QWORD *)(a1 + 8);
LABEL_22:
    v12 *= 2;
    goto LABEL_23;
  }
  v13 = *(_QWORD *)(a1 + 16);
  v14 = (v12 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v15 = (__int64 *)(v13 + 16LL * v14);
  v16 = *v15;
  if ( v2 == *v15 )
    return *((unsigned int *)v15 + 2);
  v18 = 1;
  v19 = 0;
  while ( v16 != -8 )
  {
    if ( v16 == -16 && !v19 )
      v19 = v15;
    v14 = (v12 - 1) & (v18 + v14);
    v15 = (__int64 *)(v13 + 16LL * v14);
    v16 = *v15;
    if ( v2 == *v15 )
      return *((unsigned int *)v15 + 2);
    ++v18;
  }
  if ( !v19 )
    v19 = v15;
  v20 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  v21 = v20 + 1;
  if ( 4 * (v20 + 1) >= 3 * v12 )
    goto LABEL_22;
  if ( v12 - *(_DWORD *)(a1 + 28) - v21 <= v12 >> 3 )
  {
LABEL_23:
    sub_1542080(a1 + 8, v12);
    sub_154CC80(a1 + 8, &v22, v23);
    v19 = (__int64 *)v23[0];
    v2 = v22;
    v21 = *(_DWORD *)(a1 + 24) + 1;
  }
  *(_DWORD *)(a1 + 24) = v21;
  if ( *v19 != -8 )
    --*(_DWORD *)(a1 + 28);
  *v19 = v2;
  *((_DWORD *)v19 + 2) = 0;
  return 0;
}
