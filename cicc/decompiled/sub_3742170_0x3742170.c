// Function: sub_3742170
// Address: 0x3742170
//
__int64 __fastcall sub_3742170(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rax
  unsigned int v7; // esi
  __int64 *v8; // rdx
  __int64 v9; // r9
  int v11; // edx
  unsigned int v12; // r8d
  __int64 v13; // rdi
  int v14; // r11d
  __int64 *v15; // r10
  unsigned int v16; // esi
  __int64 *v17; // rax
  __int64 v18; // rdx
  int v19; // r10d
  int v20; // eax
  int v21; // edx
  int v22; // esi
  __int64 v23; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v24[5]; // [rsp+18h] [rbp-28h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 40);
  v23 = a2;
  v5 = *(_QWORD *)(v4 + 128);
  v6 = *(unsigned int *)(v4 + 144);
  if ( (_DWORD)v6 )
  {
    v7 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( v2 == *v8 )
    {
LABEL_3:
      if ( v8 != (__int64 *)(v5 + 16 * v6) )
        return *((unsigned int *)v8 + 2);
    }
    else
    {
      v11 = 1;
      while ( v9 != -4096 )
      {
        v19 = v11 + 1;
        v7 = (v6 - 1) & (v11 + v7);
        v8 = (__int64 *)(v5 + 16LL * v7);
        v9 = *v8;
        if ( v2 == *v8 )
          goto LABEL_3;
        v11 = v19;
      }
    }
  }
  v12 = *(_DWORD *)(a1 + 32);
  if ( !v12 )
  {
    ++*(_QWORD *)(a1 + 8);
    v24[0] = 0;
LABEL_26:
    v22 = 2 * v12;
LABEL_27:
    sub_3384500(a1 + 8, v22);
    sub_337AD60(a1 + 8, &v23, v24);
    v2 = v23;
    v15 = (__int64 *)v24[0];
    v21 = *(_DWORD *)(a1 + 24) + 1;
    goto LABEL_22;
  }
  v13 = *(_QWORD *)(a1 + 16);
  v14 = 1;
  v15 = 0;
  v16 = (v12 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v17 = (__int64 *)(v13 + 16LL * v16);
  v18 = *v17;
  if ( v2 == *v17 )
    return *((unsigned int *)v17 + 2);
  while ( v18 != -4096 )
  {
    if ( v18 == -8192 && !v15 )
      v15 = v17;
    v16 = (v12 - 1) & (v14 + v16);
    v17 = (__int64 *)(v13 + 16LL * v16);
    v18 = *v17;
    if ( v2 == *v17 )
      return *((unsigned int *)v17 + 2);
    ++v14;
  }
  if ( !v15 )
    v15 = v17;
  v20 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  v21 = v20 + 1;
  v24[0] = v15;
  if ( 4 * (v20 + 1) >= 3 * v12 )
    goto LABEL_26;
  if ( v12 - *(_DWORD *)(a1 + 28) - v21 <= v12 >> 3 )
  {
    v22 = v12;
    goto LABEL_27;
  }
LABEL_22:
  *(_DWORD *)(a1 + 24) = v21;
  if ( *v15 != -4096 )
    --*(_DWORD *)(a1 + 28);
  *v15 = v2;
  *((_DWORD *)v15 + 2) = 0;
  return *((unsigned int *)v15 + 2);
}
