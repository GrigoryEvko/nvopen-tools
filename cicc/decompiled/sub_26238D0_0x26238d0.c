// Function: sub_26238D0
// Address: 0x26238d0
//
__int64 *__fastcall sub_26238D0(__int64 a1, __int64 *a2)
{
  unsigned int v4; // esi
  __int64 v5; // r9
  __int64 *v6; // rcx
  int v7; // r11d
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r8
  int v12; // eax
  int v13; // edx
  __int64 v14; // rax
  int v15; // eax
  int v16; // eax
  __int64 v17; // r8
  unsigned int v18; // edx
  __int64 v19; // rdi
  int v20; // eax
  int v21; // eax
  int v22; // r10d
  __int64 *v23; // r9
  __int64 *v24; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
    v24 = 0;
LABEL_19:
    sub_261D190(a1, 2 * v4);
    v15 = *(_DWORD *)(a1 + 24);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 8);
      v18 = v16 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v6 = (__int64 *)(v17 + 40LL * v18);
      v19 = *v6;
      if ( *v6 == *a2 )
      {
LABEL_21:
        v20 = *(_DWORD *)(a1 + 16);
        v24 = v6;
        v13 = v20 + 1;
      }
      else
      {
        v22 = 1;
        v23 = 0;
        while ( v19 != -4096 )
        {
          if ( !v23 && v19 == -8192 )
            v23 = v6;
          v18 = v16 & (v22 + v18);
          v6 = (__int64 *)(v17 + 40LL * v18);
          v19 = *v6;
          if ( *a2 == *v6 )
            goto LABEL_21;
          ++v22;
        }
        if ( !v23 )
          v23 = v6;
        v13 = *(_DWORD *)(a1 + 16) + 1;
        v24 = v23;
        v6 = v23;
      }
    }
    else
    {
      v21 = *(_DWORD *)(a1 + 16);
      v24 = 0;
      v6 = 0;
      v13 = v21 + 1;
    }
    goto LABEL_15;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 0;
  v7 = 1;
  v8 = (v4 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v9 = (__int64 *)(v5 + 40LL * v8);
  v10 = *v9;
  if ( *a2 == *v9 )
    return v9 + 1;
  while ( v10 != -4096 )
  {
    if ( !v6 && v10 == -8192 )
      v6 = v9;
    v8 = (v4 - 1) & (v7 + v8);
    v9 = (__int64 *)(v5 + 40LL * v8);
    v10 = *v9;
    if ( *a2 == *v9 )
      return v9 + 1;
    ++v7;
  }
  if ( !v6 )
    v6 = v9;
  v12 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v13 = v12 + 1;
  v24 = v6;
  if ( 4 * (v12 + 1) >= 3 * v4 )
    goto LABEL_19;
  if ( v4 - *(_DWORD *)(a1 + 20) - v13 <= v4 >> 3 )
  {
    sub_261D190(a1, v4);
    sub_2618CC0(a1, a2, &v24);
    v6 = v24;
    v13 = *(_DWORD *)(a1 + 16) + 1;
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v13;
  if ( *v6 != -4096 )
    --*(_DWORD *)(a1 + 20);
  v14 = *a2;
  *(_OWORD *)(v6 + 1) = 0;
  *v6 = v14;
  *(_OWORD *)(v6 + 3) = 0;
  return v6 + 1;
}
