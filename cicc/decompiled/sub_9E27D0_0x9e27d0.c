// Function: sub_9E27D0
// Address: 0x9e27d0
//
__int64 __fastcall sub_9E27D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  unsigned int v5; // esi
  __int64 v6; // r15
  __int64 v7; // rcx
  __int64 v8; // rdi
  int v9; // r11d
  __int64 *v10; // r10
  unsigned __int64 v11; // r14
  unsigned int v12; // edx
  __int64 *v13; // r12
  __int64 v14; // rax
  int v16; // eax
  int v17; // edx
  _BYTE *v18; // rsi
  int v19; // edx
  int v20; // esi
  __int64 v21; // rdi
  unsigned int v22; // eax
  __int64 v23; // r8
  int v24; // r10d
  __int64 *v25; // r9
  int v26; // eax
  int v27; // eax
  __int64 v28; // rdi
  __int64 *v29; // r8
  unsigned int v30; // r14d
  int v31; // r9d
  __int64 v32; // rsi
  __int64 v33; // [rsp+0h] [rbp-40h]
  __int64 v34; // [rsp+0h] [rbp-40h]
  _QWORD v35[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = a1 + 552;
  v35[0] = a2;
  v5 = *(_DWORD *)(a1 + 576);
  v6 = *(_QWORD *)(a1 + 536);
  v7 = *(_QWORD *)(a1 + 528);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 552);
    goto LABEL_22;
  }
  v8 = *(_QWORD *)(a1 + 560);
  v9 = 1;
  v10 = 0;
  v11 = ((0xBF58476D1CE4E5B9LL * a2) >> 31) ^ (0xBF58476D1CE4E5B9LL * a2);
  v12 = v11 & (v5 - 1);
  v13 = (__int64 *)(v8 + 16LL * v12);
  v14 = *v13;
  if ( a2 == *v13 )
    return *((unsigned int *)v13 + 2);
  while ( v14 != -1 )
  {
    if ( !v10 && v14 == -2 )
      v10 = v13;
    v12 = (v5 - 1) & (v9 + v12);
    v13 = (__int64 *)(v8 + 16LL * v12);
    v14 = *v13;
    if ( a2 == *v13 )
      return *((unsigned int *)v13 + 2);
    ++v9;
  }
  v16 = *(_DWORD *)(a1 + 568);
  if ( v10 )
    v13 = v10;
  ++*(_QWORD *)(a1 + 552);
  v17 = v16 + 1;
  if ( 4 * (v16 + 1) >= 3 * v5 )
  {
LABEL_22:
    v33 = v7;
    sub_9E25D0(v2, 2 * v5);
    v19 = *(_DWORD *)(a1 + 576);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 560);
      v17 = *(_DWORD *)(a1 + 568) + 1;
      v7 = v33;
      v22 = v20 & (((0xBF58476D1CE4E5B9LL * a2) >> 31) ^ (484763065 * a2));
      v13 = (__int64 *)(v21 + 16LL * v22);
      v23 = *v13;
      if ( a2 != *v13 )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -1 )
        {
          if ( !v25 && v23 == -2 )
            v25 = v13;
          v22 = v20 & (v24 + v22);
          v13 = (__int64 *)(v21 + 16LL * v22);
          v23 = *v13;
          if ( a2 == *v13 )
            goto LABEL_14;
          ++v24;
        }
        if ( v25 )
          v13 = v25;
      }
      goto LABEL_14;
    }
    goto LABEL_45;
  }
  if ( v5 - *(_DWORD *)(a1 + 572) - v17 <= v5 >> 3 )
  {
    v34 = v7;
    sub_9E25D0(v2, v5);
    v26 = *(_DWORD *)(a1 + 576);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a1 + 560);
      v29 = 0;
      v30 = v27 & v11;
      v31 = 1;
      v17 = *(_DWORD *)(a1 + 568) + 1;
      v7 = v34;
      v13 = (__int64 *)(v28 + 16LL * v30);
      v32 = *v13;
      if ( a2 != *v13 )
      {
        while ( v32 != -1 )
        {
          if ( !v29 && v32 == -2 )
            v29 = v13;
          v30 = v27 & (v31 + v30);
          v13 = (__int64 *)(v28 + 16LL * v30);
          v32 = *v13;
          if ( a2 == *v13 )
            goto LABEL_14;
          ++v31;
        }
        if ( v29 )
          v13 = v29;
      }
      goto LABEL_14;
    }
LABEL_45:
    ++*(_DWORD *)(a1 + 568);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 568) = v17;
  if ( *v13 != -1 )
    --*(_DWORD *)(a1 + 572);
  *v13 = a2;
  *((_DWORD *)v13 + 2) = (v6 - v7) >> 3;
  v18 = *(_BYTE **)(a1 + 536);
  if ( v18 == *(_BYTE **)(a1 + 544) )
  {
    sub_9CA200(a1 + 528, v18, v35);
  }
  else
  {
    if ( v18 )
    {
      *(_QWORD *)v18 = v35[0];
      v18 = *(_BYTE **)(a1 + 536);
    }
    *(_QWORD *)(a1 + 536) = v18 + 8;
  }
  return *((unsigned int *)v13 + 2);
}
