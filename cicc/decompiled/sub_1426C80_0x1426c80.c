// Function: sub_1426C80
// Address: 0x1426c80
//
__int64 __fastcall sub_1426C80(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  int v5; // r15d
  __int64 v6; // r14
  __int64 v7; // r12
  __int64 v8; // rax
  unsigned int v9; // esi
  __int64 v10; // rdi
  __int64 v11; // r8
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // rdx
  int v16; // r11d
  __int64 *v17; // r10
  int v18; // ecx
  int v19; // ecx
  int v20; // eax
  int v21; // esi
  __int64 v22; // r8
  unsigned int v23; // edx
  __int64 v24; // rdi
  int v25; // r10d
  __int64 *v26; // r9
  int v27; // eax
  int v28; // edx
  __int64 v29; // rdi
  int v30; // r9d
  unsigned int v31; // r14d
  __int64 *v32; // r8
  __int64 v33; // rsi

  v4 = sub_157E9C0(a2);
  v5 = *(_DWORD *)(a1 + 336);
  v6 = v4;
  *(_DWORD *)(a1 + 336) = v5 + 1;
  v7 = sub_1648B60(80);
  if ( v7 )
  {
    v8 = sub_1643270(v6);
    sub_1648CB0(v7, v8, 23);
    *(_QWORD *)(v7 + 64) = a2;
    *(_DWORD *)(v7 + 72) = v5;
    *(_DWORD *)(v7 + 20) &= 0xF0000000;
    *(_QWORD *)(v7 + 24) = sub_141FFA0;
    *(_QWORD *)(v7 + 32) = 0;
    *(_QWORD *)(v7 + 40) = 0;
    *(_QWORD *)(v7 + 48) = 0;
    *(_QWORD *)(v7 + 56) = 0;
    *(_DWORD *)(v7 + 76) = 0;
    sub_1648880(v7, 0, 1);
  }
  sub_14264F0(a1, v7, a2, 0);
  v9 = *(_DWORD *)(a1 + 48);
  v10 = a1 + 24;
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_16;
  }
  v11 = *(_QWORD *)(a1 + 32);
  v12 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (__int64 *)(v11 + 16LL * v12);
  v14 = *v13;
  if ( a2 == *v13 )
    goto LABEL_5;
  v16 = 1;
  v17 = 0;
  while ( v14 != -8 )
  {
    if ( v14 == -16 && !v17 )
      v17 = v13;
    v12 = (v9 - 1) & (v16 + v12);
    v13 = (__int64 *)(v11 + 16LL * v12);
    v14 = *v13;
    if ( a2 == *v13 )
      goto LABEL_5;
    ++v16;
  }
  v18 = *(_DWORD *)(a1 + 40);
  if ( v17 )
    v13 = v17;
  ++*(_QWORD *)(a1 + 24);
  v19 = v18 + 1;
  if ( 4 * v19 >= 3 * v9 )
  {
LABEL_16:
    sub_14267C0(v10, 2 * v9);
    v20 = *(_DWORD *)(a1 + 48);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 32);
      v23 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = *(_DWORD *)(a1 + 40) + 1;
      v13 = (__int64 *)(v22 + 16LL * v23);
      v24 = *v13;
      if ( a2 != *v13 )
      {
        v25 = 1;
        v26 = 0;
        while ( v24 != -8 )
        {
          if ( !v26 && v24 == -16 )
            v26 = v13;
          v23 = v21 & (v25 + v23);
          v13 = (__int64 *)(v22 + 16LL * v23);
          v24 = *v13;
          if ( a2 == *v13 )
            goto LABEL_12;
          ++v25;
        }
        if ( v26 )
          v13 = v26;
      }
      goto LABEL_12;
    }
    goto LABEL_44;
  }
  if ( v9 - *(_DWORD *)(a1 + 44) - v19 <= v9 >> 3 )
  {
    sub_14267C0(v10, v9);
    v27 = *(_DWORD *)(a1 + 48);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a1 + 32);
      v30 = 1;
      v31 = (v27 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v32 = 0;
      v19 = *(_DWORD *)(a1 + 40) + 1;
      v13 = (__int64 *)(v29 + 16LL * v31);
      v33 = *v13;
      if ( a2 != *v13 )
      {
        while ( v33 != -8 )
        {
          if ( v33 == -16 && !v32 )
            v32 = v13;
          v31 = v28 & (v30 + v31);
          v13 = (__int64 *)(v29 + 16LL * v31);
          v33 = *v13;
          if ( a2 == *v13 )
            goto LABEL_12;
          ++v30;
        }
        if ( v32 )
          v13 = v32;
      }
      goto LABEL_12;
    }
LABEL_44:
    ++*(_DWORD *)(a1 + 40);
    BUG();
  }
LABEL_12:
  *(_DWORD *)(a1 + 40) = v19;
  if ( *v13 != -8 )
    --*(_DWORD *)(a1 + 44);
  *v13 = a2;
  v13[1] = 0;
LABEL_5:
  v13[1] = v7;
  return v7;
}
