// Function: sub_1BA16F0
// Address: 0x1ba16f0
//
__int64 __fastcall sub_1BA16F0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // rax
  __int64 v4; // r12
  unsigned int v6; // esi
  unsigned int v7; // r10d
  __int64 v8; // r9
  unsigned int v9; // edx
  __int64 *v10; // rcx
  __int64 v11; // rdi
  _QWORD *v12; // rax
  __int64 v14; // r8
  __int64 v15; // r13
  int i; // r11d
  unsigned int **v17; // r13
  unsigned int v18; // esi
  __int64 (__fastcall *v19)(__int64, __int64 *, unsigned int); // r14
  __int64 v20; // r9
  unsigned int v21; // ecx
  __int64 *v22; // rdx
  __int64 v23; // r8
  __int64 *v24; // rsi
  __int64 *v25; // rdi
  int v26; // ecx
  int v27; // r11d
  __int64 *v28; // r8
  int v29; // edi
  int v30; // ecx
  int v31; // r11d
  int v32; // ecx
  __int64 *v33; // r13
  int v34; // edi
  __int64 *v35; // r11
  __int64 v36[2]; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v37[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = a2;
  v4 = a3;
  v36[0] = a2;
  v6 = *(_DWORD *)(a1 + 48);
  if ( !v6 )
  {
LABEL_7:
    v17 = *(unsigned int ***)(a1 + 232);
    v18 = *(_DWORD *)(a1 + 216);
    v19 = (__int64 (__fastcall *)(__int64, __int64 *, unsigned int))*((_QWORD *)*v17 + 2);
    if ( v18 )
    {
      v20 = *(_QWORD *)(a1 + 200);
      v21 = (v18 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v22 = (__int64 *)(v20 + 16LL * v21);
      v23 = *v22;
      if ( v3 == *v22 )
      {
        v24 = (__int64 *)v22[1];
LABEL_10:
        if ( v19 == sub_1B9DCA0 )
          return sub_1B9C240(v17[1], v24, v4);
        else
          return v19((__int64)v17, v24, v4);
      }
      v31 = 1;
      v25 = 0;
      while ( v23 != -8 )
      {
        if ( v23 != -16 || v25 )
          v22 = v25;
        v34 = v31 + 1;
        v21 = (v18 - 1) & (v31 + v21);
        v35 = (__int64 *)(v20 + 16LL * v21);
        v23 = *v35;
        if ( v3 == *v35 )
        {
          v24 = (__int64 *)v35[1];
          goto LABEL_10;
        }
        v31 = v34;
        v25 = v22;
        v22 = (__int64 *)(v20 + 16LL * v21);
      }
      v32 = *(_DWORD *)(a1 + 208);
      if ( !v25 )
        v25 = v22;
      ++*(_QWORD *)(a1 + 192);
      v26 = v32 + 1;
      if ( 4 * v26 < 3 * v18 )
      {
        if ( v18 - *(_DWORD *)(a1 + 212) - v26 > v18 >> 3 )
          goto LABEL_16;
LABEL_15:
        sub_1BA1560(a1 + 192, v18);
        sub_1BA0B20(a1 + 192, v36, v37);
        v25 = (__int64 *)v37[0];
        v3 = v36[0];
        v26 = *(_DWORD *)(a1 + 208) + 1;
LABEL_16:
        *(_DWORD *)(a1 + 208) = v26;
        if ( *v25 != -8 )
          --*(_DWORD *)(a1 + 212);
        *v25 = v3;
        v24 = 0;
        v25[1] = 0;
        goto LABEL_10;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 192);
    }
    v18 *= 2;
    goto LABEL_15;
  }
  v7 = v6 - 1;
  v8 = *(_QWORD *)(a1 + 32);
  v9 = (v6 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v10 = (__int64 *)(v8 + 40LL * v9);
  v11 = *v10;
  if ( v3 != *v10 )
  {
    v14 = *v10;
    LODWORD(v15) = (v6 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    for ( i = 1; ; ++i )
    {
      if ( v14 == -8 )
        goto LABEL_7;
      v15 = v7 & ((_DWORD)v15 + i);
      v14 = *(_QWORD *)(v8 + 40 * v15);
      if ( v3 == v14 )
        break;
    }
    v27 = 1;
    v28 = 0;
    while ( v11 != -8 )
    {
      if ( v11 != -16 || v28 )
        v10 = v28;
      v9 = v7 & (v27 + v9);
      v33 = (__int64 *)(v8 + 40LL * v9);
      v11 = *v33;
      if ( v3 == *v33 )
      {
        v12 = (_QWORD *)v33[1];
        return v12[v4];
      }
      ++v27;
      v28 = v10;
      v10 = (__int64 *)(v8 + 40LL * v9);
    }
    v29 = *(_DWORD *)(a1 + 40);
    if ( !v28 )
      v28 = v10;
    ++*(_QWORD *)(a1 + 24);
    v30 = v29 + 1;
    if ( 4 * (v29 + 1) >= 3 * v6 )
    {
      v6 *= 2;
    }
    else if ( v6 - *(_DWORD *)(a1 + 44) - v30 > v6 >> 3 )
    {
LABEL_26:
      *(_DWORD *)(a1 + 40) = v30;
      if ( *v28 != -8 )
        --*(_DWORD *)(a1 + 44);
      *v28 = v3;
      v12 = v28 + 3;
      v28[1] = (__int64)(v28 + 3);
      v28[2] = 0x200000000LL;
      return v12[v4];
    }
    sub_1BA1370(a1 + 24, v6);
    sub_1BA0A70(a1 + 24, v36, v37);
    v28 = (__int64 *)v37[0];
    v3 = v36[0];
    v30 = *(_DWORD *)(a1 + 40) + 1;
    goto LABEL_26;
  }
  v12 = (_QWORD *)v10[1];
  return v12[v4];
}
