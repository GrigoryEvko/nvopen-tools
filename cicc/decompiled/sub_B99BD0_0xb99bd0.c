// Function: sub_B99BD0
// Address: 0xb99bd0
//
__int64 __fastcall sub_B99BD0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r13
  __int64 v5; // rdx
  unsigned int v6; // esi
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 *v9; // r12
  __int64 v10; // r8
  __int64 *v11; // r8
  __int64 v12; // rcx
  char *v13; // rsi
  __int64 v14; // r9
  char *v15; // rdi
  unsigned int v16; // esi
  __int64 v17; // rcx
  int v18; // r9d
  __int64 *v19; // r12
  unsigned int v20; // eax
  __int64 v21; // r14
  __int64 v22; // r8
  int v23; // r10d
  int v24; // eax
  int v25; // ecx
  _QWORD *v26; // rax
  _QWORD *v27; // r12
  __int64 v28; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v29; // [rsp+18h] [rbp-38h] BYREF

  v28 = a2;
  result = sub_BD5C60(a1, a2);
  v4 = *(_QWORD *)result;
  if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
    goto LABEL_27;
  result = sub_B91C10(a1, 38);
  v5 = v28;
  if ( !result )
    goto LABEL_17;
  if ( v28 == result )
    return result;
  v6 = *(_DWORD *)(v4 + 3280);
  v7 = *(_QWORD *)(v4 + 3264);
  if ( v6 )
  {
    v8 = (v6 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v9 = (__int64 *)(v7 + 32LL * v8);
    v10 = *v9;
    if ( result == *v9 )
      goto LABEL_6;
    v23 = 1;
    while ( v10 != -4096 )
    {
      v8 = (v6 - 1) & (v23 + v8);
      v9 = (__int64 *)(v7 + 32LL * v8);
      v10 = *v9;
      if ( result == *v9 )
        goto LABEL_6;
      ++v23;
    }
  }
  v9 = (__int64 *)(v7 + 32LL * v6);
LABEL_6:
  v11 = (__int64 *)v9[1];
  v12 = *((unsigned int *)v9 + 4);
  v13 = (char *)&v11[v12];
  v14 = (8 * v12) >> 3;
  if ( (8 * v12) >> 5 )
  {
    v15 = (char *)v9[1];
    while ( a1 != *(_QWORD *)v15 )
    {
      if ( a1 == *((_QWORD *)v15 + 1) )
      {
        v15 += 8;
        goto LABEL_13;
      }
      if ( a1 == *((_QWORD *)v15 + 2) )
      {
        v15 += 16;
        goto LABEL_13;
      }
      if ( a1 == *((_QWORD *)v15 + 3) )
      {
        v15 += 24;
        goto LABEL_13;
      }
      v15 += 32;
      if ( &v11[4 * ((8 * v12) >> 5)] == (__int64 *)v15 )
      {
        v14 = (v13 - v15) >> 3;
        goto LABEL_34;
      }
    }
    goto LABEL_13;
  }
  v15 = (char *)v9[1];
LABEL_34:
  switch ( v14 )
  {
    case 2LL:
LABEL_40:
      if ( a1 == *(_QWORD *)v15 )
        goto LABEL_13;
      v15 += 8;
LABEL_42:
      if ( a1 != *(_QWORD *)v15 )
        v15 = (char *)&v11[v12];
      goto LABEL_13;
    case 3LL:
      if ( a1 == *(_QWORD *)v15 )
        goto LABEL_13;
      v15 += 8;
      goto LABEL_40;
    case 1LL:
      goto LABEL_42;
  }
  v15 = (char *)&v11[v12];
LABEL_13:
  if ( (_DWORD)v12 != 1 )
  {
    if ( v13 != v15 + 8 )
    {
      memmove(v15, v15 + 8, v13 - (v15 + 8));
      LODWORD(v12) = *((_DWORD *)v9 + 4);
      v5 = v28;
    }
    result = (unsigned int)(v12 - 1);
    *((_DWORD *)v9 + 4) = result;
LABEL_17:
    if ( !v5 )
      return result;
    goto LABEL_18;
  }
  result = (__int64)(v9 + 3);
  if ( v11 != v9 + 3 )
    result = _libc_free(v9[1], v13);
  *v9 = -8192;
  --*(_DWORD *)(v4 + 3272);
  ++*(_DWORD *)(v4 + 3276);
LABEL_27:
  v5 = v28;
  if ( !v28 )
    return result;
LABEL_18:
  v16 = *(_DWORD *)(v4 + 3280);
  if ( !v16 )
  {
    ++*(_QWORD *)(v4 + 3256);
    v29 = 0;
LABEL_63:
    v16 *= 2;
    goto LABEL_64;
  }
  v17 = *(_QWORD *)(v4 + 3264);
  v18 = 1;
  v19 = 0;
  v20 = (v16 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v21 = v17 + 32LL * v20;
  v22 = *(_QWORD *)v21;
  if ( v5 == *(_QWORD *)v21 )
  {
LABEL_20:
    result = *(unsigned int *)(v21 + 16);
    if ( *(unsigned int *)(v21 + 20) < (unsigned __int64)(result + 1) )
    {
      sub_C8D5F0(v21 + 8, v21 + 24, result + 1, 8);
      result = *(unsigned int *)(v21 + 16);
    }
    *(_QWORD *)(*(_QWORD *)(v21 + 8) + 8 * result) = a1;
    ++*(_DWORD *)(v21 + 16);
    return result;
  }
  while ( v22 != -4096 )
  {
    if ( v22 == -8192 && !v19 )
      v19 = (__int64 *)v21;
    v20 = (v16 - 1) & (v18 + v20);
    v21 = v17 + 32LL * v20;
    v22 = *(_QWORD *)v21;
    if ( *(_QWORD *)v21 == v5 )
      goto LABEL_20;
    ++v18;
  }
  v24 = *(_DWORD *)(v4 + 3272);
  if ( !v19 )
    v19 = (__int64 *)v21;
  ++*(_QWORD *)(v4 + 3256);
  v25 = v24 + 1;
  v29 = v19;
  if ( 4 * (v24 + 1) >= 3 * v16 )
    goto LABEL_63;
  if ( v16 - *(_DWORD *)(v4 + 3276) - v25 <= v16 >> 3 )
  {
LABEL_64:
    sub_B998D0(v4 + 3256, v16);
    sub_B92940(v4 + 3256, &v28, &v29);
    v5 = v28;
    v19 = v29;
    v25 = *(_DWORD *)(v4 + 3272) + 1;
  }
  *(_DWORD *)(v4 + 3272) = v25;
  if ( *v19 != -4096 )
    --*(_DWORD *)(v4 + 3276);
  v26 = v19 + 3;
  *v19 = v5;
  v27 = v19 + 1;
  *v27 = v26;
  v27[1] = 0x100000000LL;
  *(_QWORD *)*v27 = a1;
  ++*((_DWORD *)v27 + 2);
  return 0;
}
