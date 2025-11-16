// Function: sub_1BACEB0
// Address: 0x1baceb0
//
__int64 __fastcall sub_1BACEB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v5; // rax
  __int64 v7; // rdi
  unsigned int v8; // esi
  __int64 *v9; // rdx
  __int64 v10; // r9
  __int64 v11; // r13
  __int64 *v12; // rax
  __int64 *v13; // rbx
  int v14; // eax
  _BYTE *v15; // rsi
  unsigned int v16; // esi
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rcx
  unsigned int v20; // r8d
  __int64 result; // rax
  __int64 v22; // r10
  __int64 v23; // r12
  __int64 v24; // rdi
  _QWORD *v25; // r9
  int v26; // ecx
  int v27; // edx
  __int64 *v28; // rax
  int v29; // r14d
  int v30; // eax
  int v31; // r10d
  __int64 v32; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v33[5]; // [rsp+18h] [rbp-28h] BYREF

  v3 = a2;
  v5 = *(unsigned int *)(a1 + 48);
  v32 = a2;
  if ( !(_DWORD)v5 )
    goto LABEL_26;
  v7 = *(_QWORD *)(a1 + 32);
  v8 = (v5 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v9 = (__int64 *)(v7 + 16LL * v8);
  v10 = *v9;
  if ( a3 != *v9 )
  {
    v27 = 1;
    while ( v10 != -8 )
    {
      v31 = v27 + 1;
      v8 = (v5 - 1) & (v27 + v8);
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( a3 == *v9 )
        goto LABEL_3;
      v27 = v31;
    }
    v3 = v32;
    goto LABEL_26;
  }
LABEL_3:
  v3 = v32;
  if ( v9 == (__int64 *)(v7 + 16 * v5) )
  {
LABEL_26:
    *(_BYTE *)(a1 + 72) = 0;
    v28 = (__int64 *)sub_22077B0(56);
    v13 = v28;
    if ( !v28 )
    {
      v33[0] = 0;
      BUG();
    }
    *v28 = v3;
    v11 = 0;
    v14 = 0;
    v13[1] = 0;
    goto LABEL_7;
  }
  v11 = v9[1];
  *(_BYTE *)(a1 + 72) = 0;
  v12 = (__int64 *)sub_22077B0(56);
  v13 = v12;
  if ( !v12 )
    goto LABEL_8;
  *v12 = v3;
  v12[1] = v11;
  if ( v11 )
    v14 = *(_DWORD *)(v11 + 16) + 1;
  else
    v14 = 0;
LABEL_7:
  *((_DWORD *)v13 + 4) = v14;
  v13[3] = 0;
  v13[4] = 0;
  v13[5] = 0;
  v13[6] = -1;
LABEL_8:
  v33[0] = v13;
  v15 = *(_BYTE **)(v11 + 32);
  if ( v15 == *(_BYTE **)(v11 + 40) )
  {
    sub_15CE310(v11 + 24, v15, v33);
    v16 = *(_DWORD *)(a1 + 48);
    v17 = a1 + 24;
    if ( v16 )
      goto LABEL_12;
LABEL_19:
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_20;
  }
  if ( v15 )
  {
    *(_QWORD *)v15 = v13;
    v15 = *(_BYTE **)(v11 + 32);
  }
  *(_QWORD *)(v11 + 32) = v15 + 8;
  v16 = *(_DWORD *)(a1 + 48);
  v17 = a1 + 24;
  if ( !v16 )
    goto LABEL_19;
LABEL_12:
  v18 = v32;
  v19 = *(_QWORD *)(a1 + 32);
  v20 = (v16 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
  result = v19 + 16LL * v20;
  v22 = *(_QWORD *)result;
  if ( v32 == *(_QWORD *)result )
  {
LABEL_13:
    v23 = *(_QWORD *)(result + 8);
    *(_QWORD *)(result + 8) = v13;
    if ( v23 )
    {
      v24 = *(_QWORD *)(v23 + 24);
      if ( v24 )
        j_j___libc_free_0(v24, *(_QWORD *)(v23 + 40) - v24);
      return j_j___libc_free_0(v23, 56);
    }
    return result;
  }
  v29 = 1;
  v25 = 0;
  while ( v22 != -8 )
  {
    if ( !v25 && v22 == -16 )
      v25 = (_QWORD *)result;
    v20 = (v16 - 1) & (v20 + v29);
    result = v19 + 16LL * v20;
    v22 = *(_QWORD *)result;
    if ( v32 == *(_QWORD *)result )
      goto LABEL_13;
    ++v29;
  }
  if ( !v25 )
    v25 = (_QWORD *)result;
  v30 = *(_DWORD *)(a1 + 40);
  ++*(_QWORD *)(a1 + 24);
  v26 = v30 + 1;
  if ( 4 * (v30 + 1) < 3 * v16 )
  {
    result = v16 - *(_DWORD *)(a1 + 44) - v26;
    if ( (unsigned int)result > v16 >> 3 )
      goto LABEL_34;
    goto LABEL_21;
  }
LABEL_20:
  v16 *= 2;
LABEL_21:
  sub_15CFCF0(v17, v16);
  sub_15CE550(v17, &v32, v33);
  result = *(unsigned int *)(a1 + 40);
  v25 = (_QWORD *)v33[0];
  v18 = v32;
  v26 = result + 1;
LABEL_34:
  *(_DWORD *)(a1 + 40) = v26;
  if ( *v25 != -8 )
    --*(_DWORD *)(a1 + 44);
  *v25 = v18;
  v25[1] = v13;
  return result;
}
