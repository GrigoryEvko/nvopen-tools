// Function: sub_1B10DD0
// Address: 0x1b10dd0
//
__int64 __fastcall sub_1B10DD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 *v9; // rdx
  __int64 v10; // r9
  __int64 v11; // r13
  _QWORD *v12; // rax
  _QWORD *v13; // rbx
  int v14; // eax
  _BYTE *v15; // rsi
  __int64 v16; // rdi
  unsigned int v17; // esi
  __int64 v18; // r9
  unsigned int v19; // r8d
  __int64 result; // rax
  __int64 v21; // rcx
  __int64 v22; // r12
  __int64 v23; // rdi
  int v24; // r11d
  _QWORD *v25; // rdx
  int v26; // eax
  int v27; // ecx
  int v28; // eax
  int v29; // esi
  __int64 v30; // r8
  __int64 v31; // rdi
  int v32; // r10d
  _QWORD *v33; // r9
  int v34; // edx
  _QWORD *v35; // rax
  int v36; // r10d
  int v37; // eax
  __int64 v38; // rdi
  _QWORD *v39; // r8
  unsigned int v40; // r13d
  int v41; // r9d
  __int64 v42; // rsi
  _QWORD *v43; // [rsp+8h] [rbp-38h] BYREF

  v5 = *(unsigned int *)(a1 + 48);
  if ( !(_DWORD)v5 )
    goto LABEL_39;
  v7 = *(_QWORD *)(a1 + 32);
  v8 = (v5 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v9 = (__int64 *)(v7 + 16LL * v8);
  v10 = *v9;
  if ( a3 != *v9 )
  {
    v34 = 1;
    while ( v10 != -8 )
    {
      v36 = v34 + 1;
      v8 = (v5 - 1) & (v34 + v8);
      v9 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v9;
      if ( a3 == *v9 )
        goto LABEL_3;
      v34 = v36;
    }
    goto LABEL_39;
  }
LABEL_3:
  if ( v9 == (__int64 *)(v7 + 16 * v5) )
  {
LABEL_39:
    *(_BYTE *)(a1 + 72) = 0;
    v35 = (_QWORD *)sub_22077B0(56);
    v13 = v35;
    if ( !v35 )
    {
      v43 = 0;
      BUG();
    }
    *v35 = a2;
    v11 = 0;
    v14 = 0;
    v13[1] = 0;
    goto LABEL_7;
  }
  v11 = v9[1];
  *(_BYTE *)(a1 + 72) = 0;
  v12 = (_QWORD *)sub_22077B0(56);
  v13 = v12;
  if ( !v12 )
    goto LABEL_8;
  *v12 = a2;
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
  v43 = v13;
  v15 = *(_BYTE **)(v11 + 32);
  if ( v15 == *(_BYTE **)(v11 + 40) )
  {
    sub_15CE310(v11 + 24, v15, &v43);
    v17 = *(_DWORD *)(a1 + 48);
    v16 = a1 + 24;
    if ( v17 )
      goto LABEL_12;
LABEL_28:
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_29;
  }
  if ( v15 )
  {
    *(_QWORD *)v15 = v13;
    v15 = *(_BYTE **)(v11 + 32);
  }
  v16 = a1 + 24;
  *(_QWORD *)(v11 + 32) = v15 + 8;
  v17 = *(_DWORD *)(a1 + 48);
  if ( !v17 )
    goto LABEL_28;
LABEL_12:
  v18 = *(_QWORD *)(a1 + 32);
  v19 = (v17 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v18 + 16LL * v19;
  v21 = *(_QWORD *)result;
  if ( *(_QWORD *)result == a2 )
  {
LABEL_13:
    v22 = *(_QWORD *)(result + 8);
    *(_QWORD *)(result + 8) = v13;
    if ( v22 )
    {
      v23 = *(_QWORD *)(v22 + 24);
      if ( v23 )
        j_j___libc_free_0(v23, *(_QWORD *)(v22 + 40) - v23);
      return j_j___libc_free_0(v22, 56);
    }
    return result;
  }
  v24 = 1;
  v25 = 0;
  while ( v21 != -8 )
  {
    if ( !v25 && v21 == -16 )
      v25 = (_QWORD *)result;
    v19 = (v17 - 1) & (v24 + v19);
    result = v18 + 16LL * v19;
    v21 = *(_QWORD *)result;
    if ( *(_QWORD *)result == a2 )
      goto LABEL_13;
    ++v24;
  }
  if ( !v25 )
    v25 = (_QWORD *)result;
  v26 = *(_DWORD *)(a1 + 40);
  ++*(_QWORD *)(a1 + 24);
  v27 = v26 + 1;
  if ( 4 * (v26 + 1) >= 3 * v17 )
  {
LABEL_29:
    sub_15CFCF0(v16, 2 * v17);
    v28 = *(_DWORD *)(a1 + 48);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 32);
      result = (v28 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v27 = *(_DWORD *)(a1 + 40) + 1;
      v25 = (_QWORD *)(v30 + 16 * result);
      v31 = *v25;
      if ( *v25 != a2 )
      {
        v32 = 1;
        v33 = 0;
        while ( v31 != -8 )
        {
          if ( !v33 && v31 == -16 )
            v33 = v25;
          result = v29 & (unsigned int)(v32 + result);
          v25 = (_QWORD *)(v30 + 16LL * (unsigned int)result);
          v31 = *v25;
          if ( *v25 == a2 )
            goto LABEL_23;
          ++v32;
        }
        if ( v33 )
          v25 = v33;
      }
      goto LABEL_23;
    }
    goto LABEL_65;
  }
  result = v17 - *(_DWORD *)(a1 + 44) - v27;
  if ( (unsigned int)result <= v17 >> 3 )
  {
    sub_15CFCF0(v16, v17);
    v37 = *(_DWORD *)(a1 + 48);
    if ( v37 )
    {
      result = (unsigned int)(v37 - 1);
      v38 = *(_QWORD *)(a1 + 32);
      v39 = 0;
      v40 = result & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v41 = 1;
      v27 = *(_DWORD *)(a1 + 40) + 1;
      v25 = (_QWORD *)(v38 + 16LL * v40);
      v42 = *v25;
      if ( *v25 != a2 )
      {
        while ( v42 != -8 )
        {
          if ( !v39 && v42 == -16 )
            v39 = v25;
          v40 = result & (v41 + v40);
          v25 = (_QWORD *)(v38 + 16LL * v40);
          v42 = *v25;
          if ( *v25 == a2 )
            goto LABEL_23;
          ++v41;
        }
        if ( v39 )
          v25 = v39;
      }
      goto LABEL_23;
    }
LABEL_65:
    ++*(_DWORD *)(a1 + 40);
    BUG();
  }
LABEL_23:
  *(_DWORD *)(a1 + 40) = v27;
  if ( *v25 != -8 )
    --*(_DWORD *)(a1 + 44);
  *v25 = a2;
  v25[1] = v13;
  return result;
}
