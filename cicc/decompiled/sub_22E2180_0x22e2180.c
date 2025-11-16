// Function: sub_22E2180
// Address: 0x22e2180
//
__int64 __fastcall sub_22E2180(__int64 a1, __int64 *a2, _QWORD *a3)
{
  _QWORD *v5; // r12
  __int64 i; // rbx
  unsigned int v7; // esi
  __int64 v8; // r8
  int v9; // r11d
  _QWORD *v10; // rdx
  unsigned int v11; // edi
  _QWORD *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r15
  unsigned __int64 *v15; // rax
  __int64 *v16; // rbx
  __int64 result; // rax
  __int64 *j; // r12
  __int64 v19; // rsi
  int v20; // eax
  int v21; // ecx
  int v22; // eax
  int v23; // r8d
  __int64 v24; // rdi
  unsigned int v25; // eax
  __int64 v26; // rsi
  int v27; // r10d
  _QWORD *v28; // r9
  int v29; // eax
  int v30; // esi
  __int64 v31; // rdi
  _QWORD *v32; // r8
  unsigned int v33; // r15d
  int v34; // r9d
  __int64 v35; // rax

  v5 = a3;
  for ( i = *a2; v5[4] == i; v5 = (_QWORD *)v5[1] )
    ;
  v7 = *(_DWORD *)(a1 + 64);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 40);
    goto LABEL_23;
  }
  v8 = *(_QWORD *)(a1 + 48);
  v9 = 1;
  v10 = 0;
  v11 = (v7 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
  v12 = (_QWORD *)(v8 + 16LL * v11);
  v13 = *v12;
  if ( i == *v12 )
  {
LABEL_5:
    v14 = v12[1];
    v15 = (unsigned __int64 *)sub_22DBE70(a1, v14);
    sub_22E10D0(v5, v15, 0);
    goto LABEL_6;
  }
  while ( v13 != -4096 )
  {
    if ( v13 == -8192 && !v10 )
      v10 = v12;
    v11 = (v7 - 1) & (v9 + v11);
    v12 = (_QWORD *)(v8 + 16LL * v11);
    v13 = *v12;
    if ( i == *v12 )
      goto LABEL_5;
    ++v9;
  }
  if ( !v10 )
    v10 = v12;
  v20 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  v21 = v20 + 1;
  if ( 4 * (v20 + 1) >= 3 * v7 )
  {
LABEL_23:
    sub_22E09A0(a1 + 40, 2 * v7);
    v22 = *(_DWORD *)(a1 + 64);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 48);
      v25 = (v22 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
      v21 = *(_DWORD *)(a1 + 56) + 1;
      v10 = (_QWORD *)(v24 + 16LL * v25);
      v26 = *v10;
      if ( i != *v10 )
      {
        v27 = 1;
        v28 = 0;
        while ( v26 != -4096 )
        {
          if ( v26 == -8192 && !v28 )
            v28 = v10;
          v25 = v23 & (v27 + v25);
          v10 = (_QWORD *)(v24 + 16LL * v25);
          v26 = *v10;
          if ( i == *v10 )
            goto LABEL_19;
          ++v27;
        }
        if ( v28 )
          v10 = v28;
      }
      goto LABEL_19;
    }
    goto LABEL_46;
  }
  if ( v7 - *(_DWORD *)(a1 + 60) - v21 <= v7 >> 3 )
  {
    sub_22E09A0(a1 + 40, v7);
    v29 = *(_DWORD *)(a1 + 64);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 48);
      v32 = 0;
      v33 = (v29 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
      v34 = 1;
      v21 = *(_DWORD *)(a1 + 56) + 1;
      v10 = (_QWORD *)(v31 + 16LL * v33);
      v35 = *v10;
      if ( i != *v10 )
      {
        while ( v35 != -4096 )
        {
          if ( !v32 && v35 == -8192 )
            v32 = v10;
          v33 = v30 & (v34 + v33);
          v10 = (_QWORD *)(v31 + 16LL * v33);
          v35 = *v10;
          if ( i == *v10 )
            goto LABEL_19;
          ++v34;
        }
        if ( v32 )
          v10 = v32;
      }
      goto LABEL_19;
    }
LABEL_46:
    ++*(_DWORD *)(a1 + 56);
    BUG();
  }
LABEL_19:
  *(_DWORD *)(a1 + 56) = v21;
  if ( *v10 != -4096 )
    --*(_DWORD *)(a1 + 60);
  *v10 = i;
  v14 = (__int64)v5;
  v10[1] = v5;
LABEL_6:
  v16 = (__int64 *)a2[3];
  result = *((unsigned int *)a2 + 8);
  for ( j = &v16[result]; j != v16; result = sub_22E2180(a1, v19, v14) )
    v19 = *v16++;
  return result;
}
