// Function: sub_26CF350
// Address: 0x26cf350
//
void __fastcall sub_26CF350(__int64 a1, __int64 a2, _QWORD *a3, unsigned __int64 a4)
{
  size_t v5; // r13
  unsigned __int64 v8; // rdi
  _QWORD *v9; // r8
  _QWORD *v10; // rax
  _QWORD *v11; // rsi
  __int64 v12; // rdi
  _QWORD *v13; // r13
  size_t v14; // r15
  unsigned __int64 v15; // rdi
  _QWORD *v16; // r9
  __int64 v17; // r8
  _QWORD *v18; // rax
  _QWORD *v19; // rsi
  __int64 v20; // rdi
  unsigned int v21; // esi
  __int64 v22; // rdi
  unsigned int v23; // eax
  size_t *v24; // rcx
  size_t v25; // rdx
  int v26; // r10d
  size_t *v27; // r9
  int v28; // eax
  int v29; // edx
  __int64 i; // r13
  __int64 j; // r15
  int *v33; // [rsp+18h] [rbp-F8h]
  __int64 v34; // [rsp+20h] [rbp-F0h]
  int *v35; // [rsp+28h] [rbp-E8h]
  int *v36; // [rsp+28h] [rbp-E8h]
  __int64 v37[2]; // [rsp+30h] [rbp-E0h] BYREF
  int v38[52]; // [rsp+40h] [rbp-D0h] BYREF

  if ( *(_QWORD *)(a1 + 56) <= a4 )
    return;
  v5 = *(_QWORD *)(a1 + 24);
  v35 = *(int **)(a1 + 16);
  if ( v35 )
  {
    sub_C7D030(v38);
    sub_C7D280(v38, v35, v5);
    sub_C7D290(v38, v37);
    v5 = v37[0];
  }
  v8 = a3[1];
  v9 = *(_QWORD **)(*a3 + 8 * (v5 % v8));
  if ( !v9 )
    goto LABEL_46;
  v10 = (_QWORD *)*v9;
  if ( v5 != *(_QWORD *)(*v9 + 8LL) )
  {
    do
    {
      v11 = (_QWORD *)*v10;
      if ( !*v10 )
        goto LABEL_46;
      v9 = v10;
      if ( v5 % v8 != v11[1] % v8 )
        goto LABEL_46;
      v10 = (_QWORD *)*v10;
    }
    while ( v5 != v11[1] );
  }
  if ( !*v9 || (v12 = *(_QWORD *)(*v9 + 16LL)) == 0 || sub_B2FC80(v12) )
  {
LABEL_46:
    v37[0] = sub_26BA4C0(*(int **)(a1 + 16), *(_QWORD *)(a1 + 24));
    sub_D7AC80((__int64)v38, a2, v37);
  }
  v34 = *(_QWORD *)(a1 + 96);
  if ( v34 != a1 + 80 )
  {
    while ( 1 )
    {
      v13 = *(_QWORD **)(v34 + 64);
      if ( v13 )
        break;
LABEL_40:
      v34 = sub_220EF30(v34);
      if ( a1 + 80 == v34 )
        goto LABEL_41;
    }
    while ( 1 )
    {
      while ( v13[3] <= a4 )
      {
LABEL_16:
        v13 = (_QWORD *)*v13;
        if ( !v13 )
          goto LABEL_40;
      }
      v14 = v13[2];
      v36 = (int *)v13[1];
      if ( !v36 )
        break;
      sub_C7D030(v38);
      sub_C7D280(v38, v36, v14);
      sub_C7D290(v38, v37);
      v14 = v37[0];
      v15 = a3[1];
      v16 = *(_QWORD **)(*a3 + 8 * (v37[0] % v15));
      v17 = v37[0] % v15;
      if ( v16 )
        goto LABEL_20;
LABEL_27:
      v14 = v13[2];
      if ( v13[1] )
      {
        v33 = (int *)v13[1];
        sub_C7D030(v38);
        sub_C7D280(v38, v33, v14);
        sub_C7D290(v38, v37);
        v14 = v37[0];
      }
      v21 = *(_DWORD *)(a2 + 24);
      v37[0] = v14;
      if ( !v21 )
      {
LABEL_49:
        ++*(_QWORD *)a2;
        *(_QWORD *)v38 = 0;
LABEL_50:
        v21 *= 2;
LABEL_51:
        sub_A32210(a2, v21);
        sub_A27FA0(a2, v37, v38);
        v14 = v37[0];
        v27 = *(size_t **)v38;
        v29 = *(_DWORD *)(a2 + 16) + 1;
        goto LABEL_37;
      }
LABEL_30:
      v22 = *(_QWORD *)(a2 + 8);
      v23 = (v21 - 1) & (((0xBF58476D1CE4E5B9LL * v14) >> 31) ^ (484763065 * v14));
      v24 = (size_t *)(v22 + 8LL * v23);
      v25 = *v24;
      if ( *v24 == v14 )
        goto LABEL_16;
      v26 = 1;
      v27 = 0;
      while ( v25 != -1 )
      {
        if ( v25 != -2 || v27 )
          v24 = v27;
        v23 = (v21 - 1) & (v26 + v23);
        v25 = *(_QWORD *)(v22 + 8LL * v23);
        if ( v14 == v25 )
          goto LABEL_16;
        ++v26;
        v27 = v24;
        v24 = (size_t *)(v22 + 8LL * v23);
      }
      v28 = *(_DWORD *)(a2 + 16);
      if ( !v27 )
        v27 = v24;
      ++*(_QWORD *)a2;
      v29 = v28 + 1;
      *(_QWORD *)v38 = v27;
      if ( 4 * (v28 + 1) >= 3 * v21 )
        goto LABEL_50;
      if ( v21 - *(_DWORD *)(a2 + 20) - v29 <= v21 >> 3 )
        goto LABEL_51;
LABEL_37:
      *(_DWORD *)(a2 + 16) = v29;
      if ( *v27 != -1 )
        --*(_DWORD *)(a2 + 20);
      *v27 = v14;
      v13 = (_QWORD *)*v13;
      if ( !v13 )
        goto LABEL_40;
    }
    v15 = a3[1];
    v16 = *(_QWORD **)(*a3 + 8 * (v14 % v15));
    v17 = v14 % v15;
    if ( !v16 )
    {
      v21 = *(_DWORD *)(a2 + 24);
      v37[0] = v13[2];
      if ( !v21 )
        goto LABEL_49;
      goto LABEL_30;
    }
LABEL_20:
    v18 = (_QWORD *)*v16;
    if ( *(_QWORD *)(*v16 + 8LL) == v14 )
    {
LABEL_24:
      if ( *v16 )
      {
        v20 = *(_QWORD *)(*v16 + 16LL);
        if ( v20 )
        {
          if ( !sub_B2FC80(v20) )
            goto LABEL_16;
        }
      }
    }
    else
    {
      while ( 1 )
      {
        v19 = (_QWORD *)*v18;
        if ( !*v18 )
          break;
        v16 = v18;
        if ( v19[1] % v15 != v17 )
          break;
        v18 = (_QWORD *)*v18;
        if ( v19[1] == v14 )
          goto LABEL_24;
      }
    }
    goto LABEL_27;
  }
LABEL_41:
  for ( i = *(_QWORD *)(a1 + 144); a1 + 128 != i; i = sub_220EF30(i) )
  {
    for ( j = *(_QWORD *)(i + 64); i + 48 != j; j = sub_220EF30(j) )
      sub_26CF350(j + 48, a2, a3, a4);
  }
}
