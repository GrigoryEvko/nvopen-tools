// Function: sub_2625AD0
// Address: 0x2625ad0
//
__int64 __fastcall sub_2625AD0(__int64 a1, _BYTE *a2, size_t a3)
{
  __int64 v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rax
  unsigned int v12; // edx
  __int64 *v13; // r15
  __int64 v14; // rdi
  __int64 v15; // r12
  __int64 v16; // r13
  size_t v17; // r15
  size_t v18; // rdx
  int v19; // eax
  __int64 v20; // r12
  size_t v21; // r15
  size_t v22; // rdx
  int v23; // eax
  __int64 v24; // rbx
  int v26; // r8d
  _QWORD *v27; // [rsp+8h] [rbp-38h]

  if ( a3 )
  {
    if ( *a2 == 1 )
    {
      v6 = a3 - 1;
      v7 = (__int64)(a2 + 1);
    }
    else
    {
      v6 = a3;
      v7 = (__int64)a2;
    }
  }
  else
  {
    v6 = 0;
    v7 = (__int64)a2;
  }
  v8 = sub_B2F650(v7, v6);
  v9 = *(_QWORD *)(a1 + 8);
  v10 = v8;
  v11 = *(unsigned int *)(a1 + 24);
  if ( !(_DWORD)v11 )
    return 0;
  v12 = (v11 - 1) & (((0xBF58476D1CE4E5B9LL * v10) >> 31) ^ (484763065 * v10));
  v13 = (__int64 *)(v9 + 56LL * v12);
  v14 = *v13;
  if ( v10 != *v13 )
  {
    v26 = 1;
    while ( v14 != -1 )
    {
      v12 = (v11 - 1) & (v26 + v12);
      v13 = (__int64 *)(v9 + 56LL * v12);
      v14 = *v13;
      if ( v10 == *v13 )
        goto LABEL_6;
      ++v26;
    }
    return 0;
  }
LABEL_6:
  if ( v13 == (__int64 *)(v9 + 56 * v11) )
    return 0;
  v15 = v13[3];
  v27 = v13 + 2;
  if ( !v15 )
    return 0;
  v16 = (__int64)(v13 + 2);
  do
  {
    while ( 1 )
    {
      v17 = *(_QWORD *)(v15 + 40);
      v18 = a3;
      if ( v17 <= a3 )
        v18 = *(_QWORD *)(v15 + 40);
      if ( v18 )
      {
        v19 = memcmp(*(const void **)(v15 + 32), a2, v18);
        if ( v19 )
          break;
      }
      if ( v17 == a3 || v17 >= a3 )
      {
        v16 = v15;
        v15 = *(_QWORD *)(v15 + 16);
        goto LABEL_17;
      }
LABEL_9:
      v15 = *(_QWORD *)(v15 + 24);
      if ( !v15 )
        goto LABEL_18;
    }
    if ( v19 < 0 )
      goto LABEL_9;
    v16 = v15;
    v15 = *(_QWORD *)(v15 + 16);
LABEL_17:
    ;
  }
  while ( v15 );
LABEL_18:
  if ( v27 == (_QWORD *)v16 )
    return 0;
  v20 = v16;
  while ( 1 )
  {
    v21 = *(_QWORD *)(v20 + 40);
    v22 = a3;
    if ( v21 <= a3 )
      v22 = *(_QWORD *)(v20 + 40);
    if ( !v22 )
      break;
    v23 = memcmp(a2, *(const void **)(v20 + 32), v22);
    if ( !v23 )
      break;
    if ( v23 < 0 )
      goto LABEL_26;
LABEL_36:
    v20 = sub_220EF30(v20);
    if ( (_QWORD *)v20 == v27 )
      goto LABEL_27;
  }
  if ( v21 == a3 || v21 <= a3 )
    goto LABEL_36;
LABEL_26:
  if ( v20 == v16 )
    return 0;
LABEL_27:
  v24 = 0;
  do
  {
    ++v24;
    v16 = sub_220EF30(v16);
  }
  while ( v16 != v20 );
  return v24;
}
