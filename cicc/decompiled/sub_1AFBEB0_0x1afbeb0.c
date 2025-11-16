// Function: sub_1AFBEB0
// Address: 0x1afbeb0
//
__int64 __fastcall sub_1AFBEB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 i; // r11
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r10
  __int64 v9; // rsi
  __int64 v10; // r12
  int v11; // r13d
  unsigned int v12; // r9d
  __int64 *v13; // rcx
  __int64 v14; // r14
  __int64 *v15; // rdx
  unsigned int v16; // r15d
  unsigned int v17; // r9d
  __int64 *v18; // rcx
  __int64 v19; // r14
  __int64 v20; // rsi
  unsigned int v21; // r15d
  __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 *v24; // r14
  __int64 v25; // r10
  int v26; // r11d
  __int64 v27; // r9
  unsigned int v28; // ebx
  __int64 *v29; // rcx
  __int64 v30; // r12
  __int64 *v31; // rdx
  unsigned int v32; // r12d
  unsigned int v33; // r13d
  __int64 *v34; // rcx
  __int64 v35; // rbx
  int v36; // ecx
  int v38; // ecx
  int v39; // ecx
  int v40; // r13d
  int v41; // ecx
  int v42; // r15d
  int v43; // [rsp+0h] [rbp-4Ch]
  __int64 v45; // [rsp+Ch] [rbp-40h]
  int v46; // [rsp+Ch] [rbp-40h]

  v45 = (a3 - 1) / 2;
  if ( a2 >= v45 )
  {
    v6 = a2;
    if ( (a3 & 1) != 0 )
      goto LABEL_44;
    goto LABEL_27;
  }
  for ( i = a2; ; i = v6 )
  {
    v6 = 2 * i + 2;
    v7 = *(unsigned int *)(*(_QWORD *)a5 + 48LL);
    v8 = *(_QWORD *)(a1 + 8 * (2 * i + 1));
    if ( !(_DWORD)v7 )
      goto LABEL_49;
    v9 = *(_QWORD *)(a1 + 8 * v6);
    v10 = *(_QWORD *)(*(_QWORD *)a5 + 32LL);
    v11 = v7 - 1;
    v12 = (v7 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v13 = (__int64 *)(v10 + 16LL * v12);
    v14 = *v13;
    if ( v9 != *v13 )
    {
      v41 = 1;
      while ( v14 != -8 )
      {
        v42 = v41 + 1;
        v12 = v11 & (v41 + v12);
        v13 = (__int64 *)(v10 + 16LL * v12);
        v14 = *v13;
        if ( v9 == *v13 )
          goto LABEL_6;
        v41 = v42;
      }
LABEL_49:
      BUG();
    }
LABEL_6:
    v15 = (__int64 *)(v10 + 16 * v7);
    if ( v13 == v15 )
      BUG();
    v16 = *(_DWORD *)(v13[1] + 16);
    v17 = v11 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v18 = (__int64 *)(v10 + 16LL * v17);
    v19 = *v18;
    if ( v8 != *v18 )
    {
      v38 = 1;
      while ( v19 != -8 )
      {
        v17 = v11 & (v38 + v17);
        v43 = v38 + 1;
        v18 = (__int64 *)(v10 + 16LL * v17);
        v19 = *v18;
        if ( v8 == *v18 )
          goto LABEL_8;
        v38 = v43;
      }
LABEL_48:
      BUG();
    }
LABEL_8:
    if ( v15 == v18 )
      goto LABEL_48;
    if ( v16 > *(_DWORD *)(v18[1] + 16) )
    {
      v9 = *(_QWORD *)(a1 + 8 * (2 * i + 1));
      v6 = 2 * i + 1;
    }
    *(_QWORD *)(a1 + 8 * i) = v9;
    if ( v6 >= v45 )
      break;
  }
  if ( (a3 & 1) == 0 )
  {
LABEL_27:
    if ( (a3 - 2) / 2 == v6 )
    {
      *(_QWORD *)(a1 + 8 * v6) = *(_QWORD *)(a1 + 8 * (2 * v6 + 1));
      v6 = 2 * v6 + 1;
    }
  }
  v20 = (v6 - 1) / 2;
  if ( v6 <= a2 )
  {
LABEL_44:
    v24 = (__int64 *)(a1 + 8 * v6);
    goto LABEL_30;
  }
  v21 = ((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4);
  while ( 1 )
  {
    v23 = *(unsigned int *)(*(_QWORD *)a5 + 48LL);
    if ( !(_DWORD)v23 )
      goto LABEL_47;
    v24 = (__int64 *)(a1 + 8 * v20);
    v25 = *(_QWORD *)(*(_QWORD *)a5 + 32LL);
    v26 = v23 - 1;
    v27 = *v24;
    v28 = (v23 - 1) & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
    v29 = (__int64 *)(v25 + 16LL * v28);
    v30 = *v29;
    if ( *v29 != *v24 )
    {
      v39 = 1;
      while ( v30 != -8 )
      {
        v40 = v39 + 1;
        v28 = v26 & (v39 + v28);
        v29 = (__int64 *)(v25 + 16LL * v28);
        v30 = *v29;
        if ( v27 == *v29 )
          goto LABEL_21;
        v39 = v40;
      }
LABEL_47:
      BUG();
    }
LABEL_21:
    v31 = (__int64 *)(v25 + 16 * v23);
    if ( v31 == v29 )
      BUG();
    v32 = v21 & v26;
    v33 = *(_DWORD *)(v29[1] + 16);
    v34 = (__int64 *)(v25 + 16LL * (v21 & v26));
    v35 = *v34;
    if ( a4 != *v34 )
    {
      v36 = 1;
      while ( v35 != -8 )
      {
        v32 = v26 & (v36 + v32);
        v46 = v36 + 1;
        v34 = (__int64 *)(v25 + 16LL * v32);
        v35 = *v34;
        if ( a4 == *v34 )
          goto LABEL_15;
        v36 = v46;
      }
LABEL_50:
      BUG();
    }
LABEL_15:
    if ( v31 == v34 )
      goto LABEL_50;
    v22 = (__int64 *)(a1 + 8 * v6);
    if ( v33 <= *(_DWORD *)(v34[1] + 16) )
      break;
    *v22 = v27;
    v6 = v20;
    if ( a2 >= v20 )
      goto LABEL_30;
    v20 = (v20 - 1) / 2;
  }
  v24 = v22;
LABEL_30:
  *v24 = a4;
  return a4;
}
