// Function: sub_1CD40B0
// Address: 0x1cd40b0
//
__int64 __fastcall sub_1CD40B0(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  __int64 v9; // rsi
  __int64 v10; // rcx
  int v12; // r15d
  __int64 v13; // r14
  __int64 *v14; // rax
  __int64 v15; // r8
  __int64 *v16; // r9
  __int64 v17; // rdx
  __int64 v18; // r10
  __int64 *v19; // r9
  __int64 v20; // r11
  __int64 **v23; // rax
  __int64 *v24; // r12
  __int64 v25; // r15
  __int64 *v26; // r13
  __int64 *v27; // r14
  __int64 v28; // r11
  unsigned int v29; // r10d
  int v30; // r9d
  int v31; // r9d
  int v32; // r11d
  __int64 v33; // r9
  int v34; // r9d
  __int64 *v35; // rdx
  int v36; // eax
  int v37; // eax
  int v38; // [rsp+Ch] [rbp-64h]
  unsigned int v39; // [rsp+10h] [rbp-60h]
  __int64 v40; // [rsp+10h] [rbp-60h]
  __int64 *v42; // [rsp+20h] [rbp-50h]
  int v43; // [rsp+20h] [rbp-50h]
  __int64 v44; // [rsp+28h] [rbp-48h] BYREF
  __int64 *v45; // [rsp+38h] [rbp-38h] BYREF

  v44 = a1;
  if ( (unsigned int)dword_4FC05C0 <= 1 )
    return 0;
  if ( !a4 )
    return 0;
  v7 = *(_QWORD *)(a2 + 8);
  v9 = *(unsigned int *)(a2 + 24);
  if ( !(_DWORD)v9 )
    return 0;
  v10 = v44;
  v12 = v9 - 1;
  LODWORD(v13) = (v9 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
  v14 = (__int64 *)(v7 + 16LL * (unsigned int)v13);
  v15 = *v14;
  v16 = v14;
  if ( v44 != *v14 )
  {
    v28 = *v14;
    v29 = (v9 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
    v30 = 1;
    while ( v28 != -8 )
    {
      v29 = v12 & (v30 + v29);
      v43 = v30 + 1;
      v16 = (__int64 *)(v7 + 16LL * v29);
      v28 = *v16;
      if ( v44 == *v16 )
        goto LABEL_5;
      v30 = v43;
    }
    return 0;
  }
LABEL_5:
  if ( (__int64 *)(v7 + 16 * v9) == v16 )
    return 0;
  v17 = *(unsigned int *)(a6 + 24);
  if ( (_DWORD)v17 )
  {
    v18 = *(_QWORD *)(a6 + 8);
    v39 = (v17 - 1) & (((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4));
    v19 = (__int64 *)(v18 + 8LL * v39);
    v20 = *v19;
    if ( a5 == *v19 )
    {
LABEL_8:
      if ( v19 != (__int64 *)(v18 + 8 * v17) )
        return 0;
    }
    else
    {
      v31 = 1;
      while ( v20 != -8 )
      {
        v32 = v31 + 1;
        v33 = ((_DWORD)v17 - 1) & (v39 + v31);
        v39 = v33;
        v19 = (__int64 *)(v18 + 8 * v33);
        v38 = v32;
        v20 = *v19;
        if ( a5 == *v19 )
          goto LABEL_8;
        v31 = v38;
      }
    }
  }
  if ( v44 == v15 )
  {
LABEL_12:
    v23 = (__int64 **)v14[1];
    goto LABEL_13;
  }
  v34 = 1;
  v35 = 0;
  while ( v15 != -8 )
  {
    if ( v15 == -16 && !v35 )
      v35 = v14;
    v13 = v12 & (unsigned int)(v13 + v34);
    v14 = (__int64 *)(v7 + 16 * v13);
    v15 = *v14;
    if ( v44 == *v14 )
      goto LABEL_12;
    ++v34;
  }
  if ( !v35 )
    v35 = v14;
  v36 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  v37 = v36 + 1;
  if ( 4 * v37 >= (unsigned int)(3 * v9) )
  {
    LODWORD(v9) = 2 * v9;
    goto LABEL_49;
  }
  if ( (int)v9 - *(_DWORD *)(a2 + 20) - v37 <= (unsigned int)v9 >> 3 )
  {
LABEL_49:
    sub_1CD3DF0(a2, v9);
    sub_1CD30F0(a2, &v44, &v45);
    v35 = v45;
    v10 = v44;
    v37 = *(_DWORD *)(a2 + 16) + 1;
  }
  *(_DWORD *)(a2 + 16) = v37;
  if ( *v35 != -8 )
    --*(_DWORD *)(a2 + 20);
  *v35 = v10;
  v23 = 0;
  v35[1] = 0;
LABEL_13:
  v42 = v23[1];
  if ( v42 == *v23 )
    return 0;
  v40 = a5;
  v24 = *v23;
  while ( 1 )
  {
    v25 = *v24;
    if ( *v24 && sub_15CC890(a3, *(_QWORD *)(v25 + 40), v40) )
    {
      v26 = *(__int64 **)(a6 + 8);
      v27 = &v26[*(unsigned int *)(a6 + 24)];
      if ( !*(_DWORD *)(a6 + 16) || v26 == v27 )
        return v25;
      while ( *v26 == -16 || *v26 == -8 )
      {
        if ( ++v26 == v27 )
          return v25;
      }
LABEL_25:
      if ( v26 == v27 )
        return v25;
      if ( !sub_15CC890(a3, *(_QWORD *)(v25 + 40), *v26) )
        break;
    }
    if ( v42 == ++v24 )
      return 0;
  }
  while ( 1 )
  {
    if ( ++v26 == v27 )
      return v25;
    if ( *v26 != -16 && *v26 != -8 )
      goto LABEL_25;
  }
}
