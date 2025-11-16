// Function: sub_1E20BB0
// Address: 0x1e20bb0
//
__int64 __fastcall sub_1E20BB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 i; // r13
  __int64 v7; // r14
  __int64 *v8; // rbx
  __int64 v9; // r12
  __int64 v10; // r11
  __int64 j; // r13
  unsigned int v12; // r8d
  __int64 v13; // rdi
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r10
  unsigned int v17; // r9d
  __int64 v18; // rcx
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r10
  __int64 *v22; // rdx
  unsigned int v23; // esi
  __int64 v24; // rcx
  __int64 *v25; // r9
  int v26; // edx
  __int64 *v27; // r11
  int v28; // eax
  int v29; // eax
  __int64 v30; // rax
  int v32; // eax
  __int64 v33; // rax
  __int64 v35; // [rsp+8h] [rbp-88h]
  __int64 v37; // [rsp+20h] [rbp-70h]
  int v38; // [rsp+20h] [rbp-70h]
  __int64 v40; // [rsp+30h] [rbp-60h]
  __int64 *v41; // [rsp+30h] [rbp-60h]
  int v42; // [rsp+30h] [rbp-60h]
  __int64 v43; // [rsp+38h] [rbp-58h] BYREF
  __int64 v44; // [rsp+48h] [rbp-48h] BYREF
  __int64 v45; // [rsp+50h] [rbp-40h] BYREF
  _QWORD v46[7]; // [rsp+58h] [rbp-38h] BYREF

  v43 = a5;
  v37 = (a3 - 1) / 2;
  v35 = a3 & 1;
  if ( a2 >= v37 )
  {
    v8 = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_33;
    v7 = a2;
  }
  else
  {
    for ( i = a2; ; i = v7 )
    {
      v7 = 2 * (i + 1);
      v8 = (__int64 *)(a1 + 16 * (i + 1));
      if ( sub_1E20950(&v43, *v8, *(_QWORD *)(a1 + 8 * (v7 - 1))) )
        v8 = (__int64 *)(a1 + 8 * --v7);
      *(_QWORD *)(a1 + 8 * i) = *v8;
      if ( v7 >= v37 )
        break;
    }
    if ( v35 )
      goto LABEL_8;
  }
  if ( (a3 - 2) / 2 == v7 )
  {
    v7 = 2 * v7 + 1;
    *v8 = *(_QWORD *)(a1 + 8 * v7);
    v8 = (__int64 *)(a1 + 8 * v7);
  }
LABEL_8:
  v9 = v43;
  if ( v7 <= a2 )
    goto LABEL_33;
  v10 = a4;
  for ( j = (v7 - 1) / 2; ; j = (j - 1) / 2 )
  {
    v8 = (__int64 *)(a1 + 8 * j);
    v23 = *(_DWORD *)(v9 + 24);
    v45 = v10;
    v24 = *v8;
    v44 = *v8;
    if ( !v23 )
    {
      ++*(_QWORD *)v9;
LABEL_18:
      v40 = v10;
      v23 *= 2;
      goto LABEL_19;
    }
    v12 = v23 - 1;
    v13 = *(_QWORD *)(v9 + 8);
    v14 = (v23 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
    v15 = (__int64 *)(v13 + 16LL * v14);
    v16 = *v15;
    if ( v24 == *v15 )
    {
LABEL_11:
      v17 = *((_DWORD *)v15 + 2);
      v18 = v10;
      goto LABEL_12;
    }
    v42 = 1;
    v25 = 0;
    while ( v16 != -8 )
    {
      if ( !v25 && v16 == -16 )
        v25 = v15;
      v14 = v12 & (v42 + v14);
      v15 = (__int64 *)(v13 + 16LL * v14);
      v16 = *v15;
      if ( v24 == *v15 )
        goto LABEL_11;
      ++v42;
    }
    if ( !v25 )
      v25 = v15;
    v32 = *(_DWORD *)(v9 + 16);
    ++*(_QWORD *)v9;
    v26 = v32 + 1;
    if ( 4 * (v32 + 1) >= 3 * v23 )
      goto LABEL_18;
    if ( v23 - *(_DWORD *)(v9 + 20) - v26 > v23 >> 3 )
      goto LABEL_40;
    v40 = v10;
LABEL_19:
    sub_1E20790(v9, v23);
    sub_1E1F1A0(v9, &v44, v46);
    v25 = (__int64 *)v46[0];
    v10 = v40;
    v26 = *(_DWORD *)(v9 + 16) + 1;
LABEL_40:
    *(_DWORD *)(v9 + 16) = v26;
    if ( *v25 != -8 )
      --*(_DWORD *)(v9 + 20);
    v33 = v44;
    *((_DWORD *)v25 + 2) = 0;
    *v25 = v33;
    v23 = *(_DWORD *)(v9 + 24);
    if ( !v23 )
    {
      ++*(_QWORD *)v9;
LABEL_44:
      v23 *= 2;
      goto LABEL_45;
    }
    v13 = *(_QWORD *)(v9 + 8);
    v18 = v45;
    v12 = v23 - 1;
    v17 = 0;
LABEL_12:
    v19 = v12 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
    v20 = (__int64 *)(v13 + 16LL * v19);
    v21 = *v20;
    if ( *v20 != v18 )
      break;
LABEL_13:
    v22 = (__int64 *)(a1 + 8 * v7);
    if ( *((_DWORD *)v20 + 2) <= v17 )
    {
      v8 = (__int64 *)(a1 + 8 * v7);
      goto LABEL_33;
    }
    v7 = j;
    *v22 = *v8;
    if ( a2 >= j )
      goto LABEL_33;
  }
  v38 = 1;
  v41 = 0;
  while ( v21 != -8 )
  {
    if ( v21 == -16 )
    {
      if ( v41 )
        v20 = v41;
      v41 = v20;
    }
    v19 = v12 & (v38 + v19);
    v20 = (__int64 *)(v13 + 16LL * v19);
    v21 = *v20;
    if ( *v20 == v18 )
      goto LABEL_13;
    ++v38;
  }
  v27 = v41;
  if ( !v41 )
    v27 = v20;
  v28 = *(_DWORD *)(v9 + 16);
  ++*(_QWORD *)v9;
  v29 = v28 + 1;
  if ( 4 * v29 >= 3 * v23 )
    goto LABEL_44;
  if ( v23 - (v29 + *(_DWORD *)(v9 + 20)) > v23 >> 3 )
    goto LABEL_30;
LABEL_45:
  sub_1E20790(v9, v23);
  sub_1E1F1A0(v9, &v45, v46);
  v27 = (__int64 *)v46[0];
  v29 = *(_DWORD *)(v9 + 16) + 1;
LABEL_30:
  *(_DWORD *)(v9 + 16) = v29;
  if ( *v27 != -8 )
    --*(_DWORD *)(v9 + 20);
  v30 = v45;
  *((_DWORD *)v27 + 2) = 0;
  v8 = (__int64 *)(a1 + 8 * v7);
  *v27 = v30;
LABEL_33:
  *v8 = a4;
  return a4;
}
