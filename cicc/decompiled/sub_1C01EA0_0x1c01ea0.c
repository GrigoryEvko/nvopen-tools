// Function: sub_1C01EA0
// Address: 0x1c01ea0
//
__int64 __fastcall sub_1C01EA0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v5; // rcx
  __int64 v6; // rsi
  int v7; // edi
  unsigned int v8; // r8d
  __int64 *v9; // rax
  __int64 v10; // r9
  unsigned int v11; // r9d
  __int64 *v12; // rax
  __int64 v13; // r8
  int v15; // eax
  int v16; // r15d
  _QWORD *v17; // rax
  _QWORD *v18; // r14
  unsigned int v19; // esi
  __int64 v20; // rcx
  __int64 v21; // rdi
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // r9
  int v25; // eax
  int v26; // esi
  __int64 v27; // r8
  unsigned int v28; // ecx
  int v29; // eax
  __int64 *v30; // rdx
  __int64 v31; // rdi
  int v32; // r10d
  __int64 *v33; // r9
  int v34; // r10d
  int v35; // eax
  int v36; // r10d
  int v37; // eax
  int v38; // ecx
  __int64 v39; // rdi
  int v40; // r9d
  unsigned int v41; // r14d
  __int64 *v42; // r8
  __int64 v43; // rsi
  int v44; // r11d
  __int64 *v45; // r10
  int v46; // edx
  int v47; // edi
  __int64 *v48; // r11
  __int64 v49; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v50; // [rsp+8h] [rbp-38h] BYREF

  v2 = a1 + 80;
  v5 = *(_QWORD *)(a1 + 88);
  v6 = *(unsigned int *)(a1 + 104);
  if ( (_DWORD)v6 )
  {
    v7 = v6 - 1;
    v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v5 + 16LL * v8);
    v10 = *v9;
    if ( *v9 == a2 )
    {
LABEL_3:
      if ( (__int64 *)(v5 + 16 * v6) != v9 )
        goto LABEL_4;
    }
    else
    {
      v15 = 1;
      while ( v10 != -8 )
      {
        v36 = v15 + 1;
        v8 = v7 & (v15 + v8);
        v9 = (__int64 *)(v5 + 16LL * v8);
        v10 = *v9;
        if ( *v9 == a2 )
          goto LABEL_3;
        v15 = v36;
      }
    }
  }
  v49 = a2;
  v16 = *(_DWORD *)a1;
  v17 = (_QWORD *)sub_22077B0(32);
  v18 = v17;
  if ( v17 )
    sub_1BFC440(v17, v16);
  v19 = *(_DWORD *)(a1 + 104);
  if ( !v19 )
  {
    ++*(_QWORD *)(a1 + 80);
    goto LABEL_48;
  }
  v20 = v49;
  v21 = *(_QWORD *)(a1 + 88);
  v22 = (v19 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
  v23 = (__int64 *)(v21 + 16LL * v22);
  v24 = *v23;
  if ( v49 != *v23 )
  {
    v44 = 1;
    v45 = 0;
    while ( v24 != -8 )
    {
      if ( v24 == -16 && !v45 )
        v45 = v23;
      v22 = (v19 - 1) & (v44 + v22);
      v23 = (__int64 *)(v21 + 16LL * v22);
      v24 = *v23;
      if ( v49 == *v23 )
        goto LABEL_12;
      ++v44;
    }
    v46 = *(_DWORD *)(a1 + 96);
    if ( v45 )
      v23 = v45;
    ++*(_QWORD *)(a1 + 80);
    v47 = v46 + 1;
    if ( 4 * (v46 + 1) < 3 * v19 )
    {
      if ( v19 - *(_DWORD *)(a1 + 100) - v47 > v19 >> 3 )
      {
LABEL_44:
        *(_DWORD *)(a1 + 96) = v47;
        if ( *v23 != -8 )
          --*(_DWORD *)(a1 + 100);
        *v23 = v20;
        v23[1] = 0;
        goto LABEL_12;
      }
LABEL_49:
      sub_1C01CE0(v2, v19);
      sub_1BFDA70(v2, &v49, &v50);
      v23 = v50;
      v20 = v49;
      v47 = *(_DWORD *)(a1 + 96) + 1;
      goto LABEL_44;
    }
LABEL_48:
    v19 *= 2;
    goto LABEL_49;
  }
LABEL_12:
  v23[1] = (__int64)v18;
  LODWORD(v6) = *(_DWORD *)(a1 + 104);
  v5 = *(_QWORD *)(a1 + 88);
  if ( !(_DWORD)v6 )
  {
    ++*(_QWORD *)(a1 + 80);
    goto LABEL_14;
  }
  v7 = v6 - 1;
LABEL_4:
  v11 = v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = (__int64 *)(v5 + 16LL * v11);
  v13 = *v12;
  if ( *v12 == a2 )
    return v12[1];
  v34 = 1;
  v30 = 0;
  while ( v13 != -8 )
  {
    if ( v13 != -16 || v30 )
      v12 = v30;
    v11 = v7 & (v34 + v11);
    v48 = (__int64 *)(v5 + 16LL * v11);
    v13 = *v48;
    if ( *v48 == a2 )
      return v48[1];
    ++v34;
    v30 = v12;
    v12 = (__int64 *)(v5 + 16LL * v11);
  }
  if ( !v30 )
    v30 = v12;
  v35 = *(_DWORD *)(a1 + 96);
  ++*(_QWORD *)(a1 + 80);
  v29 = v35 + 1;
  if ( 4 * v29 >= (unsigned int)(3 * v6) )
  {
LABEL_14:
    sub_1C01CE0(v2, 2 * v6);
    v25 = *(_DWORD *)(a1 + 104);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 88);
      v28 = (v25 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v29 = *(_DWORD *)(a1 + 96) + 1;
      v30 = (__int64 *)(v27 + 16LL * v28);
      v31 = *v30;
      if ( *v30 != a2 )
      {
        v32 = 1;
        v33 = 0;
        while ( v31 != -8 )
        {
          if ( v31 == -16 && !v33 )
            v33 = v30;
          v28 = v26 & (v32 + v28);
          v30 = (__int64 *)(v27 + 16LL * v28);
          v31 = *v30;
          if ( *v30 == a2 )
            goto LABEL_27;
          ++v32;
        }
        if ( v33 )
          v30 = v33;
      }
      goto LABEL_27;
    }
    goto LABEL_72;
  }
  if ( (int)v6 - (v29 + *(_DWORD *)(a1 + 100)) <= (unsigned int)v6 >> 3 )
  {
    sub_1C01CE0(v2, v6);
    v37 = *(_DWORD *)(a1 + 104);
    if ( v37 )
    {
      v38 = v37 - 1;
      v39 = *(_QWORD *)(a1 + 88);
      v40 = 1;
      v41 = (v37 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v42 = 0;
      v29 = *(_DWORD *)(a1 + 96) + 1;
      v30 = (__int64 *)(v39 + 16LL * v41);
      v43 = *v30;
      if ( *v30 != a2 )
      {
        while ( v43 != -8 )
        {
          if ( !v42 && v43 == -16 )
            v42 = v30;
          v41 = v38 & (v40 + v41);
          v30 = (__int64 *)(v39 + 16LL * v41);
          v43 = *v30;
          if ( *v30 == a2 )
            goto LABEL_27;
          ++v40;
        }
        if ( v42 )
          v30 = v42;
      }
      goto LABEL_27;
    }
LABEL_72:
    ++*(_DWORD *)(a1 + 96);
    BUG();
  }
LABEL_27:
  *(_DWORD *)(a1 + 96) = v29;
  if ( *v30 != -8 )
    --*(_DWORD *)(a1 + 100);
  *v30 = a2;
  v30[1] = 0;
  return 0;
}
