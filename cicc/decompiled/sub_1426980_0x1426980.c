// Function: sub_1426980
// Address: 0x1426980
//
_QWORD *__fastcall sub_1426980(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v5; // r15d
  unsigned int v10; // esi
  __int64 v11; // rdx
  __int64 v12; // r10
  __int64 v13; // r8
  unsigned int v14; // edi
  __int64 *v15; // rax
  __int64 v16; // r11
  unsigned int v17; // r9d
  __int64 *v18; // rax
  int v19; // r11d
  __int64 *v20; // rdi
  int v21; // eax
  int v22; // eax
  int v23; // eax
  int v24; // edx
  __int64 v25; // rsi
  unsigned int v26; // ebx
  int v27; // r9d
  int v28; // eax
  __int64 v29; // rsi
  unsigned int v30; // edx
  int v31; // eax
  int v32; // r9d
  int v33; // r10d
  __int64 *v34; // r9

  v5 = a4;
  if ( *(_BYTE *)(a2 + 16) != 23 )
    goto LABEL_2;
  v10 = *(_DWORD *)(a1 + 48);
  v11 = *(_QWORD *)(a1 + 32);
  v12 = a1 + 24;
  if ( !v10 )
    goto LABEL_20;
  v13 = *(_QWORD *)(a2 + 64);
  a4 = v10 - 1;
  v14 = a4 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
  v15 = (__int64 *)(v11 + 16LL * v14);
  v16 = *v15;
  if ( v13 == *v15 )
  {
LABEL_5:
    *v15 = -16;
    v10 = *(_DWORD *)(a1 + 48);
    --*(_DWORD *)(a1 + 40);
    v11 = *(_QWORD *)(a1 + 32);
    ++*(_DWORD *)(a1 + 44);
    if ( v10 )
    {
      a4 = v10 - 1;
      goto LABEL_7;
    }
LABEL_20:
    ++*(_QWORD *)(a1 + 24);
    v10 = 0;
    goto LABEL_21;
  }
  v31 = 1;
  while ( v16 != -8 )
  {
    v32 = v31 + 1;
    v14 = a4 & (v31 + v14);
    v15 = (__int64 *)(v11 + 16LL * v14);
    v16 = *v15;
    if ( v13 == *v15 )
      goto LABEL_5;
    v31 = v32;
  }
LABEL_7:
  v17 = a4 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v18 = (__int64 *)(v11 + 16LL * v17);
  a5 = *v18;
  if ( a3 == *v18 )
    goto LABEL_2;
  v19 = 1;
  v20 = 0;
  while ( a5 != -8 )
  {
    if ( a5 != -16 || v20 )
      v18 = v20;
    v17 = a4 & (v19 + v17);
    a5 = *(_QWORD *)(v11 + 16LL * v17);
    if ( a3 == a5 )
      goto LABEL_2;
    ++v19;
    v20 = v18;
    v18 = (__int64 *)(v11 + 16LL * v17);
  }
  if ( !v20 )
    v20 = v18;
  v21 = *(_DWORD *)(a1 + 40);
  ++*(_QWORD *)(a1 + 24);
  v22 = v21 + 1;
  if ( 4 * v22 >= 3 * v10 )
  {
LABEL_21:
    sub_14267C0(v12, 2 * v10);
    v28 = *(_DWORD *)(a1 + 48);
    if ( v28 )
    {
      a4 = (unsigned int)(v28 - 1);
      v29 = *(_QWORD *)(a1 + 32);
      v30 = a4 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v22 = *(_DWORD *)(a1 + 40) + 1;
      v20 = (__int64 *)(v29 + 16LL * v30);
      a5 = *v20;
      if ( a3 != *v20 )
      {
        v33 = 1;
        v34 = 0;
        while ( a5 != -8 )
        {
          if ( !v34 && a5 == -16 )
            v34 = v20;
          v30 = a4 & (v33 + v30);
          v20 = (__int64 *)(v29 + 16LL * v30);
          a5 = *v20;
          if ( a3 == *v20 )
            goto LABEL_23;
          ++v33;
        }
        if ( v34 )
          v20 = v34;
      }
      goto LABEL_23;
    }
    goto LABEL_50;
  }
  a4 = v10 - (v22 + *(_DWORD *)(a1 + 44));
  if ( (unsigned int)a4 <= v10 >> 3 )
  {
    sub_14267C0(v12, v10);
    v23 = *(_DWORD *)(a1 + 48);
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 32);
      a5 = 0;
      v26 = (v23 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v27 = 1;
      v22 = *(_DWORD *)(a1 + 40) + 1;
      v20 = (__int64 *)(v25 + 16LL * v26);
      a4 = *v20;
      if ( a3 != *v20 )
      {
        while ( a4 != -8 )
        {
          if ( a4 == -16 && !a5 )
            a5 = (__int64)v20;
          v26 = v24 & (v27 + v26);
          v20 = (__int64 *)(v25 + 16LL * v26);
          a4 = *v20;
          if ( a3 == *v20 )
            goto LABEL_23;
          ++v27;
        }
        if ( a5 )
          v20 = (__int64 *)a5;
      }
      goto LABEL_23;
    }
LABEL_50:
    ++*(_DWORD *)(a1 + 40);
    BUG();
  }
LABEL_23:
  *(_DWORD *)(a1 + 40) = v22;
  if ( *v20 != -8 )
    --*(_DWORD *)(a1 + 44);
  *v20 = a3;
  v20[1] = a2;
LABEL_2:
  sub_1422460(a1, a2, 0, a4, a5);
  *(_QWORD *)(a2 + 64) = a3;
  return sub_14264F0(a1, a2, a3, v5);
}
