// Function: sub_196D8D0
// Address: 0x196d8d0
//
__int64 __fastcall sub_196D8D0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v8; // r14
  unsigned int v9; // esi
  __int64 v10; // rcx
  int v11; // r10d
  __int64 *v12; // r13
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r8
  __int64 v16; // rax
  int v18; // eax
  int v19; // edx
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rsi
  char *v23; // rdi
  int v24; // eax
  __int64 v25; // rdi
  unsigned int v26; // eax
  __int64 v27; // rsi
  __int64 *v28; // r8
  int v29; // eax
  int v30; // eax
  __int64 v31; // rsi
  int v32; // r8d
  unsigned int v33; // r15d
  __int64 *v34; // rdi
  __int64 v35; // [rsp+50h] [rbp-90h] BYREF
  char *v36; // [rsp+58h] [rbp-88h] BYREF
  __int64 v37; // [rsp+60h] [rbp-80h]
  _BYTE v38[120]; // [rsp+68h] [rbp-78h] BYREF

  v8 = *a2;
  v9 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_28;
  }
  v10 = *(_QWORD *)(a1 + 8);
  v11 = 1;
  v12 = 0;
  v13 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v14 = (__int64 *)(v10 + 16LL * v13);
  v15 = *v14;
  if ( v8 == *v14 )
  {
LABEL_3:
    v16 = *((unsigned int *)v14 + 2);
    return *(_QWORD *)(a1 + 32) + 88 * v16 + 8;
  }
  while ( v15 != -8 )
  {
    if ( v15 == -16 && !v12 )
      v12 = v14;
    a6 = v11 + 1;
    v13 = (v9 - 1) & (v11 + v13);
    v14 = (__int64 *)(v10 + 16LL * v13);
    v15 = *v14;
    if ( v8 == *v14 )
      goto LABEL_3;
    ++v11;
  }
  if ( !v12 )
    v12 = v14;
  v18 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v9 )
  {
LABEL_28:
    sub_177C7D0(a1, 2 * v9);
    v24 = *(_DWORD *)(a1 + 24);
    if ( v24 )
    {
      v20 = (unsigned int)(v24 - 1);
      v25 = *(_QWORD *)(a1 + 8);
      v26 = v20 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v12 = (__int64 *)(v25 + 16LL * v26);
      v27 = *v12;
      if ( v8 != *v12 )
      {
        a6 = 1;
        v28 = 0;
        while ( v27 != -8 )
        {
          if ( !v28 && v27 == -16 )
            v28 = v12;
          v26 = v20 & (a6 + v26);
          v12 = (__int64 *)(v25 + 16LL * v26);
          v27 = *v12;
          if ( v8 == *v12 )
            goto LABEL_15;
          ++a6;
        }
        if ( v28 )
          v12 = v28;
      }
      goto LABEL_15;
    }
    goto LABEL_51;
  }
  v20 = v9 >> 3;
  if ( v9 - *(_DWORD *)(a1 + 20) - v19 <= (unsigned int)v20 )
  {
    sub_177C7D0(a1, v9);
    v29 = *(_DWORD *)(a1 + 24);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 8);
      v32 = 1;
      v33 = v30 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v34 = 0;
      v12 = (__int64 *)(v31 + 16LL * v33);
      v20 = *v12;
      if ( v8 != *v12 )
      {
        while ( v20 != -8 )
        {
          if ( !v34 && v20 == -16 )
            v34 = v12;
          a6 = v32 + 1;
          v33 = v30 & (v32 + v33);
          v12 = (__int64 *)(v31 + 16LL * v33);
          v20 = *v12;
          if ( v8 == *v12 )
            goto LABEL_15;
          ++v32;
        }
        if ( v34 )
          v12 = v34;
      }
      goto LABEL_15;
    }
LABEL_51:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v19;
  if ( *v12 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v12 = v8;
  *((_DWORD *)v12 + 2) = 0;
  v21 = *a2;
  v22 = *(_QWORD *)(a1 + 40);
  v35 = *a2;
  v36 = v38;
  v37 = 0x800000000LL;
  if ( v22 == *(_QWORD *)(a1 + 48) )
  {
    sub_196D260((__int64 *)(a1 + 32), (char *)v22, (__int64)&v35, v20);
    v23 = v36;
  }
  else
  {
    v23 = v38;
    if ( v22 )
    {
      *(_QWORD *)v22 = v21;
      *(_QWORD *)(v22 + 8) = v22 + 24;
      *(_QWORD *)(v22 + 16) = 0x800000000LL;
      if ( (_DWORD)v37 )
        sub_1968FC0(v22 + 8, &v36, v22 + 24, v20, (int)&v36, a6);
      v22 = *(_QWORD *)(a1 + 40);
      v23 = v36;
    }
    *(_QWORD *)(a1 + 40) = v22 + 88;
  }
  if ( v23 != v38 )
    _libc_free((unsigned __int64)v23);
  v16 = -1171354717 * (unsigned int)((__int64)(*(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32)) >> 3) - 1;
  *((_DWORD *)v12 + 2) = v16;
  return *(_QWORD *)(a1 + 32) + 88 * v16 + 8;
}
