// Function: sub_2685EC0
// Address: 0x2685ec0
//
__int64 __fastcall sub_2685EC0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // r13
  unsigned int v9; // esi
  __int64 v10; // rax
  __int64 v11; // rdi
  int v12; // r10d
  _QWORD *v13; // r9
  unsigned int v14; // edx
  _QWORD *v15; // rax
  __int64 v16; // rcx
  int v17; // edx
  __int64 v18; // rcx
  __int64 *v19; // rax
  int v20; // eax
  int v21; // esi
  __int64 v22; // rdi
  unsigned int v23; // edx
  __int64 v24; // r8
  int v25; // edi
  int v26; // eax
  int v27; // r10d
  __int64 *v28; // r9
  int v29; // eax
  __int64 *v30; // [rsp+8h] [rbp-38h] BYREF
  __int64 v31; // [rsp+10h] [rbp-30h] BYREF
  __int64 v32; // [rsp+18h] [rbp-28h]

  v2 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)v2 != 85 || a2 != v2 - 32 )
    return 0;
  if ( *(char *)(v2 + 7) < 0 )
  {
    v5 = sub_BD2BC0(*(_QWORD *)(a2 + 24));
    v7 = v5 + v6;
    if ( *(char *)(v2 + 7) < 0 )
      v7 -= sub_BD2BC0(v2);
    if ( (unsigned int)(v7 >> 4) )
      return 0;
  }
  v8 = *a1;
  v9 = *(_DWORD *)(*a1 + 24);
  v10 = *(_QWORD *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF));
  v31 = v2;
  v32 = v10;
  if ( !v9 )
  {
    v30 = 0;
    ++*(_QWORD *)v8;
LABEL_25:
    sub_2685CE0(v8, 2 * v9);
    v20 = *(_DWORD *)(v8 + 24);
    if ( v20 )
    {
      v18 = v31;
      v21 = v20 - 1;
      v22 = *(_QWORD *)(v8 + 8);
      v23 = (v20 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v19 = (__int64 *)(v22 + 16LL * v23);
      v24 = *v19;
      if ( *v19 == v31 )
      {
LABEL_27:
        v25 = *(_DWORD *)(v8 + 16);
        v30 = v19;
        v17 = v25 + 1;
      }
      else
      {
        v27 = 1;
        v28 = 0;
        while ( v24 != -4096 )
        {
          if ( !v28 && v24 == -8192 )
            v28 = v19;
          v23 = v21 & (v27 + v23);
          v19 = (__int64 *)(v22 + 16LL * v23);
          v24 = *v19;
          if ( v31 == *v19 )
            goto LABEL_27;
          ++v27;
        }
        if ( !v28 )
          v28 = v19;
        v29 = *(_DWORD *)(v8 + 16);
        v30 = v28;
        v17 = v29 + 1;
        v19 = v28;
      }
    }
    else
    {
      v26 = *(_DWORD *)(v8 + 16);
      v18 = v31;
      v30 = 0;
      v17 = v26 + 1;
      v19 = 0;
    }
    goto LABEL_17;
  }
  v11 = *(_QWORD *)(v8 + 8);
  v12 = 1;
  v13 = 0;
  v14 = (v9 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v15 = (_QWORD *)(v11 + 16LL * v14);
  v16 = *v15;
  if ( v2 == *v15 )
    return 0;
  while ( v16 != -4096 )
  {
    if ( v13 || v16 != -8192 )
      v15 = v13;
    v14 = (v9 - 1) & (v12 + v14);
    v16 = *(_QWORD *)(v11 + 16LL * v14);
    if ( v2 == v16 )
      return 0;
    ++v12;
    v13 = v15;
    v15 = (_QWORD *)(v11 + 16LL * v14);
  }
  if ( !v13 )
    v13 = v15;
  v17 = *(_DWORD *)(v8 + 16) + 1;
  v30 = v13;
  ++*(_QWORD *)v8;
  if ( 4 * v17 >= 3 * v9 )
    goto LABEL_25;
  if ( v9 - *(_DWORD *)(v8 + 20) - v17 <= v9 >> 3 )
  {
    sub_2685CE0(v8, v9);
    sub_2677F80(v8, &v31, &v30);
    v18 = v31;
    v17 = *(_DWORD *)(v8 + 16) + 1;
  }
  else
  {
    v18 = v31;
  }
  v19 = v30;
LABEL_17:
  *(_DWORD *)(v8 + 16) = v17;
  if ( *v19 != -4096 )
    --*(_DWORD *)(v8 + 20);
  *v19 = v18;
  v19[1] = v32;
  *(_DWORD *)a1[1] = 0;
  return 0;
}
