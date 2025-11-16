// Function: sub_31E4CF0
// Address: 0x31e4cf0
//
__int64 __fastcall sub_31E4CF0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  unsigned int v5; // esi
  __int64 v6; // rcx
  unsigned int v7; // edx
  _QWORD *v8; // rax
  __int64 v9; // r10
  __int64 v10; // r13
  int v12; // eax
  int v13; // ecx
  __int64 v14; // rdi
  unsigned int v15; // eax
  __int64 v16; // rsi
  int v17; // edx
  _BYTE *v18; // rsi
  __int64 v19; // rdx
  _QWORD *v20; // rbx
  const void *v21; // r13
  void *v22; // r12
  __int64 v23; // r14
  __int64 v24; // rdi
  int v25; // r14d
  int v26; // eax
  int v27; // eax
  __int64 v28; // rsi
  _QWORD *v29; // rdi
  unsigned int v30; // r13d
  int v31; // r9d
  __int64 v32; // rcx
  int v33; // r10d
  _QWORD *v34; // r9
  _QWORD *v35; // [rsp+18h] [rbp-88h]
  _QWORD *v36; // [rsp+18h] [rbp-88h]
  void *s1[2]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v38[2]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v39[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v40; // [rsp+60h] [rbp-40h]

  if ( !*(_BYTE *)(a2 + 43) )
    return 0;
  v4 = a1 + 456;
  v5 = *(_DWORD *)(a1 + 480);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 456);
    goto LABEL_8;
  }
  v6 = *(_QWORD *)(a1 + 464);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (_QWORD *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( a2 == *v8 )
    return v8[1];
  v36 = 0;
  v25 = 1;
  while ( v9 != -4096 )
  {
    if ( !v36 )
    {
      if ( v9 != -8192 )
        v8 = 0;
      v36 = v8;
    }
    v7 = (v5 - 1) & (v25 + v7);
    v8 = (_QWORD *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
      return v8[1];
    ++v25;
  }
  if ( v36 )
    v8 = v36;
  ++*(_QWORD *)(a1 + 456);
  v35 = v8;
  v17 = *(_DWORD *)(a1 + 472) + 1;
  if ( 4 * v17 >= 3 * v5 )
  {
LABEL_8:
    sub_31E4AF0(v4, 2 * v5);
    v12 = *(_DWORD *)(a1 + 480);
    if ( v12 )
    {
      v13 = v12 - 1;
      v14 = *(_QWORD *)(a1 + 464);
      v15 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v35 = (_QWORD *)(v14 + 16LL * v15);
      v16 = *v35;
      v17 = *(_DWORD *)(a1 + 472) + 1;
      if ( a2 != *v35 )
      {
        v33 = 1;
        v34 = 0;
        while ( v16 != -4096 )
        {
          if ( !v34 && v16 == -8192 )
            v34 = v35;
          v15 = v13 & (v33 + v15);
          v35 = (_QWORD *)(v14 + 16LL * v15);
          v16 = *v35;
          if ( a2 == *v35 )
            goto LABEL_10;
          ++v33;
        }
        if ( !v34 )
          v34 = v35;
        v35 = v34;
      }
      goto LABEL_10;
    }
    goto LABEL_57;
  }
  if ( v5 - *(_DWORD *)(a1 + 476) - v17 <= v5 >> 3 )
  {
    sub_31E4AF0(v4, v5);
    v26 = *(_DWORD *)(a1 + 480);
    if ( v26 )
    {
      v27 = v26 - 1;
      v28 = *(_QWORD *)(a1 + 464);
      v29 = 0;
      v30 = v27 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v31 = 1;
      v17 = *(_DWORD *)(a1 + 472) + 1;
      v35 = (_QWORD *)(v28 + 16LL * v30);
      v32 = *v35;
      if ( a2 != *v35 )
      {
        while ( v32 != -4096 )
        {
          if ( !v29 && v32 == -8192 )
            v29 = v35;
          v30 = v27 & (v31 + v30);
          v35 = (_QWORD *)(v28 + 16LL * v30);
          v32 = *v35;
          if ( a2 == *v35 )
            goto LABEL_10;
          ++v31;
        }
        if ( !v29 )
          v29 = v35;
        v35 = v29;
      }
      goto LABEL_10;
    }
LABEL_57:
    ++*(_DWORD *)(a1 + 472);
    BUG();
  }
LABEL_10:
  *(_DWORD *)(a1 + 472) = v17;
  if ( *v35 != -4096 )
    --*(_DWORD *)(a1 + 476);
  *v35 = a2;
  v35[1] = 0;
  v18 = *(_BYTE **)(a2 + 8);
  v19 = *(_QWORD *)(a2 + 16);
  s1[0] = v38;
  sub_31D5230((__int64 *)s1, v18, (__int64)&v18[v19]);
  v20 = (_QWORD *)unk_503B140;
  if ( !unk_503B140 )
  {
LABEL_41:
    v39[0] = "no GCMetadataPrinter registered for GC: ";
    v39[2] = s1;
    v40 = 1027;
    sub_C64D30((__int64)v39, 1u);
  }
  v21 = s1[0];
  v22 = s1[1];
  while ( 1 )
  {
    v23 = v20[1];
    if ( *(void **)(v23 + 8) == v22 && (!v22 || !memcmp(v21, *(const void **)v23, (size_t)v22)) )
      break;
    v20 = (_QWORD *)*v20;
    if ( !v20 )
      goto LABEL_41;
  }
  (*(void (__fastcall **)(_QWORD *))(v23 + 32))(v39);
  v10 = v39[0];
  *(_QWORD *)(v39[0] + 8LL) = a2;
  v39[0] = 0;
  v24 = v35[1];
  v35[1] = v10;
  if ( v24 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v24 + 8LL))(v24);
    v10 = v35[1];
    if ( v39[0] )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v39[0] + 8LL))(v39[0]);
  }
  if ( s1[0] != v38 )
    j_j___libc_free_0((unsigned __int64)s1[0]);
  return v10;
}
