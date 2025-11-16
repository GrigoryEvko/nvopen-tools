// Function: sub_16298C0
// Address: 0x16298c0
//
__int64 __fastcall sub_16298C0(__int64 a1, __int64 a2)
{
  int v3; // r12d
  __int64 v4; // r15
  unsigned int v5; // r9d
  int v6; // esi
  __int64 *v7; // rsi
  __int64 v8; // rdi
  int v9; // eax
  __int64 result; // rax
  __int64 *v11; // r14
  int v12; // r12d
  unsigned int v13; // eax
  int v14; // r9d
  int v15; // ecx
  __int64 *v16; // r13
  __int64 v17; // rax
  const void *v18; // rsi
  int v19; // eax
  __int64 v20; // r14
  __int64 v21; // r12
  int v22; // r12d
  unsigned int v23; // edx
  __int64 *v24; // r8
  __int64 v25; // rcx
  int v26; // r9d
  __int64 *v27; // r10
  int v28; // eax
  int v29; // [rsp+8h] [rbp-58h]
  int v30; // [rsp+Ch] [rbp-54h]
  size_t n; // [rsp+10h] [rbp-50h]
  __int64 v32; // [rsp+18h] [rbp-48h] BYREF
  __int64 *v33; // [rsp+28h] [rbp-38h] BYREF

  v3 = *(_DWORD *)(a2 + 24);
  v4 = *(_QWORD *)(a2 + 8);
  v32 = a1;
  if ( !v3 )
    goto LABEL_2;
  v11 = *(__int64 **)(a1 + 24);
  v12 = v3 - 1;
  n = *(_QWORD *)(a1 + 32) - (_QWORD)v11;
  v13 = sub_15B1DB0(v11, *(_QWORD *)(a1 + 32));
  v14 = 1;
  v15 = v12 & v13;
  v16 = (__int64 *)(v4 + 8LL * (v12 & v13));
  v17 = *v16;
  if ( *v16 == -8 )
  {
LABEL_18:
    v20 = *(_QWORD *)(a2 + 8);
    LODWORD(v21) = *(_DWORD *)(a2 + 24);
    goto LABEL_19;
  }
  while ( 1 )
  {
    if ( v17 != -16 )
    {
      v18 = *(const void **)(v17 + 24);
      if ( n == *(_QWORD *)(v17 + 32) - (_QWORD)v18 )
      {
        v29 = v14;
        v30 = v15;
        if ( !n )
          break;
        v19 = memcmp(v11, v18, n);
        v15 = v30;
        v14 = v29;
        if ( !v19 )
          break;
      }
    }
    v16 = (__int64 *)(v4 + 8LL * (v12 & (unsigned int)(v15 + v14)));
    v15 = v12 & (v15 + v14);
    v17 = *v16;
    if ( *v16 == -8 )
      goto LABEL_18;
    ++v14;
  }
  v20 = *(_QWORD *)(a2 + 8);
  v21 = *(unsigned int *)(a2 + 24);
  if ( v16 == (__int64 *)(v20 + 8 * v21) || (result = *v16) == 0 )
  {
LABEL_19:
    if ( (_DWORD)v21 )
    {
      v22 = v21 - 1;
      v8 = v32;
      v23 = v22 & sub_15B1DB0(*(__int64 **)(v32 + 24), *(_QWORD *)(v32 + 32));
      v24 = (__int64 *)(v20 + 8LL * v23);
      result = v32;
      v25 = *v24;
      if ( *v24 == v32 )
        return result;
      v26 = 1;
      v7 = 0;
      while ( v25 != -8 )
      {
        if ( v25 != -16 || v7 )
          v24 = v7;
        v23 = v22 & (v26 + v23);
        v27 = (__int64 *)(v20 + 8LL * v23);
        v25 = *v27;
        if ( *v27 == v32 )
          return result;
        ++v26;
        v7 = v24;
        v24 = (__int64 *)(v20 + 8LL * v23);
      }
      v28 = *(_DWORD *)(a2 + 16);
      v5 = *(_DWORD *)(a2 + 24);
      if ( !v7 )
        v7 = v24;
      ++*(_QWORD *)a2;
      v9 = v28 + 1;
      if ( 4 * v9 < 3 * v5 )
      {
        if ( v5 - (v9 + *(_DWORD *)(a2 + 20)) > v5 >> 3 )
          goto LABEL_5;
        v6 = v5;
LABEL_4:
        sub_15C40B0(a2, v6);
        sub_15B9210(a2, &v32, &v33);
        v7 = v33;
        v8 = v32;
        v9 = *(_DWORD *)(a2 + 16) + 1;
LABEL_5:
        *(_DWORD *)(a2 + 16) = v9;
        if ( *v7 != -8 )
          --*(_DWORD *)(a2 + 20);
        *v7 = v8;
        return v32;
      }
LABEL_3:
      v6 = 2 * v5;
      goto LABEL_4;
    }
LABEL_2:
    ++*(_QWORD *)a2;
    v5 = 0;
    goto LABEL_3;
  }
  return result;
}
