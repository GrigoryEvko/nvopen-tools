// Function: sub_1BC21B0
// Address: 0x1bc21b0
//
void __fastcall sub_1BC21B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  int v7; // r13d
  __int64 v8; // rbx
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // r12
  unsigned int v12; // eax
  __int64 i; // rax
  __int64 v14; // rdx
  size_t v15; // rdx
  _DWORD *v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rbx
  unsigned int v19; // r13d
  unsigned int v20; // eax
  __int64 v21; // r12
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // rcx
  int v25; // r8d
  int v26; // r9d
  __int64 v27; // rbx
  void *s2; // [rsp+0h] [rbp-70h] BYREF
  __int64 v29; // [rsp+8h] [rbp-68h]
  _DWORD v30[4]; // [rsp+10h] [rbp-60h] BYREF
  _DWORD *v31; // [rsp+20h] [rbp-50h] BYREF
  __int64 v32; // [rsp+28h] [rbp-48h]
  _DWORD v33[16]; // [rsp+30h] [rbp-40h] BYREF

  v7 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v7 && !*(_DWORD *)(a1 + 20) )
    return;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = 64;
  v10 = *(unsigned int *)(a1 + 24);
  v11 = v8 + 40 * v10;
  v12 = 4 * v7;
  if ( (unsigned int)(4 * v7) < 0x40 )
    v12 = 64;
  if ( (unsigned int)v10 <= v12 )
  {
    v30[0] = -2;
    s2 = v30;
    v29 = 0x400000001LL;
    v31 = v33;
    v32 = 0x400000001LL;
    v33[0] = -3;
    if ( v8 != v11 )
    {
      for ( i = 1; ; i = (unsigned int)v29 )
      {
        v14 = *(unsigned int *)(v8 + 8);
        if ( v14 != i )
          break;
        v15 = 4 * v14;
        if ( !v15 )
          goto LABEL_9;
        if ( memcmp(*(const void **)v8, s2, v15) )
          break;
        v8 += 40;
        if ( v8 == v11 )
        {
LABEL_15:
          v16 = s2;
          *(_QWORD *)(a1 + 16) = 0;
          if ( v16 != v30 )
            goto LABEL_16;
          return;
        }
LABEL_10:
        ;
      }
      sub_1BB9EE0(v8, (__int64)&s2, v14, v9, a5, a6);
LABEL_9:
      v8 += 40;
      if ( v8 == v11 )
        goto LABEL_15;
      goto LABEL_10;
    }
LABEL_25:
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  do
  {
    if ( *(_QWORD *)v8 != v8 + 16 )
      _libc_free(*(_QWORD *)v8);
    v8 += 40;
  }
  while ( v8 != v11 );
  v17 = *(unsigned int *)(a1 + 24);
  if ( !v7 )
  {
    if ( !(_DWORD)v17 )
      goto LABEL_25;
    j___libc_free_0(*(_QWORD *)(a1 + 8));
    *(_DWORD *)(a1 + 24) = 0;
LABEL_24:
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  v18 = 64;
  v19 = v7 - 1;
  if ( v19 )
  {
    _BitScanReverse(&v20, v19);
    v9 = 33 - (v20 ^ 0x1F);
    v18 = (unsigned int)(1 << (33 - (v20 ^ 0x1F)));
    if ( (int)v18 < 64 )
      v18 = 64;
  }
  v21 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)v18 != (_DWORD)v17 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 8));
    v22 = sub_1454B60(4 * (int)v18 / 3u + 1);
    *(_DWORD *)(a1 + 24) = v22;
    if ( v22 )
    {
      *(_QWORD *)(a1 + 8) = sub_22077B0(40LL * v22);
      sub_1BC2100(a1, a2, v23, v24, v25, v26);
      return;
    }
    goto LABEL_24;
  }
  *(_QWORD *)(a1 + 16) = 0;
  v32 = 0x400000001LL;
  v31 = v33;
  v27 = v21 + 40 * v18;
  v33[0] = -2;
  do
  {
    if ( v21 )
    {
      *(_DWORD *)(v21 + 8) = 0;
      *(_QWORD *)v21 = v21 + 16;
      *(_DWORD *)(v21 + 12) = 4;
      if ( (_DWORD)v32 )
        sub_1BB9EE0(v21, (__int64)&v31, v17, v9, a5, a6);
    }
    v21 += 40;
  }
  while ( v27 != v21 );
  v16 = v31;
  if ( v31 != v33 )
LABEL_16:
    _libc_free((unsigned __int64)v16);
}
