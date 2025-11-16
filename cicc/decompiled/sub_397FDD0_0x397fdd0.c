// Function: sub_397FDD0
// Address: 0x397fdd0
//
void __fastcall sub_397FDD0(__int64 a1, __int64 *a2)
{
  int v3; // eax
  __int64 v4; // rdx
  _QWORD *v5; // rax
  _QWORD *i; // rdx
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // rbx
  unsigned __int64 v10; // rdi
  unsigned int v11; // ecx
  _QWORD *v12; // rdi
  unsigned int v13; // eax
  int v14; // eax
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  int v17; // ebx
  unsigned __int64 v18; // r13
  _QWORD *v19; // rax
  __int64 v20; // rdx
  _QWORD *j; // rdx
  __int64 v22; // rax
  _QWORD *v23; // rax

  if ( *(_BYTE *)(*(_QWORD *)(a1 + 16) + 1744LL) )
  {
    v22 = sub_1626D20(*a2);
    if ( v22 )
    {
      if ( *(_DWORD *)(*(_QWORD *)(v22 + 8 * (5LL - *(unsigned int *)(v22 + 8))) + 36LL) )
        (*(void (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 112LL))(a1, a2);
    }
  }
  v3 = *(_DWORD *)(a1 + 312);
  ++*(_QWORD *)(a1 + 296);
  if ( !v3 )
  {
    if ( !*(_DWORD *)(a1 + 316) )
      goto LABEL_8;
    v4 = *(unsigned int *)(a1 + 320);
    if ( (unsigned int)v4 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 304));
      *(_QWORD *)(a1 + 304) = 0;
      *(_QWORD *)(a1 + 312) = 0;
      *(_DWORD *)(a1 + 320) = 0;
      goto LABEL_8;
    }
    goto LABEL_5;
  }
  v11 = 4 * v3;
  v4 = *(unsigned int *)(a1 + 320);
  if ( (unsigned int)(4 * v3) < 0x40 )
    v11 = 64;
  if ( (unsigned int)v4 <= v11 )
  {
LABEL_5:
    v5 = *(_QWORD **)(a1 + 304);
    for ( i = &v5[3 * v4]; i != v5; *(v5 - 2) = -8 )
    {
      *v5 = -8;
      v5 += 3;
    }
    *(_QWORD *)(a1 + 312) = 0;
    goto LABEL_8;
  }
  v12 = *(_QWORD **)(a1 + 304);
  v13 = v3 - 1;
  if ( !v13 )
  {
    v18 = 3072;
    v17 = 128;
LABEL_23:
    j___libc_free_0((unsigned __int64)v12);
    *(_DWORD *)(a1 + 320) = v17;
    v19 = (_QWORD *)sub_22077B0(v18);
    v20 = *(unsigned int *)(a1 + 320);
    *(_QWORD *)(a1 + 312) = 0;
    *(_QWORD *)(a1 + 304) = v19;
    for ( j = &v19[3 * v20]; j != v19; v19 += 3 )
    {
      if ( v19 )
      {
        *v19 = -8;
        v19[1] = -8;
      }
    }
    goto LABEL_8;
  }
  _BitScanReverse(&v13, v13);
  v14 = 1 << (33 - (v13 ^ 0x1F));
  if ( v14 < 64 )
    v14 = 64;
  if ( (_DWORD)v4 != v14 )
  {
    v15 = (4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1);
    v16 = ((((v15 >> 2) | v15 | (((v15 >> 2) | v15) >> 4)) >> 8)
         | (v15 >> 2)
         | v15
         | (((v15 >> 2) | v15) >> 4)
         | (((((v15 >> 2) | v15 | (((v15 >> 2) | v15) >> 4)) >> 8) | (v15 >> 2) | v15 | (((v15 >> 2) | v15) >> 4)) >> 16))
        + 1;
    v17 = v16;
    v18 = 24 * v16;
    goto LABEL_23;
  }
  *(_QWORD *)(a1 + 312) = 0;
  v23 = &v12[3 * v4];
  do
  {
    if ( v12 )
    {
      *v12 = -8;
      v12[1] = -8;
    }
    v12 += 3;
  }
  while ( v23 != v12 );
LABEL_8:
  v7 = *(_QWORD *)(a1 + 328);
  v8 = *(_QWORD *)(a1 + 336);
  if ( v7 != v8 )
  {
    v9 = *(_QWORD *)(a1 + 328);
    do
    {
      v10 = *(_QWORD *)(v9 + 16);
      if ( v10 != v9 + 32 )
        _libc_free(v10);
      v9 += 96;
    }
    while ( v8 != v9 );
    *(_QWORD *)(a1 + 336) = v7;
  }
  sub_397FC20(a1 + 352);
  sub_397FC20(a1 + 384);
}
