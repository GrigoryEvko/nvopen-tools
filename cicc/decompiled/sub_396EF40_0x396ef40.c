// Function: sub_396EF40
// Address: 0x396ef40
//
void __fastcall sub_396EF40(__int64 a1)
{
  int v2; // r8d
  int v3; // r9d
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rax
  int v8; // eax
  unsigned int v9; // ecx
  __int64 v10; // rdx
  _QWORD *v11; // rax
  _QWORD *k; // rdx
  __int64 v13; // rax
  __int64 *v14; // rbx
  __int64 *v15; // r14
  __int64 v16; // rsi
  _QWORD *v17; // rdi
  unsigned int v18; // eax
  int v19; // eax
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  int v22; // ebx
  unsigned __int64 v23; // r14
  _QWORD *v24; // rax
  __int64 v25; // rdx
  _QWORD *j; // rdx
  _QWORD *v27; // rax
  __int64 *v28; // [rsp+10h] [rbp-80h] BYREF
  __int64 i; // [rsp+18h] [rbp-78h]
  _BYTE v30[112]; // [rsp+20h] [rbp-70h] BYREF

  if ( !*(_BYTE *)(sub_396DD80(a1) + 776) )
    return;
  v4 = *(_QWORD *)(a1 + 352);
  v5 = *(_QWORD *)(a1 + 360);
  v28 = (__int64 *)v30;
  for ( i = 0x800000000LL; v5 != v4; LODWORD(i) = i + 1 )
  {
    while ( !*(_DWORD *)(v4 + 16) )
    {
      v4 += 24;
      if ( v5 == v4 )
        goto LABEL_9;
    }
    v6 = *(_QWORD *)(v4 + 8);
    v7 = (unsigned int)i;
    if ( (unsigned int)i >= HIDWORD(i) )
    {
      sub_16CD150((__int64)&v28, v30, 0, 8, v2, v3);
      v7 = (unsigned int)i;
    }
    v4 += 24;
    v28[v7] = v6;
  }
LABEL_9:
  v8 = *(_DWORD *)(a1 + 336);
  ++*(_QWORD *)(a1 + 320);
  if ( v8 )
  {
    v9 = 4 * v8;
    v10 = *(unsigned int *)(a1 + 344);
    if ( (unsigned int)(4 * v8) < 0x40 )
      v9 = 64;
    if ( v9 >= (unsigned int)v10 )
      goto LABEL_13;
    v17 = *(_QWORD **)(a1 + 328);
    v18 = v8 - 1;
    if ( v18 )
    {
      _BitScanReverse(&v18, v18);
      v19 = 1 << (33 - (v18 ^ 0x1F));
      if ( v19 < 64 )
        v19 = 64;
      if ( (_DWORD)v10 == v19 )
      {
        *(_QWORD *)(a1 + 336) = 0;
        v27 = &v17[2 * (unsigned int)v10];
        do
        {
          if ( v17 )
            *v17 = -8;
          v17 += 2;
        }
        while ( v27 != v17 );
        goto LABEL_16;
      }
      v20 = (((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
          | (4 * v19 / 3u + 1)
          | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)
          | (((((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
            | (4 * v19 / 3u + 1)
            | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 4);
      v21 = (v20 >> 8) | v20;
      v22 = (v21 | (v21 >> 16)) + 1;
      v23 = 16 * ((v21 | (v21 >> 16)) + 1);
    }
    else
    {
      v23 = 2048;
      v22 = 128;
    }
    j___libc_free_0((unsigned __int64)v17);
    *(_DWORD *)(a1 + 344) = v22;
    v24 = (_QWORD *)sub_22077B0(v23);
    v25 = *(unsigned int *)(a1 + 344);
    *(_QWORD *)(a1 + 336) = 0;
    *(_QWORD *)(a1 + 328) = v24;
    for ( j = &v24[2 * v25]; j != v24; v24 += 2 )
    {
      if ( v24 )
        *v24 = -8;
    }
  }
  else if ( *(_DWORD *)(a1 + 340) )
  {
    v10 = *(unsigned int *)(a1 + 344);
    if ( (unsigned int)v10 <= 0x40 )
    {
LABEL_13:
      v11 = *(_QWORD **)(a1 + 328);
      for ( k = &v11[2 * v10]; k != v11; v11 += 2 )
        *v11 = -8;
      *(_QWORD *)(a1 + 336) = 0;
      goto LABEL_16;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 328));
    *(_QWORD *)(a1 + 328) = 0;
    *(_QWORD *)(a1 + 336) = 0;
    *(_DWORD *)(a1 + 344) = 0;
  }
LABEL_16:
  v13 = *(_QWORD *)(a1 + 352);
  if ( v13 != *(_QWORD *)(a1 + 360) )
    *(_QWORD *)(a1 + 360) = v13;
  v14 = v28;
  v15 = &v28[(unsigned int)i];
  if ( v28 != v15 )
  {
    do
    {
      v16 = *v14++;
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 208LL))(a1, v16);
    }
    while ( v15 != v14 );
    v15 = v28;
  }
  if ( v15 != (__int64 *)v30 )
    _libc_free((unsigned __int64)v15);
}
