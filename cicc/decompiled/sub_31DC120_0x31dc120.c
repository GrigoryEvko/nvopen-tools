// Function: sub_31DC120
// Address: 0x31dc120
//
void __fastcall sub_31DC120(__int64 a1)
{
  __int64 v2; // r8
  __int64 v3; // r9
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 i; // r14
  __int64 v7; // rax
  __int64 v8; // r15
  unsigned __int64 v9; // rdx
  int v10; // eax
  unsigned int v11; // ecx
  __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *k; // rdx
  __int64 *v15; // rbx
  __int64 v16; // rax
  __int64 *v17; // r14
  __int64 v18; // rsi
  unsigned int v19; // eax
  _QWORD *v20; // rdi
  int v21; // ebx
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rdi
  _QWORD *v24; // rax
  __int64 v25; // rdx
  _QWORD *j; // rdx
  _QWORD *v27; // rax
  __int64 *v28; // [rsp+10h] [rbp-80h] BYREF
  __int64 v29; // [rsp+18h] [rbp-78h]
  _BYTE v30[112]; // [rsp+20h] [rbp-70h] BYREF

  if ( !*(_BYTE *)(sub_31DA6B0(a1) + 936) )
    return;
  v4 = *(_QWORD *)(a1 + 384);
  v29 = 0x800000000LL;
  v5 = *(unsigned int *)(a1 + 392);
  v28 = (__int64 *)v30;
  for ( i = v4 + 24 * v5; i != v4; LODWORD(v29) = v29 + 1 )
  {
    while ( !*(_DWORD *)(v4 + 16) )
    {
      v4 += 24;
      if ( i == v4 )
        goto LABEL_9;
    }
    v7 = (unsigned int)v29;
    v8 = *(_QWORD *)(v4 + 8);
    v9 = (unsigned int)v29 + 1LL;
    if ( v9 > HIDWORD(v29) )
    {
      sub_C8D5F0((__int64)&v28, v30, v9, 8u, v2, v3);
      v7 = (unsigned int)v29;
    }
    v4 += 24;
    v28[v7] = v8;
  }
LABEL_9:
  v10 = *(_DWORD *)(a1 + 368);
  ++*(_QWORD *)(a1 + 352);
  if ( v10 )
  {
    v11 = 4 * v10;
    v12 = *(unsigned int *)(a1 + 376);
    if ( (unsigned int)(4 * v10) < 0x40 )
      v11 = 64;
    if ( (unsigned int)v12 <= v11 )
      goto LABEL_13;
    v19 = v10 - 1;
    if ( v19 )
    {
      _BitScanReverse(&v19, v19);
      v20 = *(_QWORD **)(a1 + 360);
      v21 = 1 << (33 - (v19 ^ 0x1F));
      if ( v21 < 64 )
        v21 = 64;
      if ( (_DWORD)v12 == v21 )
      {
        *(_QWORD *)(a1 + 368) = 0;
        v27 = &v20[2 * (unsigned int)v12];
        do
        {
          if ( v20 )
            *v20 = -4096;
          v20 += 2;
        }
        while ( v27 != v20 );
        goto LABEL_16;
      }
    }
    else
    {
      v20 = *(_QWORD **)(a1 + 360);
      v21 = 64;
    }
    sub_C7D6A0((__int64)v20, 16LL * (unsigned int)v12, 8);
    v22 = ((((((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
             | (4 * v21 / 3u + 1)
             | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
           | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
           | (4 * v21 / 3u + 1)
           | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
           | (4 * v21 / 3u + 1)
           | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
         | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
         | (4 * v21 / 3u + 1)
         | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 16;
    v23 = (v22
         | (((((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
             | (4 * v21 / 3u + 1)
             | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
           | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
           | (4 * v21 / 3u + 1)
           | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
           | (4 * v21 / 3u + 1)
           | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
         | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
         | (4 * v21 / 3u + 1)
         | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 376) = v23;
    v24 = (_QWORD *)sub_C7D670(16 * v23, 8);
    v25 = *(unsigned int *)(a1 + 376);
    *(_QWORD *)(a1 + 368) = 0;
    *(_QWORD *)(a1 + 360) = v24;
    for ( j = &v24[2 * v25]; j != v24; v24 += 2 )
    {
      if ( v24 )
        *v24 = -4096;
    }
  }
  else if ( *(_DWORD *)(a1 + 372) )
  {
    v12 = *(unsigned int *)(a1 + 376);
    if ( (unsigned int)v12 <= 0x40 )
    {
LABEL_13:
      v13 = *(_QWORD **)(a1 + 360);
      for ( k = &v13[2 * v12]; k != v13; v13 += 2 )
        *v13 = -4096;
      *(_QWORD *)(a1 + 368) = 0;
      goto LABEL_16;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 360), 16LL * (unsigned int)v12, 8);
    *(_QWORD *)(a1 + 360) = 0;
    *(_QWORD *)(a1 + 368) = 0;
    *(_DWORD *)(a1 + 376) = 0;
  }
LABEL_16:
  v15 = v28;
  v16 = (unsigned int)v29;
  *(_DWORD *)(a1 + 392) = 0;
  v17 = &v15[v16];
  if ( v15 != v17 )
  {
    do
    {
      v18 = *v15++;
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 224LL))(a1, v18);
    }
    while ( v17 != v15 );
    v17 = v28;
  }
  if ( v17 != (__int64 *)v30 )
    _libc_free((unsigned __int64)v17);
}
