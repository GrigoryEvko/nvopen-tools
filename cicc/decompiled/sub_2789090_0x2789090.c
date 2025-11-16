// Function: sub_2789090
// Address: 0x2789090
//
__int64 __fastcall sub_2789090(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r15
  __int64 v6; // rbx
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  int v12; // eax
  __int64 v13; // rdx
  _QWORD *v14; // rax
  _QWORD *i; // rdx
  __int64 v16; // rax
  __int64 result; // rax
  unsigned int v18; // eax
  _QWORD *v19; // rdi
  int v20; // eax
  int v21; // ebx
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rax
  _QWORD *v24; // rax
  __int64 v25; // rdx
  _QWORD *j; // rdx
  _QWORD *v27; // rax
  unsigned __int8 v28; // [rsp+Fh] [rbp-31h]

  sub_2784A60(*(_QWORD *)(a1 + 176));
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = a1 + 168;
  *(_QWORD *)(a1 + 192) = a1 + 168;
  *(_QWORD *)(a1 + 200) = 0;
  sub_2784A60(0);
  sub_2785140(a1);
  v5 = *(_QWORD *)(a1 + 32);
  v6 = v5 + 40LL * *(unsigned int *)(a1 + 40);
  while ( v5 != v6 )
  {
    while ( 1 )
    {
      v6 -= 40;
      if ( *(_DWORD *)(v6 + 32) > 0x40u )
      {
        v7 = *(_QWORD *)(v6 + 24);
        if ( v7 )
          j_j___libc_free_0_0(v7);
      }
      if ( *(_DWORD *)(v6 + 16) <= 0x40u )
        break;
      v8 = *(_QWORD *)(v6 + 8);
      if ( !v8 )
        break;
      j_j___libc_free_0_0(v8);
      if ( v5 == v6 )
        goto LABEL_9;
    }
  }
LABEL_9:
  *(_DWORD *)(a1 + 40) = 0;
  sub_2785140(a1 + 208);
  v12 = *(_DWORD *)(a1 + 64);
  ++*(_QWORD *)(a1 + 48);
  *(_DWORD *)(a1 + 248) = 0;
  if ( !v12 )
  {
    if ( !*(_DWORD *)(a1 + 68) )
      goto LABEL_15;
    v13 = *(unsigned int *)(a1 + 72);
    if ( (unsigned int)v13 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 56), 8 * v13, 8);
      *(_QWORD *)(a1 + 56) = 0;
      *(_QWORD *)(a1 + 64) = 0;
      *(_DWORD *)(a1 + 72) = 0;
      goto LABEL_15;
    }
    goto LABEL_12;
  }
  v9 = (unsigned int)(4 * v12);
  v13 = *(unsigned int *)(a1 + 72);
  if ( (unsigned int)v9 < 0x40 )
    v9 = 64;
  if ( (unsigned int)v13 <= (unsigned int)v9 )
  {
LABEL_12:
    v14 = *(_QWORD **)(a1 + 56);
    for ( i = &v14[v13]; i != v14; ++v14 )
      *v14 = -4096;
    *(_QWORD *)(a1 + 64) = 0;
    goto LABEL_15;
  }
  v18 = v12 - 1;
  if ( !v18 )
  {
    v19 = *(_QWORD **)(a1 + 56);
    v21 = 64;
LABEL_24:
    sub_C7D6A0((__int64)v19, 8 * v13, 8);
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
    *(_DWORD *)(a1 + 72) = v23;
    v24 = (_QWORD *)sub_C7D670(8 * v23, 8);
    v25 = *(unsigned int *)(a1 + 72);
    *(_QWORD *)(a1 + 64) = 0;
    *(_QWORD *)(a1 + 56) = v24;
    for ( j = &v24[v25]; j != v24; ++v24 )
    {
      if ( v24 )
        *v24 = -4096;
    }
    goto LABEL_15;
  }
  _BitScanReverse(&v18, v18);
  v19 = *(_QWORD **)(a1 + 56);
  v20 = v18 ^ 0x1F;
  v9 = (unsigned int)(33 - v20);
  v21 = 1 << (33 - v20);
  if ( v21 < 64 )
    v21 = 64;
  if ( (_DWORD)v13 != v21 )
    goto LABEL_24;
  *(_QWORD *)(a1 + 64) = 0;
  v27 = &v19[v13];
  do
  {
    if ( v19 )
      *v19 = -4096;
    ++v19;
  }
  while ( v27 != v19 );
LABEL_15:
  *(_DWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 256) = **(_QWORD **)(a2 + 40);
  sub_2786690(a1, a2, a3, (_DWORD *)v9, v10, v11);
  sub_2788380(a1);
  sub_2787ED0(a1);
  v16 = sub_B2BEC0(a2);
  result = sub_2787550(a1, v16);
  if ( (_BYTE)result )
  {
    v28 = result;
    sub_2784F90(a1);
    return v28;
  }
  return result;
}
