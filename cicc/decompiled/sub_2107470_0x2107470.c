// Function: sub_2107470
// Address: 0x2107470
//
unsigned __int64 __fastcall sub_2107470(__int64 *a1, int a2)
{
  __int64 v3; // r12
  int v4; // eax
  unsigned int v5; // ecx
  __int64 v6; // rdx
  _QWORD *v7; // rax
  _QWORD *i; // rdx
  __int64 v9; // rax
  unsigned __int64 result; // rax
  __int64 v11; // rax
  _QWORD *v12; // rdi
  unsigned int v13; // eax
  int v14; // eax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  int v17; // r14d
  __int64 v18; // r15
  _QWORD *v19; // rax
  __int64 v20; // rdx
  _QWORD *j; // rdx
  _QWORD *v22; // rax

  v3 = *a1;
  if ( *a1 )
  {
    v4 = *(_DWORD *)(v3 + 16);
    ++*(_QWORD *)v3;
    if ( !v4 )
    {
      if ( !*(_DWORD *)(v3 + 20) )
        goto LABEL_9;
      v6 = *(unsigned int *)(v3 + 24);
      if ( (unsigned int)v6 > 0x40 )
      {
        j___libc_free_0(*(_QWORD *)(v3 + 8));
        *(_QWORD *)(v3 + 8) = 0;
        *(_QWORD *)(v3 + 16) = 0;
        *(_DWORD *)(v3 + 24) = 0;
        goto LABEL_9;
      }
      goto LABEL_6;
    }
    v5 = 4 * v4;
    v6 = *(unsigned int *)(v3 + 24);
    if ( (unsigned int)(4 * v4) < 0x40 )
      v5 = 64;
    if ( v5 >= (unsigned int)v6 )
    {
LABEL_6:
      v7 = *(_QWORD **)(v3 + 8);
      for ( i = &v7[2 * v6]; i != v7; v7 += 2 )
        *v7 = -8;
      *(_QWORD *)(v3 + 16) = 0;
      goto LABEL_9;
    }
    v12 = *(_QWORD **)(v3 + 8);
    v13 = v4 - 1;
    if ( !v13 )
    {
      v18 = 2048;
      v17 = 128;
LABEL_21:
      j___libc_free_0(v12);
      *(_DWORD *)(v3 + 24) = v17;
      v19 = (_QWORD *)sub_22077B0(v18);
      v20 = *(unsigned int *)(v3 + 24);
      *(_QWORD *)(v3 + 16) = 0;
      *(_QWORD *)(v3 + 8) = v19;
      for ( j = &v19[2 * v20]; j != v19; v19 += 2 )
      {
        if ( v19 )
          *v19 = -8;
      }
      goto LABEL_9;
    }
    _BitScanReverse(&v13, v13);
    v14 = 1 << (33 - (v13 ^ 0x1F));
    if ( v14 < 64 )
      v14 = 64;
    if ( (_DWORD)v6 != v14 )
    {
      v15 = (((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
          | (4 * v14 / 3u + 1)
          | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)
          | (((((4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 2)
            | (4 * v14 / 3u + 1)
            | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1)) >> 4);
      v16 = (v15 >> 8) | v15;
      v17 = (v16 | (v16 >> 16)) + 1;
      v18 = 16 * ((v16 | (v16 >> 16)) + 1);
      goto LABEL_21;
    }
    *(_QWORD *)(v3 + 16) = 0;
    v22 = &v12[2 * (unsigned int)v6];
    do
    {
      if ( v12 )
        *v12 = -8;
      v12 += 2;
    }
    while ( v22 != v12 );
  }
  else
  {
    v11 = sub_22077B0(32);
    if ( v11 )
    {
      *(_QWORD *)v11 = 0;
      *(_QWORD *)(v11 + 8) = 0;
      *(_QWORD *)(v11 + 16) = 0;
      *(_DWORD *)(v11 + 24) = 0;
    }
    *a1 = v11;
  }
LABEL_9:
  v9 = a1[5];
  *((_DWORD *)a1 + 2) = a2;
  result = *(_QWORD *)(*(_QWORD *)(v9 + 24) + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  a1[2] = result;
  return result;
}
