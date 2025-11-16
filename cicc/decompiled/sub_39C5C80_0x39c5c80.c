// Function: sub_39C5C80
// Address: 0x39c5c80
//
void __fastcall sub_39C5C80(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  unsigned int v6; // ecx
  _QWORD *v7; // rdi
  unsigned int v8; // eax
  int v9; // eax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  int v12; // r13d
  unsigned __int64 v13; // r12
  _QWORD *v14; // rax
  __int64 v15; // rdx
  _QWORD *j; // rdx
  _QWORD *v17; // rax

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      return;
    v3 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v3 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return;
    }
    goto LABEL_4;
  }
  v6 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v6 = 64;
  if ( v6 >= (unsigned int)v3 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 8);
    for ( i = &v4[2 * v3]; v4 != i; v4 += 2 )
      *v4 = -8;
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  v7 = *(_QWORD **)(a1 + 8);
  v8 = v2 - 1;
  if ( !v8 )
  {
    v13 = 2048;
    v12 = 128;
LABEL_16:
    j___libc_free_0((unsigned __int64)v7);
    *(_DWORD *)(a1 + 24) = v12;
    v14 = (_QWORD *)sub_22077B0(v13);
    v15 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v14;
    for ( j = &v14[2 * v15]; j != v14; v14 += 2 )
    {
      if ( v14 )
        *v14 = -8;
    }
    return;
  }
  _BitScanReverse(&v8, v8);
  v9 = 1 << (33 - (v8 ^ 0x1F));
  if ( v9 < 64 )
    v9 = 64;
  if ( (_DWORD)v3 != v9 )
  {
    v10 = (((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
        | (4 * v9 / 3u + 1)
        | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)
        | (((((4 * v9 / 3u + 1) | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 2)
          | (4 * v9 / 3u + 1)
          | ((unsigned __int64)(4 * v9 / 3u + 1) >> 1)) >> 4);
    v11 = (v10 >> 8) | v10;
    v12 = (v11 | (v11 >> 16)) + 1;
    v13 = 16 * ((v11 | (v11 >> 16)) + 1);
    goto LABEL_16;
  }
  *(_QWORD *)(a1 + 16) = 0;
  v17 = &v7[2 * (unsigned int)v3];
  do
  {
    if ( v7 )
      *v7 = -8;
    v7 += 2;
  }
  while ( v17 != v7 );
}
