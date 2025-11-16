// Function: sub_136F890
// Address: 0x136f890
//
__int64 __fastcall sub_136F890(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *i; // rdx
  unsigned int v11; // ecx
  _QWORD *v12; // rdi
  unsigned int v13; // eax
  int v14; // eax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  int v17; // ebx
  __int64 v18; // r13
  _QWORD *v19; // rax
  __int64 v20; // rdx
  _QWORD *j; // rdx
  _QWORD *v22; // rax

  *(_QWORD *)(a1 + 112) = a3;
  *(_QWORD *)(a1 + 120) = a4;
  *(_QWORD *)(a1 + 128) = a2;
  sub_1371990();
  v5 = *(_QWORD *)(a1 + 136);
  if ( v5 != *(_QWORD *)(a1 + 144) )
    *(_QWORD *)(a1 + 144) = v5;
  v6 = *(_DWORD *)(a1 + 176);
  ++*(_QWORD *)(a1 + 160);
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 180) )
      goto LABEL_9;
    v7 = *(unsigned int *)(a1 + 184);
    if ( (unsigned int)v7 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 168));
      *(_QWORD *)(a1 + 168) = 0;
      *(_QWORD *)(a1 + 176) = 0;
      *(_DWORD *)(a1 + 184) = 0;
      goto LABEL_9;
    }
    goto LABEL_6;
  }
  v11 = 4 * v6;
  v7 = *(unsigned int *)(a1 + 184);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v11 = 64;
  if ( v11 >= (unsigned int)v7 )
  {
LABEL_6:
    v8 = *(_QWORD **)(a1 + 168);
    for ( i = &v8[2 * v7]; i != v8; v8 += 2 )
      *v8 = -8;
    *(_QWORD *)(a1 + 176) = 0;
    goto LABEL_9;
  }
  v12 = *(_QWORD **)(a1 + 168);
  v13 = v6 - 1;
  if ( !v13 )
  {
    v18 = 2048;
    v17 = 128;
LABEL_20:
    j___libc_free_0(v12);
    *(_DWORD *)(a1 + 184) = v17;
    v19 = (_QWORD *)sub_22077B0(v18);
    v20 = *(unsigned int *)(a1 + 184);
    *(_QWORD *)(a1 + 176) = 0;
    *(_QWORD *)(a1 + 168) = v19;
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
  if ( (_DWORD)v7 != v14 )
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
    goto LABEL_20;
  }
  *(_QWORD *)(a1 + 176) = 0;
  v22 = &v12[2 * (unsigned int)v7];
  do
  {
    if ( v12 )
      *v12 = -8;
    v12 += 2;
  }
  while ( v22 != v12 );
LABEL_9:
  sub_136E2F0(a1);
  sub_136A800(a1);
  sub_136F810(a1);
  if ( !(unsigned __int8)sub_136A140((_QWORD *)a1) )
  {
    sub_136F5F0(a1, 0, *(_QWORD *)(a1 + 88));
    sub_136A140((_QWORD *)a1);
  }
  sub_1371EC0(a1);
  return sub_13720B0(a1);
}
