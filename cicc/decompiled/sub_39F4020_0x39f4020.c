// Function: sub_39F4020
// Address: 0x39f4020
//
__int64 __fastcall sub_39F4020(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  unsigned int v7; // ecx
  _QWORD *v8; // rdi
  unsigned int v9; // eax
  int v10; // eax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  int v13; // ebx
  unsigned __int64 v14; // r13
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _QWORD *j; // rdx
  _QWORD *v18; // rax

  v2 = *(_DWORD *)(a1 + 344);
  ++*(_QWORD *)(a1 + 328);
  *(_BYTE *)(a1 + 322) = 0;
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 348) )
      return sub_38D6240(a1);
    v3 = *(unsigned int *)(a1 + 352);
    if ( (unsigned int)v3 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 336));
      *(_QWORD *)(a1 + 336) = 0;
      *(_QWORD *)(a1 + 344) = 0;
      *(_DWORD *)(a1 + 352) = 0;
      return sub_38D6240(a1);
    }
    goto LABEL_4;
  }
  v7 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 352);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v7 = 64;
  if ( v7 >= (unsigned int)v3 )
  {
LABEL_4:
    v4 = *(_QWORD **)(a1 + 336);
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -8;
    *(_QWORD *)(a1 + 344) = 0;
    return sub_38D6240(a1);
  }
  v8 = *(_QWORD **)(a1 + 336);
  v9 = v2 - 1;
  if ( !v9 )
  {
    v14 = 2048;
    v13 = 128;
LABEL_16:
    j___libc_free_0((unsigned __int64)v8);
    *(_DWORD *)(a1 + 352) = v13;
    v15 = (_QWORD *)sub_22077B0(v14);
    v16 = *(unsigned int *)(a1 + 352);
    *(_QWORD *)(a1 + 344) = 0;
    *(_QWORD *)(a1 + 336) = v15;
    for ( j = &v15[2 * v16]; j != v15; v15 += 2 )
    {
      if ( v15 )
        *v15 = -8;
    }
    return sub_38D6240(a1);
  }
  _BitScanReverse(&v9, v9);
  v10 = 1 << (33 - (v9 ^ 0x1F));
  if ( v10 < 64 )
    v10 = 64;
  if ( (_DWORD)v3 != v10 )
  {
    v11 = (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
        | (4 * v10 / 3u + 1)
        | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)
        | (((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
          | (4 * v10 / 3u + 1)
          | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4);
    v12 = (v11 >> 8) | v11;
    v13 = (v12 | (v12 >> 16)) + 1;
    v14 = 16 * ((v12 | (v12 >> 16)) + 1);
    goto LABEL_16;
  }
  *(_QWORD *)(a1 + 344) = 0;
  v18 = &v8[2 * (unsigned int)v3];
  do
  {
    if ( v8 )
      *v8 = -8;
    v8 += 2;
  }
  while ( v18 != v8 );
  return sub_38D6240(a1);
}
