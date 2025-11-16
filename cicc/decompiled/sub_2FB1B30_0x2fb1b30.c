// Function: sub_2FB1B30
// Address: 0x2fb1b30
//
__int64 __fastcall sub_2FB1B30(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v4; // r9
  __int64 *v5; // rcx
  __int64 v6; // rsi
  __int64 *v7; // r10
  __int64 v8; // r11
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  unsigned int v12; // r8d
  __int64 v13; // r11
  __int64 v14; // r9
  __int64 v15; // rdi
  unsigned __int64 v16; // r9
  unsigned int v17; // edi
  int v18; // r12d
  unsigned int v19; // r11d
  __int64 v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rbx
  unsigned int v23; // edi
  __int64 *v24; // rsi

  v2 = *(unsigned int *)(a2 + 8);
  if ( !(_DWORD)v2 )
    return 0;
  v4 = 3 * v2;
  v5 = *(__int64 **)a2;
  v6 = **(_QWORD **)a2;
  v7 = &v5[3 * v2];
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL);
  v9 = *(_QWORD *)((v6 & 0xFFFFFFFFFFFFFFF8LL) + 16);
  if ( v9 )
  {
    v10 = *(_QWORD *)(v9 + 24);
  }
  else
  {
    v21 = *(unsigned int *)(v8 + 304);
    v22 = *(_QWORD **)(v8 + 296);
    if ( *(_DWORD *)(v8 + 304) )
    {
      v23 = *(_DWORD *)((v6 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v6 >> 1) & 3;
      do
      {
        v24 = &v22[2 * (v21 >> 1)];
        if ( v23 >= (*(_DWORD *)((*v24 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v24 >> 1) & 3) )
        {
          v22 = v24 + 2;
          v21 = v21 - (v21 >> 1) - 1;
        }
        else
        {
          v21 >>= 1;
        }
      }
      while ( v21 > 0 );
    }
    v10 = *(v22 - 1);
  }
  v11 = *(_QWORD *)(v8 + 152);
  v12 = 0;
  v13 = *(_QWORD *)(v11 + 16LL * *(unsigned int *)(v10 + 24) + 8);
  v14 = v5[v4 - 2];
  v15 = v14 >> 1;
  v16 = v14 & 0xFFFFFFFFFFFFFFF8LL;
  v17 = v15 & 3;
  v18 = *(_DWORD *)((v13 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  while ( 1 )
  {
    ++v12;
    v19 = v18 | (v13 >> 1) & 3;
    if ( v19 >= (v17 | *(_DWORD *)(v16 + 24)) )
      break;
    if ( (*(_DWORD *)((v5[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v5[1] >> 1) & 3) <= v19 )
    {
      do
      {
        v20 = v5[4];
        v5 += 3;
      }
      while ( v19 >= (*(_DWORD *)((v20 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v20 >> 1) & 3) );
    }
    if ( v7 == v5 )
      break;
    do
    {
      v10 = *(_QWORD *)(v10 + 8);
      v13 = *(_QWORD *)(v11 + 16LL * *(unsigned int *)(v10 + 24) + 8);
      v18 = *(_DWORD *)((v13 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    }
    while ( (v18 | (unsigned int)(v13 >> 1) & 3) <= (*(_DWORD *)((*v5 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                   | (unsigned int)(*v5 >> 1) & 3) );
  }
  return v12;
}
