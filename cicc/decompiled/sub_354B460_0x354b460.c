// Function: sub_354B460
// Address: 0x354b460
//
int *__fastcall sub_354B460(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  _QWORD *v8; // rdi
  __int64 v10; // r14
  __int64 v11; // r14
  unsigned int v12; // r15d
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r14
  unsigned __int64 v16; // rax
  __int64 i; // rdx
  __int64 v18; // rcx
  __int64 v19; // rdx
  unsigned __int64 v20; // r14
  __int64 v21; // rax
  unsigned __int64 j; // rdx
  __int64 v23; // r15
  char **v24; // rax
  __int64 v25; // r14
  char **v26; // r13
  char *v27; // r15
  unsigned __int64 v28; // r14
  char *v29; // rax
  int *result; // rax
  int *v31; // r8
  int v32; // ecx
  unsigned __int64 v33; // rsi
  unsigned __int64 v34; // rdx

  v6 = a1 + 56;
  v8 = (_QWORD *)(a1 + 72);
  *(v8 - 8) = 0;
  *(v8 - 4) = v6;
  *(v8 - 3) = 0;
  v10 = a2[1] - *a2;
  *(v8 - 9) = a2;
  v11 = v10 >> 8;
  *(v8 - 7) = 0;
  *(v8 - 6) = 0;
  v12 = (unsigned int)(v11 + 63) >> 6;
  *((_DWORD *)v8 - 10) = 0;
  *(_QWORD *)(a1 + 56) = v8;
  *(_QWORD *)(a1 + 64) = 0x600000000LL;
  if ( v12 > 6 )
  {
    sub_C8D5F0(v6, v8, v12, 8u, v6, a6);
    memset(*(void **)(a1 + 56), 0, 8LL * v12);
    *(_DWORD *)(a1 + 64) = v12;
  }
  else
  {
    if ( v12 )
      memset(v8, 0, (size_t)&v8[v12 - 9] - a1);
    *(_DWORD *)(a1 + 64) = v12;
  }
  *(_DWORD *)(a1 + 120) = v11;
  v13 = a1 + 144;
  v14 = a2[1] - *a2;
  *(_QWORD *)(a1 + 128) = a1 + 144;
  *(_QWORD *)(a1 + 136) = 0xA00000000LL;
  v15 = v14 >> 8;
  if ( v14 >> 8 )
  {
    v16 = a1 + 144;
    if ( (unsigned __int64)v14 > 0xA00 )
    {
      sub_354B380(a1 + 128, v14 >> 8, v13, v14, v6, a6);
      v13 = *(_QWORD *)(a1 + 128);
      v16 = v13 + ((unsigned __int64)*(unsigned int *)(a1 + 136) << 6);
    }
    for ( i = (v15 << 6) + v13; i != v16; v16 += 64LL )
    {
      if ( v16 )
      {
        *(_QWORD *)v16 = 0;
        *(_QWORD *)(v16 + 8) = v16 + 32;
        *(_DWORD *)(v16 + 16) = 4;
        *(_DWORD *)(v16 + 20) = 0;
        *(_DWORD *)(v16 + 24) = 0;
        *(_BYTE *)(v16 + 28) = 1;
      }
    }
    *(_DWORD *)(a1 + 136) = v15;
  }
  v18 = a2[1] - *a2;
  v19 = a1 + 800;
  *(_QWORD *)(a1 + 784) = a1 + 800;
  v20 = v18 >> 8;
  *(_QWORD *)(a1 + 792) = 0x1000000000LL;
  if ( v18 >> 8 )
  {
    v21 = a1 + 800;
    if ( (unsigned __int64)v18 > 0x1000 )
    {
      sub_2FD0D40(a1 + 784, v20, v19, v18, v6, a6);
      v19 = *(_QWORD *)(a1 + 784);
      v21 = v19 + 32LL * *(unsigned int *)(a1 + 792);
    }
    for ( j = 32 * v20 + v19; j != v21; v21 += 32 )
    {
      if ( v21 )
      {
        *(_DWORD *)(v21 + 8) = 0;
        *(_QWORD *)v21 = v21 + 16;
        *(_DWORD *)(v21 + 12) = 4;
      }
    }
    *(_DWORD *)(a1 + 792) = v20;
  }
  *(_DWORD *)(a1 + 1320) = 0;
  v23 = a2[1] - *a2;
  v24 = (char **)sub_22077B0(0x18u);
  v25 = v23 >> 8;
  v26 = v24;
  if ( v24 )
  {
    if ( v23 < 0 )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    *v24 = 0;
    v27 = 0;
    v24[1] = 0;
    v24[2] = 0;
    if ( v25 )
    {
      v28 = 4 * v25;
      v29 = (char *)sub_22077B0(v28);
      v27 = &v29[v28];
      *v26 = v29;
      v26[2] = &v29[v28];
      if ( &v29[v28] != v29 )
        memset(v29, 0, v28);
    }
    v26[1] = v27;
  }
  *(_QWORD *)(a1 + 1312) = v26;
  result = *(int **)(a3 + 296);
  v31 = *(int **)(a3 + 304);
  if ( v31 != result )
  {
    v32 = 0;
    while ( 1 )
    {
      v33 = *result;
      v34 = (v26[1] - *v26) >> 2;
      if ( v33 >= v34 )
        sub_222CF80("vector::_M_range_check: __n (which is %zu) >= this->size() (which is %zu)", v33, v34);
      ++result;
      *(_DWORD *)&(*v26)[4 * v33] = v32++;
      if ( v31 == result )
        break;
      v26 = *(char ***)(a1 + 1312);
    }
  }
  return result;
}
