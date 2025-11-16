// Function: sub_234E270
// Address: 0x234e270
//
__int64 __fastcall sub_234E270(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  __int64 v4; // rsi
  unsigned __int64 v5; // rdi
  _QWORD *v6; // r12
  __int64 v7; // rsi
  _QWORD *v8; // r13
  __int64 v9; // rdx
  unsigned int v10; // ecx
  __int64 v11; // rax
  unsigned __int64 v12; // r15
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // rdi
  _QWORD *v15; // r13
  _QWORD *i; // r15
  unsigned __int64 v17; // r12
  unsigned __int64 j; // rbx
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  __int64 v21; // rsi

  v2 = a1 + 656;
  v3 = *(_QWORD *)(a1 + 640);
  if ( v3 != v2 )
    _libc_free(v3);
  sub_C7D6A0(*(_QWORD *)(a1 + 616), 8LL * *(unsigned int *)(a1 + 632), 8);
  v4 = 16LL * *(unsigned int *)(a1 + 600);
  sub_C7D6A0(*(_QWORD *)(a1 + 584), v4, 8);
  v5 = *(_QWORD *)(a1 + 432);
  if ( v5 != a1 + 448 )
    _libc_free(v5);
  sub_234E0E0(a1 + 336);
  sub_B72320(a1 + 336, v4);
  sub_C7D6A0(*(_QWORD *)(a1 + 312), 16LL * *(unsigned int *)(a1 + 328), 8);
  v6 = *(_QWORD **)(a1 + 224);
  v7 = *(unsigned int *)(a1 + 232);
  v8 = &v6[v7];
  if ( v6 != v8 )
  {
    v9 = *(_QWORD *)(a1 + 224);
    while ( 1 )
    {
      v10 = (unsigned int)(((__int64)v6 - v9) >> 3) >> 7;
      v11 = 4096LL << v10;
      if ( v10 >= 0x1E )
        v11 = 0x40000000000LL;
      v12 = *v6 + v11;
      v13 = (*v6 + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( *v6 == *(_QWORD *)(v9 + 8 * v7 - 8) )
        v12 = *(_QWORD *)(a1 + 208);
      while ( 1 )
      {
        v13 += 32LL;
        if ( v12 < v13 )
          break;
        while ( 1 )
        {
          v14 = *(_QWORD *)(v13 - 24);
          if ( v14 == v13 - 8 )
            break;
          _libc_free(v14);
          v13 += 32LL;
          if ( v12 < v13 )
            goto LABEL_14;
        }
      }
LABEL_14:
      if ( v8 == ++v6 )
        break;
      v9 = *(_QWORD *)(a1 + 224);
      v7 = *(unsigned int *)(a1 + 232);
    }
  }
  v15 = *(_QWORD **)(a1 + 272);
  for ( i = &v15[2 * *(unsigned int *)(a1 + 280)]; i != v15; v15 += 2 )
  {
    v17 = *v15 + v15[1];
    for ( j = ((*v15 + 7LL) & 0xFFFFFFFFFFFFFFF8LL) + 32; v17 >= j; j += 32LL )
    {
      v19 = *(_QWORD *)(j - 24);
      if ( v19 != j - 8 )
        _libc_free(v19);
    }
  }
  sub_E66D20(a1 + 208);
  sub_B72320(a1 + 208, v7);
  sub_C7D6A0(*(_QWORD *)(a1 + 184), 16LL * *(unsigned int *)(a1 + 200), 8);
  v20 = *(_QWORD *)(a1 + 128);
  if ( v20 != a1 + 144 )
    _libc_free(v20);
  v21 = 16LL * *(unsigned int *)(a1 + 120);
  sub_C7D6A0(*(_QWORD *)(a1 + 104), v21, 8);
  sub_234DF60(a1);
  return sub_B72320(a1, v21);
}
