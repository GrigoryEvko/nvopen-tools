// Function: sub_1D17100
// Address: 0x1d17100
//
void __fastcall sub_1D17100(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  _QWORD *v6; // r15
  unsigned __int64 v7; // r13
  unsigned int v8; // eax
  unsigned __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned __int64 *v11; // rcx
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // r8
  unsigned int v14; // r14d
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  unsigned __int64 v17; // [rsp+8h] [rbp-38h]

  v6 = *(_QWORD **)(a2 + 32);
  if ( v6 )
  {
    v7 = *(unsigned int *)(a2 + 56);
    v8 = 0;
    if ( *(_DWORD *)(a2 + 56) && (--v7, v7) )
    {
      _BitScanReverse64(&v7, v7);
      v9 = *(unsigned int *)(a1 + 472);
      v8 = 64 - (v7 ^ 0x3F);
      v7 = 8LL * v8;
      if ( (unsigned int)v9 > v8 )
        goto LABEL_4;
    }
    else
    {
      v9 = *(unsigned int *)(a1 + 472);
      if ( (_DWORD)v9 )
        goto LABEL_4;
    }
    v13 = v8 + 1;
    v14 = v8 + 1;
    if ( v13 < v9 )
    {
      *(_DWORD *)(a1 + 472) = v13;
    }
    else if ( v13 > v9 )
    {
      if ( v13 > *(unsigned int *)(a1 + 476) )
      {
        v17 = v8 + 1;
        sub_16CD150(a1 + 464, (const void *)(a1 + 480), v13, 8, v13, a6);
        v9 = *(unsigned int *)(a1 + 472);
        v13 = v17;
      }
      v10 = *(_QWORD *)(a1 + 464);
      v15 = (_QWORD *)(v10 + 8 * v9);
      v16 = (_QWORD *)(v10 + 8 * v13);
      if ( v15 != v16 )
      {
        do
        {
          if ( v15 )
            *v15 = 0;
          ++v15;
        }
        while ( v16 != v15 );
        v10 = *(_QWORD *)(a1 + 464);
      }
      *(_DWORD *)(a1 + 472) = v14;
      goto LABEL_5;
    }
LABEL_4:
    v10 = *(_QWORD *)(a1 + 464);
LABEL_5:
    *v6 = *(_QWORD *)(v10 + v7);
    *(_QWORD *)(*(_QWORD *)(a1 + 464) + v7) = v6;
    *(_DWORD *)(a2 + 56) = 0;
    *(_QWORD *)(a2 + 32) = 0;
  }
  v11 = *(unsigned __int64 **)(a2 + 16);
  v12 = *(_QWORD *)(a2 + 8) & 0xFFFFFFFFFFFFFFF8LL;
  *v11 = v12 | *v11 & 7;
  *(_QWORD *)(v12 + 8) = v11;
  *(_QWORD *)(a2 + 16) = 0;
  *(_QWORD *)(a2 + 8) &= 7uLL;
  *(_QWORD *)a2 = *(_QWORD *)(a1 + 208);
  *(_QWORD *)(a1 + 208) = a2;
  *(_WORD *)(a2 + 24) = 0;
  sub_1D17020(*(_QWORD *)(a1 + 648), a2);
}
