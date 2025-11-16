// Function: sub_36DFC00
// Address: 0x36dfc00
//
__int64 __fastcall sub_36DFC00(
        __int64 a1,
        __m128i a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6,
        int a7,
        __int64 *a8)
{
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rdx
  __int64 v12; // r8
  int v13; // r9d
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rdx

  v8 = *((unsigned int *)a8 + 2);
  if ( a7 )
    BUG();
  v9 = (unsigned int)(v8 + 2);
  v12 = *((unsigned int *)a8 + 2);
  v13 = v8 + 2;
  if ( v9 == v8 )
    goto LABEL_14;
  if ( v9 < v8 )
  {
    *((_DWORD *)a8 + 2) = v9;
LABEL_14:
    v15 = *a8;
    return sub_36DF750(a1, a5, a6, v15 + 24 * v8, v15 + 24LL * (unsigned int)(v12 + 1), a2);
  }
  v14 = *((unsigned int *)a8 + 2);
  if ( v9 > *((unsigned int *)a8 + 3) )
  {
    sub_C8D5F0((__int64)a8, a8 + 2, v9, 0x18u, v12, v9);
    v14 = *((unsigned int *)a8 + 2);
    LODWORD(v12) = v8;
    v13 = v8 + 2;
    v9 = (unsigned int)(v8 + 2);
  }
  v15 = *a8;
  v16 = *a8 + 24 * v14;
  v17 = *a8 + 24 * v9;
  if ( v16 != v17 )
  {
    do
    {
      if ( v16 )
      {
        *(_QWORD *)v16 = 0;
        *(_DWORD *)(v16 + 8) = 0;
        *(_QWORD *)(v16 + 16) = 0;
      }
      v16 += 24;
    }
    while ( v17 != v16 );
    v15 = *a8;
  }
  *((_DWORD *)a8 + 2) = v13;
  return sub_36DF750(a1, a5, a6, v15 + 24 * v8, v15 + 24LL * (unsigned int)(v12 + 1), a2);
}
