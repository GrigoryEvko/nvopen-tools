// Function: sub_1AC6A30
// Address: 0x1ac6a30
//
__int64 __fastcall sub_1AC6A30(
        __int64 a1,
        unsigned __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rbx
  unsigned __int64 v13; // rax
  _QWORD *v14; // rdx
  __int64 v15; // rcx
  unsigned __int64 v16; // r14
  _QWORD *v17; // rax
  _QWORD *v18; // rcx
  __int64 v19; // r15
  __int64 v20; // r12
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 v24; // [rsp+8h] [rbp-38h]

  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  v10 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v11 = ((v10
        | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
        | (*(unsigned int *)(a1 + 12) + 2LL)
        | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
      | v10
      | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v12 = a2;
  v13 = (v11 | (v11 >> 16) | HIDWORD(v11)) + 1;
  if ( v13 >= a2 )
    v12 = v13;
  if ( v12 > 0xFFFFFFFF )
    v12 = 0xFFFFFFFFLL;
  v24 = malloc(8 * v12);
  if ( !v24 )
    sub_16BD1C0("Allocation failed", 1u);
  v14 = *(_QWORD **)a1;
  v15 = 8LL * *(unsigned int *)(a1 + 8);
  v16 = *(_QWORD *)a1 + v15;
  if ( *(_QWORD *)a1 != v16 )
  {
    v17 = (_QWORD *)v24;
    v18 = (_QWORD *)(v24 + v15);
    do
    {
      if ( v17 )
      {
        *v17 = *v14;
        *v14 = 0;
      }
      ++v17;
      ++v14;
    }
    while ( v17 != v18 );
    v16 = *(_QWORD *)a1;
    v19 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( v19 != *(_QWORD *)a1 )
    {
      do
      {
        v20 = *(_QWORD *)(v19 - 8);
        v19 -= 8;
        if ( v20 )
        {
          sub_15E5530(v20);
          sub_159D9E0(v20);
          sub_164BE60(v20, a3, a4, a5, a6, v21, v22, a9, a10);
          *(_DWORD *)(v20 + 20) = *(_DWORD *)(v20 + 20) & 0xF0000000 | 1;
          sub_1648B90(v20);
        }
      }
      while ( v16 != v19 );
      v16 = *(_QWORD *)a1;
    }
  }
  if ( v16 != a1 + 16 )
    _libc_free(v16);
  *(_DWORD *)(a1 + 12) = v12;
  *(_QWORD *)a1 = v24;
  return v24;
}
