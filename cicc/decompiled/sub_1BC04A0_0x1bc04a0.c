// Function: sub_1BC04A0
// Address: 0x1bc04a0
//
void __fastcall sub_1BC04A0(
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
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // r14
  double v17; // xmm4_8
  double v18; // xmm5_8
  unsigned __int64 *v19; // rdx
  __int64 v20; // rcx
  unsigned __int64 v21; // r15
  __int64 v22; // rcx
  _QWORD *v23; // rax
  __int64 v24; // rbx
  __int64 v25; // rdi

  v11 = a2;
  if ( a2 > 0xFFFFFFFF )
  {
    a2 = 1;
    sub_16BD1C0("SmallVector capacity overflow during allocation", 1u);
  }
  v12 = ((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4;
  v13 = ((v12
        | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
        | (*(unsigned int *)(a1 + 12) + 2LL)
        | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
      | v12
      | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
      | (*(unsigned int *)(a1 + 12) + 2LL)
      | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v14 = (v13 | (v13 >> 16) | HIDWORD(v13)) + 1;
  if ( v14 >= v11 )
    v11 = v14;
  v15 = v11;
  if ( v11 > 0xFFFFFFFF )
    v15 = 0xFFFFFFFFLL;
  v16 = malloc(8 * v15);
  if ( !v16 )
  {
    a2 = 1;
    sub_16BD1C0("Allocation failed", 1u);
  }
  v19 = *(unsigned __int64 **)a1;
  v20 = 8LL * *(unsigned int *)(a1 + 8);
  v21 = *(_QWORD *)a1 + v20;
  if ( *(_QWORD *)a1 != v21 )
  {
    v22 = v16 + v20;
    v23 = (_QWORD *)v16;
    do
    {
      if ( v23 )
      {
        a2 = *v19;
        *v23 = *v19;
        *v19 = 0;
      }
      ++v23;
      ++v19;
    }
    while ( v23 != (_QWORD *)v22 );
    v21 = *(_QWORD *)a1;
    v24 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( v24 != *(_QWORD *)a1 )
    {
      do
      {
        v25 = *(_QWORD *)(v24 - 8);
        v24 -= 8;
        if ( v25 )
          sub_164BEC0(v25, a2, (__int64)v19, v22, a3, a4, a5, a6, v17, v18, a9, a10);
      }
      while ( v21 != v24 );
      v21 = *(_QWORD *)a1;
    }
  }
  if ( v21 != a1 + 16 )
    _libc_free(v21);
  *(_QWORD *)a1 = v16;
  *(_DWORD *)(a1 + 12) = v15;
}
