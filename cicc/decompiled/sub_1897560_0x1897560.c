// Function: sub_1897560
// Address: 0x1897560
//
void __fastcall sub_1897560(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 *v10; // r12
  __int64 *v11; // r13
  __int64 v12; // rdi
  __int64 v13; // r12
  __int64 v14; // r13
  unsigned __int64 *v15; // r14
  unsigned __int64 *v16; // r12
  unsigned __int64 *v17; // r12
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi

  sub_164D160(*(_QWORD *)(a1 + 8), *(_QWORD *)a1, a2, a3, a4, a5, a6, a7, a8, a9);
  sub_15E3D00(*(_QWORD *)(a1 + 8));
  if ( !*(_BYTE *)(a1 + 96) )
  {
    v10 = *(__int64 **)(a1 + 16);
    v11 = &v10[2 * *(unsigned int *)(a1 + 24)];
    while ( v11 != v10 )
    {
      v12 = *v10;
      v10 += 2;
      sub_15E3D00(v12);
    }
  }
  v13 = *(_QWORD *)(a1 + 120);
  if ( v13 )
  {
    sub_1368A00(*(__int64 **)(a1 + 120));
    j_j___libc_free_0(v13, 8);
  }
  v14 = *(_QWORD *)(a1 + 112);
  if ( v14 )
  {
    v15 = *(unsigned __int64 **)v14;
    v16 = (unsigned __int64 *)(*(_QWORD *)v14 + 104LL * *(unsigned int *)(v14 + 8));
    if ( *(unsigned __int64 **)v14 != v16 )
    {
      do
      {
        v16 -= 13;
        if ( (unsigned __int64 *)*v16 != v16 + 2 )
          _libc_free(*v16);
      }
      while ( v15 != v16 );
      v16 = *(unsigned __int64 **)v14;
    }
    if ( v16 != (unsigned __int64 *)(v14 + 16) )
      _libc_free((unsigned __int64)v16);
    j_j___libc_free_0(v14, 432);
  }
  v17 = *(unsigned __int64 **)(a1 + 104);
  if ( v17 )
  {
    v18 = v17[8];
    if ( (unsigned __int64 *)v18 != v17 + 10 )
      _libc_free(v18);
    if ( (unsigned __int64 *)*v17 != v17 + 2 )
      _libc_free(*v17);
    j_j___libc_free_0(v17, 112);
  }
  v19 = *(_QWORD *)(a1 + 16);
  if ( v19 != a1 + 32 )
    _libc_free(v19);
}
