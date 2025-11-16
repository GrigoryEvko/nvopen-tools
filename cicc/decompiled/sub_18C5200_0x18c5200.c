// Function: sub_18C5200
// Address: 0x18c5200
//
__int64 __fastcall sub_18C5200(__int64 a1)
{
  __int64 v2; // r12
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  _QWORD *v5; // rdi
  unsigned __int64 v6; // rdi

  v2 = *(unsigned int *)(a1 + 320);
  v3 = *(_QWORD **)(a1 + 312);
  *(_QWORD *)a1 = off_49F2528;
  v4 = &v3[8 * v2];
  if ( v3 != v4 )
  {
    do
    {
      v4 -= 8;
      v5 = (_QWORD *)v4[4];
      if ( v5 != v4 + 6 )
        j_j___libc_free_0(v5, v4[6] + 1LL);
      if ( (_QWORD *)*v4 != v4 + 2 )
        j_j___libc_free_0(*v4, v4[2] + 1LL);
    }
    while ( v3 != v4 );
    v4 = *(_QWORD **)(a1 + 312);
  }
  if ( v4 != (_QWORD *)(a1 + 328) )
    _libc_free((unsigned __int64)v4);
  v6 = *(_QWORD *)(a1 + 160);
  if ( v6 != a1 + 176 )
    _libc_free(v6);
  sub_1636790((_QWORD *)a1);
  return j_j___libc_free_0(a1, 2376);
}
