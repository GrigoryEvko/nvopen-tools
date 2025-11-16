// Function: sub_B15410
// Address: 0xb15410
//
__int64 __fastcall sub_B15410(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  _QWORD *v5; // rdi

  v3 = *(_QWORD **)(a1 + 80);
  *(_QWORD *)a1 = &unk_49D9D40;
  v4 = &v3[10 * *(unsigned int *)(a1 + 88)];
  if ( v3 != v4 )
  {
    do
    {
      v4 -= 10;
      v5 = (_QWORD *)v4[4];
      if ( v5 != v4 + 6 )
      {
        a2 = v4[6] + 1LL;
        j_j___libc_free_0(v5, a2);
      }
      if ( (_QWORD *)*v4 != v4 + 2 )
      {
        a2 = v4[2] + 1LL;
        j_j___libc_free_0(*v4, a2);
      }
    }
    while ( v3 != v4 );
    v4 = *(_QWORD **)(a1 + 80);
  }
  if ( v4 != (_QWORD *)(a1 + 96) )
    _libc_free(v4, a2);
  return j_j___libc_free_0(a1, 432);
}
