// Function: sub_214A180
// Address: 0x214a180
//
__int64 __fastcall sub_214A180(__int64 a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12

  v1 = *(_QWORD **)(a1 + 16);
  v2 = &v1[4 * *(unsigned int *)(a1 + 24)];
  *(_QWORD *)a1 = &unk_4A01240;
  if ( v1 != v2 )
  {
    do
    {
      v2 -= 4;
      if ( (_QWORD *)*v2 != v2 + 2 )
        j_j___libc_free_0(*v2, v2[2] + 1LL);
    }
    while ( v1 != v2 );
    v2 = *(_QWORD **)(a1 + 16);
  }
  if ( v2 != (_QWORD *)(a1 + 32) )
    _libc_free((unsigned __int64)v2);
  return nullsub_1936(a1);
}
