// Function: sub_1E37750
// Address: 0x1e37750
//
void __fastcall sub_1E37750(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  _QWORD *v4; // rdi

  v2 = *(_QWORD **)(a1 + 88);
  *(_QWORD *)a1 = &unk_49ECF68;
  v3 = &v2[11 * *(unsigned int *)(a1 + 96)];
  if ( v2 != v3 )
  {
    do
    {
      v3 -= 11;
      v4 = (_QWORD *)v3[4];
      if ( v4 != v3 + 6 )
        j_j___libc_free_0(v4, v3[6] + 1LL);
      if ( (_QWORD *)*v3 != v3 + 2 )
        j_j___libc_free_0(*v3, v3[2] + 1LL);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD **)(a1 + 88);
  }
  if ( v3 != (_QWORD *)(a1 + 104) )
    _libc_free((unsigned __int64)v3);
}
