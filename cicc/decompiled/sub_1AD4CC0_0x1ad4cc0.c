// Function: sub_1AD4CC0
// Address: 0x1ad4cc0
//
void __fastcall sub_1AD4CC0(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // r12
  __int64 v4; // rdi

  v2 = *(_QWORD **)a1;
  v3 = (_QWORD *)(*(_QWORD *)a1 + 56LL * *(unsigned int *)(a1 + 8));
  if ( *(_QWORD **)a1 != v3 )
  {
    do
    {
      v4 = *(v3 - 3);
      v3 -= 7;
      if ( v4 )
        j_j___libc_free_0(v4, v3[6] - v4);
      if ( (_QWORD *)*v3 != v3 + 2 )
        j_j___libc_free_0(*v3, v3[2] + 1LL);
    }
    while ( v2 != v3 );
    v3 = *(_QWORD **)a1;
  }
  if ( v3 != (_QWORD *)(a1 + 16) )
    _libc_free((unsigned __int64)v3);
}
