// Function: sub_2485550
// Address: 0x2485550
//
void __fastcall sub_2485550(_QWORD *a1)
{
  _QWORD *v1; // r13
  unsigned __int64 v2; // r12
  _QWORD *v3; // rbx
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = (unsigned __int64)v1;
      sub_2485550(v1[3]);
      v3 = (_QWORD *)v1[7];
      v1 = (_QWORD *)v1[2];
      while ( v3 )
      {
        v4 = (unsigned __int64)v3;
        v3 = (_QWORD *)*v3;
        j_j___libc_free_0(v4);
      }
      memset(*(void **)(v2 + 40), 0, 8LL * *(_QWORD *)(v2 + 48));
      v5 = *(_QWORD *)(v2 + 40);
      *(_QWORD *)(v2 + 64) = 0;
      *(_QWORD *)(v2 + 56) = 0;
      if ( v5 != v2 + 88 )
        j_j___libc_free_0(v5);
      j_j___libc_free_0(v2);
    }
    while ( v1 );
  }
}
