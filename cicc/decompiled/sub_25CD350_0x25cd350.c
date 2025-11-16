// Function: sub_25CD350
// Address: 0x25cd350
//
void __fastcall sub_25CD350(_QWORD *a1)
{
  _QWORD *v1; // r14
  unsigned __int64 v2; // r13
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v4; // r12
  unsigned __int64 v5; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = (unsigned __int64)v1;
      sub_25CD350(v1[3]);
      v3 = (unsigned __int64 *)v1[9];
      v4 = (unsigned __int64 *)v1[8];
      v1 = (_QWORD *)v1[2];
      if ( v3 != v4 )
      {
        do
        {
          if ( (unsigned __int64 *)*v4 != v4 + 2 )
            j_j___libc_free_0(*v4);
          v4 += 4;
        }
        while ( v3 != v4 );
        v4 = *(unsigned __int64 **)(v2 + 64);
      }
      if ( v4 )
        j_j___libc_free_0((unsigned __int64)v4);
      v5 = *(_QWORD *)(v2 + 32);
      if ( v5 != v2 + 48 )
        j_j___libc_free_0(v5);
      j_j___libc_free_0(v2);
    }
    while ( v1 );
  }
}
