// Function: sub_2A2E1E0
// Address: 0x2a2e1e0
//
void __fastcall sub_2A2E1E0(_QWORD *a1)
{
  _QWORD *v1; // r13
  unsigned __int64 v2; // r14
  __int64 v3; // rbx
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = (unsigned __int64)v1;
      sub_2A2E1E0(v1[3]);
      v3 = v1[6];
      v4 = v1[5];
      v1 = (_QWORD *)v1[2];
      if ( v3 != v4 )
      {
        do
        {
          v5 = *(_QWORD *)(v4 + 8);
          if ( v5 )
            j_j___libc_free_0(v5);
          v4 += 32LL;
        }
        while ( v3 != v4 );
        v4 = *(_QWORD *)(v2 + 40);
      }
      if ( v4 )
        j_j___libc_free_0(v4);
      j_j___libc_free_0(v2);
    }
    while ( v1 );
  }
}
