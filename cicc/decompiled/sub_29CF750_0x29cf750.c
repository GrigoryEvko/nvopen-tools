// Function: sub_29CF750
// Address: 0x29cf750
//
void __fastcall sub_29CF750(__int64 *a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // r13
  __int64 v4; // r14
  unsigned __int64 v5; // r12

  v1 = *a1;
  if ( *a1 )
  {
    if ( (v1 & 4) != 0 )
    {
      v2 = v1 & 0xFFFFFFFFFFFFFFF8LL;
      v3 = v2;
      if ( v2 )
      {
        v4 = *(_QWORD *)(v2 + 8);
        v5 = v4 + 8LL * *(unsigned int *)(v2 + 16);
        if ( v4 != v5 )
        {
          do
          {
            v5 -= 8LL;
            sub_29CF750(v5);
          }
          while ( v4 != v5 );
          v5 = *(_QWORD *)(v3 + 8);
        }
        if ( v5 != v3 + 24 )
          _libc_free(v5);
        j_j___libc_free_0(v3);
      }
    }
  }
  *a1 = 0;
}
