// Function: sub_2A8AC40
// Address: 0x2a8ac40
//
void __fastcall sub_2A8AC40(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // r12
  __int64 v5; // rbx
  unsigned __int64 v6; // rdi

  v2 = *a1;
  v3 = *a1 + 24LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v3 )
  {
    do
    {
      if ( a2 )
      {
        *(_QWORD *)a2 = *(_QWORD *)v2;
        *(_DWORD *)(a2 + 16) = *(_DWORD *)(v2 + 16);
        *(_QWORD *)(a2 + 8) = *(_QWORD *)(v2 + 8);
        *(_DWORD *)(v2 + 16) = 0;
      }
      v2 += 24;
      a2 += 24;
    }
    while ( v3 != v2 );
    v4 = *a1;
    v5 = *a1 + 24LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v5 )
    {
      do
      {
        v5 -= 24;
        if ( *(_DWORD *)(v5 + 16) > 0x40u )
        {
          v6 = *(_QWORD *)(v5 + 8);
          if ( v6 )
            j_j___libc_free_0_0(v6);
        }
      }
      while ( v5 != v4 );
    }
  }
}
