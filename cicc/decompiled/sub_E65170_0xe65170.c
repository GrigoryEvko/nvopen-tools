// Function: sub_E65170
// Address: 0xe65170
//
void __fastcall sub_E65170(__int64 a1)
{
  _OWORD *v1; // rax
  __int64 **v2; // r13
  __int64 *v3; // rbx
  char *v4; // r12
  __int64 *v5; // rbx
  __int64 *v6; // r12
  __int64 *v7; // rdi

  if ( !*(_QWORD *)(a1 + 88) )
  {
    v1 = (_OWORD *)sub_22077B0(64);
    if ( v1 )
    {
      *v1 = 0;
      v1[1] = 0;
      v1[2] = 0;
      v1[3] = 0;
    }
    v2 = *(__int64 ***)(a1 + 88);
    *(_QWORD *)(a1 + 88) = v1;
    if ( v2 )
    {
      v3 = v2[4];
      v4 = (char *)v2[3];
      if ( v3 != (__int64 *)v4 )
      {
        do
        {
          if ( *(char **)v4 != v4 + 16 )
            j_j___libc_free_0(*(_QWORD *)v4, *((_QWORD *)v4 + 2) + 1LL);
          v4 += 32;
        }
        while ( v3 != (__int64 *)v4 );
        v4 = (char *)v2[3];
      }
      if ( v4 )
        j_j___libc_free_0(v4, (char *)v2[5] - v4);
      v5 = v2[1];
      v6 = *v2;
      if ( v5 != *v2 )
      {
        do
        {
          v7 = v6;
          v6 += 3;
          sub_C8EE20(v7);
        }
        while ( v5 != v6 );
        v6 = *v2;
      }
      if ( v6 )
        j_j___libc_free_0(v6, (char *)v2[2] - (char *)v6);
      j_j___libc_free_0(v2, 64);
    }
  }
}
