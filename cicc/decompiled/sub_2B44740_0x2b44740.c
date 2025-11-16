// Function: sub_2B44740
// Address: 0x2b44740
//
void __fastcall sub_2B44740(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r14
  __int64 v5; // r12
  unsigned __int64 v6; // rbx

  v2 = *(_QWORD *)a1;
  v3 = *(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6);
  if ( *(_QWORD *)a1 != v3 )
  {
    do
    {
      if ( a2 )
        sub_C8CF70(a2, (void *)(a2 + 32), 4, v2 + 32, v2);
      v2 += 64;
      a2 += 64;
    }
    while ( v3 != v2 );
    v5 = *(_QWORD *)a1;
    v6 = *(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6);
    if ( *(_QWORD *)a1 != v6 )
    {
      do
      {
        while ( 1 )
        {
          v6 -= 64LL;
          if ( !*(_BYTE *)(v6 + 28) )
            break;
          if ( v6 == v5 )
            return;
        }
        _libc_free(*(_QWORD *)(v6 + 8));
      }
      while ( v6 != v5 );
    }
  }
}
