// Function: sub_311AF30
// Address: 0x311af30
//
void __fastcall sub_311AF30(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rdx
  _QWORD *v4; // rdx
  _QWORD *v5; // r14
  __int64 v6; // rbx
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r13

  v2 = *(_QWORD **)a1;
  v3 = *(unsigned int *)(a1 + 8);
  if ( v3 * 8 )
  {
    v4 = &a2[v3];
    do
    {
      if ( a2 )
      {
        *a2 = *v2;
        *v2 = 0;
      }
      ++a2;
      ++v2;
    }
    while ( v4 != a2 );
    v5 = *(_QWORD **)a1;
    v6 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    while ( v5 != (_QWORD *)v6 )
    {
      while ( 1 )
      {
        v7 = *(_QWORD *)(v6 - 8);
        v6 -= 8;
        if ( !v7 )
          break;
        v8 = *(_QWORD *)(v7 + 24);
        if ( v8 )
        {
          sub_C7D6A0(*(_QWORD *)(v8 + 8), 16LL * *(unsigned int *)(v8 + 24), 8);
          j_j___libc_free_0(v8);
        }
        j_j___libc_free_0(v7);
        if ( v5 == (_QWORD *)v6 )
          return;
      }
    }
  }
}
