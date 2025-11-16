// Function: sub_2DDD850
// Address: 0x2ddd850
//
void __fastcall sub_2DDD850(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rdx
  _QWORD *v4; // rdx
  __int64 v5; // rcx
  _QWORD *v6; // r13
  __int64 v7; // rbx
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // rdi

  v2 = *(_QWORD **)a1;
  v3 = 4LL * *(unsigned int *)(a1 + 8);
  if ( v3 * 8 )
  {
    v4 = &a2[v3];
    do
    {
      if ( a2 )
      {
        *a2 = *v2;
        a2[1] = v2[1];
        a2[2] = v2[2];
        v5 = v2[3];
        v2[2] = 0;
        a2[3] = v5;
        v2[3] = 0;
      }
      a2 += 4;
      v2 += 4;
    }
    while ( a2 != v4 );
    v6 = *(_QWORD **)a1;
    v7 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v7 )
    {
      do
      {
        v8 = *(_QWORD *)(v7 - 8);
        v7 -= 32;
        if ( v8 )
        {
          sub_C7D6A0(*(_QWORD *)(v8 + 8), 16LL * *(unsigned int *)(v8 + 24), 8);
          j_j___libc_free_0(v8);
        }
        v9 = *(_QWORD *)(v7 + 16);
        if ( v9 )
        {
          v10 = *(_QWORD *)(v9 + 32);
          if ( v10 != v9 + 48 )
            _libc_free(v10);
          sub_C7D6A0(*(_QWORD *)(v9 + 8), 8LL * *(unsigned int *)(v9 + 24), 4);
          j_j___libc_free_0(v9);
        }
      }
      while ( (_QWORD *)v7 != v6 );
    }
  }
}
