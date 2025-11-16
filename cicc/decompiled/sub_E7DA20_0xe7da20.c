// Function: sub_E7DA20
// Address: 0xe7da20
//
__int64 __fastcall sub_E7DA20(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rdi

  v3 = *(_QWORD *)(a1 + 3528);
  *(_QWORD *)a1 = &unk_49E1FB0;
  v4 = v3 + 48LL * *(unsigned int *)(a1 + 3536);
  if ( v3 != v4 )
  {
    do
    {
      v4 -= 48;
      v5 = *(_QWORD *)(v4 + 16);
      if ( v5 != v4 + 32 )
      {
        a2 = *(_QWORD *)(v4 + 32) + 1LL;
        j_j___libc_free_0(v5, a2);
      }
    }
    while ( v3 != v4 );
    v4 = *(_QWORD *)(a1 + 3528);
  }
  if ( v4 != a1 + 3544 )
    _libc_free(v4, a2);
  v6 = *(_QWORD *)(a1 + 440);
  v7 = v6 + 48LL * *(unsigned int *)(a1 + 448);
  if ( v6 != v7 )
  {
    do
    {
      v7 -= 48;
      v8 = *(_QWORD *)(v7 + 16);
      if ( v8 != v7 + 32 )
      {
        a2 = *(_QWORD *)(v7 + 32) + 1LL;
        j_j___libc_free_0(v8, a2);
      }
    }
    while ( v6 != v7 );
    v7 = *(_QWORD *)(a1 + 440);
  }
  if ( v7 != a1 + 456 )
    _libc_free(v7, a2);
  return sub_E8A760(a1);
}
