// Function: sub_B18DF0
// Address: 0xb18df0
//
__int64 __fastcall sub_B18DF0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 v6; // rdi
  __int64 v7; // rdi

  v3 = *(_QWORD *)(a1 + 200);
  *(_QWORD *)a1 = &unk_49D9FE8;
  v4 = v3 + 8LL * *(unsigned int *)(a1 + 208);
  if ( v3 != v4 )
  {
    do
    {
      v5 = *(_QWORD *)(v4 - 8);
      v4 -= 8;
      if ( v5 )
      {
        v6 = *(_QWORD *)(v5 + 24);
        if ( v6 != v5 + 40 )
          _libc_free(v6, a2);
        a2 = 80;
        j_j___libc_free_0(v5, 80);
      }
    }
    while ( v3 != v4 );
    v4 = *(_QWORD *)(a1 + 200);
  }
  if ( v4 != a1 + 216 )
    _libc_free(v4, a2);
  v7 = *(_QWORD *)(a1 + 176);
  if ( v7 != a1 + 192 )
    _libc_free(v7, a2);
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
