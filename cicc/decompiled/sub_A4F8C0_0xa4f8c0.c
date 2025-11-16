// Function: sub_A4F8C0
// Address: 0xa4f8c0
//
__int64 __fastcall sub_A4F8C0(__int64 a1, __int64 a2)
{
  _DWORD *v3; // rbx
  _DWORD *v4; // r12
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // rdi
  _BYTE *v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // rdi

  v3 = *(_DWORD **)(a1 + 40);
  *(_QWORD *)a1 = off_4979450;
  v4 = &v3[10 * *(unsigned int *)(a1 + 48)];
  while ( v4 != v3 )
  {
    v7 = *(_QWORD *)(a1 + 280);
    v8 = *(_BYTE **)(v7 + 32);
    if ( *(_BYTE **)(v7 + 24) == v8 )
    {
      sub_CB6200(v7, "\n", 1);
    }
    else
    {
      *v8 = 10;
      ++*(_QWORD *)(v7 + 32);
    }
    v5 = *v3;
    v3 += 10;
    v6 = sub_CB69B0(*(_QWORD *)(a1 + 280), (unsigned int)(2 * v5));
    a2 = *((_QWORD *)v3 - 4);
    sub_CB6200(v6, a2, *((_QWORD *)v3 - 3));
  }
  if ( !*(_BYTE *)(a1 + 244) )
    _libc_free(*(_QWORD *)(a1 + 224), a2);
  v9 = *(_QWORD *)(a1 + 40);
  v10 = v9 + 40LL * *(unsigned int *)(a1 + 48);
  if ( v9 != v10 )
  {
    do
    {
      v10 -= 40;
      v11 = *(_QWORD *)(v10 + 8);
      if ( v11 != v10 + 24 )
      {
        a2 = *(_QWORD *)(v10 + 24) + 1LL;
        j_j___libc_free_0(v11, a2);
      }
    }
    while ( v9 != v10 );
    v10 = *(_QWORD *)(a1 + 40);
  }
  if ( v10 != a1 + 56 )
    _libc_free(v10, a2);
  return j_j___libc_free_0(a1, 288);
}
