// Function: sub_A4F9D0
// Address: 0xa4f9d0
//
__int64 __fastcall sub_A4F9D0(__int64 a1, __int64 a2)
{
  _DWORD *v3; // rbx
  _DWORD *v4; // r12
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // rdi
  _BYTE *v8; // rax
  __int64 v9; // rbx
  __int64 result; // rax
  __int64 v11; // r12
  __int64 v12; // rdi

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
  result = 5LL * *(unsigned int *)(a1 + 48);
  v11 = v9 + 40LL * *(unsigned int *)(a1 + 48);
  if ( v9 != v11 )
  {
    do
    {
      v11 -= 40;
      v12 = *(_QWORD *)(v11 + 8);
      result = v11 + 24;
      if ( v12 != v11 + 24 )
      {
        a2 = *(_QWORD *)(v11 + 24) + 1LL;
        result = j_j___libc_free_0(v12, a2);
      }
    }
    while ( v9 != v11 );
    v11 = *(_QWORD *)(a1 + 40);
  }
  if ( v11 != a1 + 56 )
    return _libc_free(v11, a2);
  return result;
}
