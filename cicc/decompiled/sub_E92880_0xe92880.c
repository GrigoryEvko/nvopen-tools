// Function: sub_E92880
// Address: 0xe92880
//
__int64 __fastcall sub_E92880(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 result; // rax
  __int64 v5; // r13
  _QWORD *i; // rbx
  __int64 v7; // rdi

  v3 = *(_QWORD *)(a1 + 88);
  *(_QWORD *)a1 = &unk_49E3588;
  result = 3LL * *(unsigned int *)(a1 + 96);
  v5 = v3 + 24LL * *(unsigned int *)(a1 + 96);
  if ( v5 != v3 )
  {
    do
    {
      for ( i = *(_QWORD **)(v3 + 8); i; result = sub_E81B70(v7, a2) )
      {
        v7 = (__int64)i;
        i = (_QWORD *)*i;
      }
      v3 += 24;
    }
    while ( v3 != v5 );
    v3 = *(_QWORD *)(a1 + 88);
  }
  if ( v3 != a1 + 104 )
    return _libc_free(v3, a2);
  return result;
}
