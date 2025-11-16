// Function: sub_B1AC50
// Address: 0xb1ac50
//
__int64 __fastcall sub_B1AC50(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 result; // rax

  v3 = *(_QWORD *)(a1 + 528);
  v4 = v3 + 56LL * *(unsigned int *)(a1 + 536);
  if ( v3 != v4 )
  {
    do
    {
      v4 -= 56;
      v5 = *(_QWORD *)(v4 + 24);
      if ( v5 != v4 + 40 )
        _libc_free(v5, a2);
    }
    while ( v3 != v4 );
    v4 = *(_QWORD *)(a1 + 528);
  }
  result = a1 + 544;
  if ( v4 != a1 + 544 )
    result = _libc_free(v4, a2);
  if ( *(_QWORD *)a1 != a1 + 16 )
    return _libc_free(*(_QWORD *)a1, a2);
  return result;
}
