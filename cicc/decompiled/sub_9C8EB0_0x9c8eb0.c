// Function: sub_9C8EB0
// Address: 0x9c8eb0
//
__int64 __fastcall sub_9C8EB0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 result; // rax

  v3 = a1[1];
  v4 = *a1;
  if ( v3 != *a1 )
  {
    do
    {
      v5 = *(_QWORD *)(v4 + 8);
      result = v4 + 24;
      if ( v5 != v4 + 24 )
        result = _libc_free(v5, a2);
      v4 += 72;
    }
    while ( v3 != v4 );
    v4 = *a1;
  }
  if ( v4 )
    return j_j___libc_free_0(v4, a1[2] - v4);
  return result;
}
