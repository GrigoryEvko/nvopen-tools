// Function: sub_1B8EA10
// Address: 0x1b8ea10
//
__int64 __fastcall sub_1B8EA10(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r13
  __int64 v3; // rax
  unsigned __int64 *v4; // r14
  unsigned __int64 *v5; // r12
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_1B8EA10(*(_QWORD *)(v1 + 24));
      v3 = *(unsigned int *)(v1 + 48);
      v4 = *(unsigned __int64 **)(v1 + 40);
      v1 = *(_QWORD *)(v1 + 16);
      v5 = &v4[6 * v3];
      if ( v4 != v5 )
      {
        do
        {
          v5 -= 6;
          if ( (unsigned __int64 *)*v5 != v5 + 2 )
            _libc_free(*v5);
        }
        while ( v4 != v5 );
        v5 = *(unsigned __int64 **)(v2 + 40);
      }
      if ( v5 != (unsigned __int64 *)(v2 + 56) )
        _libc_free((unsigned __int64)v5);
      result = j_j___libc_free_0(v2, 152);
    }
    while ( v1 );
  }
  return result;
}
