// Function: sub_1E472E0
// Address: 0x1e472e0
//
__int64 __fastcall sub_1E472E0(__int64 *a1)
{
  __int64 v2; // rdi
  __int64 *v3; // rbx
  unsigned __int64 v4; // r12
  __int64 v5; // rdi
  __int64 result; // rax

  v2 = *a1;
  if ( v2 )
  {
    v3 = (__int64 *)a1[5];
    v4 = a1[9] + 8;
    if ( v4 > (unsigned __int64)v3 )
    {
      do
      {
        v5 = *v3++;
        j_j___libc_free_0(v5, 512);
      }
      while ( v4 > (unsigned __int64)v3 );
      v2 = *a1;
    }
    return j_j___libc_free_0(v2, 8 * a1[1]);
  }
  return result;
}
