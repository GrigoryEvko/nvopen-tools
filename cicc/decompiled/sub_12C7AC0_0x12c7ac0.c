// Function: sub_12C7AC0
// Address: 0x12c7ac0
//
__int64 __fastcall sub_12C7AC0(_DWORD *a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v4; // r8
  __int64 v5; // rbx
  __int64 v6; // rdi

  result = (unsigned int)*a1;
  v4 = *a2;
  if ( (int)result > 0 )
  {
    v5 = 0;
    do
    {
      v6 = *(_QWORD *)(v4 + 8 * v5);
      if ( v6 )
      {
        result = j_j___libc_free_0_0(v6);
        v4 = *a2;
      }
      ++v5;
    }
    while ( *a1 > (int)v5 );
  }
  if ( v4 )
    result = j_j___libc_free_0_0(v4);
  *a1 = 0;
  *a2 = 0;
  return result;
}
