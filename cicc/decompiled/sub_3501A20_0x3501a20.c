// Function: sub_3501A20
// Address: 0x3501a20
//
__int64 __fastcall sub_3501A20(unsigned __int64 *a1)
{
  __int64 result; // rax
  __int64 v2; // r12

  result = *(unsigned int *)(*a1 + 16);
  if ( a1[4] != result )
  {
    _libc_free(a1[3]);
    v2 = *(unsigned int *)(*a1 + 16);
    a1[4] = v2;
    result = (__int64)_libc_calloc(v2, 1u);
    if ( !result && (v2 || (result = malloc(1u)) == 0) )
      sub_C64F00("Allocation failed", 1u);
    a1[3] = result;
  }
  return result;
}
