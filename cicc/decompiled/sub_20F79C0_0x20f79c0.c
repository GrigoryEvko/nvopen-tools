// Function: sub_20F79C0
// Address: 0x20f79c0
//
__int64 __fastcall sub_20F79C0(unsigned __int64 *a1)
{
  __int64 result; // rax
  __int64 v2; // r12
  __int64 v3; // [rsp-20h] [rbp-20h]

  result = *(unsigned int *)(*a1 + 16);
  if ( a1[4] != result )
  {
    _libc_free(a1[3]);
    v2 = *(unsigned int *)(*a1 + 16);
    a1[4] = v2;
    result = (__int64)_libc_calloc(v2, 1u);
    if ( !result && (v2 || (result = malloc(1u)) == 0) )
    {
      v3 = result;
      sub_16BD1C0("Allocation failed", 1u);
      result = v3;
    }
    a1[3] = result;
  }
  return result;
}
