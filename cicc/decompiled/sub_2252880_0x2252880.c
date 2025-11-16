// Function: sub_2252880
// Address: 0x2252880
//
__int64 *sub_2252880()
{
  __int64 *result; // rax
  __int64 v1; // rdi
  int v2; // edx
  int v3; // edx

  result = (__int64 *)sub_22529A0();
  v1 = *result;
  if ( *result )
  {
    if ( (unsigned __int64)(*(_QWORD *)(v1 + 80) - 0x474E5543432B2B00LL) > 1 )
    {
      *result = 0;
      return (__int64 *)sub_39F88E0(v1 + 80);
    }
    v2 = *(_DWORD *)(v1 + 40);
    if ( v2 < 0 )
    {
      v3 = v2 + 1;
      if ( !v3 )
        *result = *(_QWORD *)(v1 + 32);
    }
    else
    {
      v3 = v2 - 1;
      if ( !v3 )
      {
        *result = *(_QWORD *)(v1 + 32);
        return (__int64 *)sub_39F88E0(v1 + 80);
      }
      if ( v3 == -1 )
        sub_2207530();
    }
    *(_DWORD *)(v1 + 40) = v3;
  }
  return result;
}
