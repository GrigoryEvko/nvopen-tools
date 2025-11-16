// Function: sub_16880C0
// Address: 0x16880c0
//
__int64 __fastcall sub_16880C0(_DWORD *a1)
{
  unsigned int v1; // edx
  __int64 result; // rax
  unsigned int v3; // ecx

  if ( !a1 )
    return 0;
  v1 = a1[2];
  result = 0;
  if ( v1 < *(_DWORD *)(*(_QWORD *)a1 + 80LL) )
  {
    v3 = a1[3];
    if ( v3 )
    {
      _BitScanForward(&v3, v3);
      result = *(_QWORD *)(*(_QWORD *)a1 + 88LL) + 8LL * (v3 + 32 * v1);
      if ( result )
        return *(_QWORD *)result;
    }
  }
  return result;
}
