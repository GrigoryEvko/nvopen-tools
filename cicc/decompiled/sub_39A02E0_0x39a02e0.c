// Function: sub_39A02E0
// Address: 0x39a02e0
//
__int64 __fastcall sub_39A02E0(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 result; // rax
  __int64 *v3; // r13
  unsigned int v4; // r12d

  v1 = *(__int64 **)(a1 + 168);
  result = *(unsigned int *)(a1 + 176);
  v3 = &v1[result];
  if ( v1 != v3 )
  {
    v4 = 0;
    do
    {
      result = *v1;
      if ( *(_DWORD *)(*(_QWORD *)(*v1 + 80) + 36LL) != 3 )
      {
        *(_QWORD *)(result + 64) = v4;
        result = sub_39A02B0((__int64 *)a1, *v1);
        v4 += result;
      }
      ++v1;
    }
    while ( v3 != v1 );
  }
  return result;
}
