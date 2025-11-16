// Function: sub_1594B20
// Address: 0x1594b20
//
__int64 __fastcall sub_1594B20(__int64 a1)
{
  unsigned int v1; // edx
  __int64 v2; // r8
  __int64 v3; // rax
  unsigned __int64 v4; // rdx

  v1 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v2 = *(_QWORD *)(a1 - 24LL * v1);
  if ( v1 > 1 )
  {
    v3 = a1 - 24LL * v1;
    v4 = a1 + 24 * (v1 - 2 - (unsigned __int64)v1) + 24;
    while ( v2 == *(_QWORD *)(v3 + 24) )
    {
      v3 += 24;
      if ( v3 == v4 )
        return v2;
    }
    return 0;
  }
  return v2;
}
