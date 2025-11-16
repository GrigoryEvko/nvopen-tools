// Function: sub_1DCB3F0
// Address: 0x1dcb3f0
//
__int64 __fastcall sub_1DCB3F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx

  v2 = *(_QWORD *)(a1 + 32);
  v3 = (*(_QWORD *)(a1 + 40) - v2) >> 3;
  if ( !(_DWORD)v3 )
    return 0;
  v4 = v2 + 8LL * (unsigned int)(v3 - 1) + 8;
  while ( a2 != *(_QWORD *)(*(_QWORD *)v2 + 24LL) )
  {
    v2 += 8;
    if ( v2 == v4 )
      return 0;
  }
  return *(_QWORD *)v2;
}
