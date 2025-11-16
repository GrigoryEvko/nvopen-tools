// Function: sub_35E72A0
// Address: 0x35e72a0
//
__int64 __fastcall sub_35E72A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned int v5; // esi
  unsigned int v6; // ecx

  v2 = *(_QWORD *)(a2 + 32);
  v3 = v2 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  v4 = v2 + 40LL * (unsigned int)sub_2E88FE0(a2);
  if ( v3 == v4 )
    return 0;
  v5 = 0;
  while ( 1 )
  {
    while ( 1 )
    {
      v6 = *(_DWORD *)(v4 + 8);
      if ( *(_BYTE *)v4 )
        break;
      v4 += 40;
      v5 = v6;
      if ( v3 == v4 )
        return 0;
    }
    if ( *(_BYTE *)v4 == 4 && *(_QWORD *)(a1 + 24) == *(_QWORD *)(v4 + 24) )
      break;
    v4 += 40;
    if ( v3 == v4 )
      return 0;
  }
  return v5;
}
