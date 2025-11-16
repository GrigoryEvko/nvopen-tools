// Function: sub_1B8F7D0
// Address: 0x1b8f7d0
//
__int64 __fastcall sub_1B8F7D0(__int64 a1)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rcx

  if ( !a1 )
    return 0;
  if ( !*(_QWORD *)(a1 + 48) )
  {
    v2 = 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    {
      v3 = *(_QWORD *)(a1 - 8);
      v4 = v3 + v2;
    }
    else
    {
      v4 = a1;
      v3 = a1 - v2;
    }
    while ( v3 != v4 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v3 + 16LL) > 0x17u && *(_QWORD *)(*(_QWORD *)v3 + 48LL) )
        return *(_QWORD *)v3;
      v3 += 24;
    }
  }
  return a1;
}
