// Function: sub_1F032D0
// Address: 0x1f032d0
//
__int64 __fastcall sub_1F032D0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  __int64 v5; // rbx
  unsigned int v6; // r14d
  __int64 v7; // r15

  v4 = sub_1F03240(a1, a3, a2);
  if ( (_BYTE)v4 )
    return 1;
  v5 = *(_QWORD *)(a2 + 32);
  v6 = v4;
  v7 = v5 + 16LL * *(unsigned int *)(a2 + 40);
  if ( v7 != v5 )
  {
    while ( (*(_QWORD *)v5 & 6) != 0
         || !*(_DWORD *)(v5 + 8)
         || !(unsigned __int8)sub_1F03240(a1, a3, *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL) )
    {
      v5 += 16;
      if ( v7 == v5 )
        return v6;
    }
    return 1;
  }
  return v6;
}
