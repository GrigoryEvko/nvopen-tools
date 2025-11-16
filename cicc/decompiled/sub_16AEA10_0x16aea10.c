// Function: sub_16AEA10
// Address: 0x16aea10
//
__int64 __fastcall sub_16AEA10(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 v3; // r8
  __int64 v4; // rdi
  unsigned int v5; // esi
  bool v6; // r10
  __int64 v7; // rdx
  __int64 result; // rax
  __int64 v9; // rdi
  __int64 v10; // r8

  v2 = *(_DWORD *)(a1 + 8);
  v3 = *(_QWORD *)a2;
  v4 = *(_QWORD *)a1;
  if ( v2 <= 0x40 )
  {
    v9 = v4 << (64 - (unsigned __int8)v2) >> (64 - (unsigned __int8)v2);
    v10 = v3 << (64 - (unsigned __int8)v2) >> (64 - (unsigned __int8)v2);
    result = v10 < v9;
    if ( v10 > v9 )
      return 0xFFFFFFFFLL;
  }
  else
  {
    v5 = *(_DWORD *)(a2 + 8);
    v6 = (*(_QWORD *)(v4 + 8LL * ((v2 - 1) >> 6)) & (1LL << ((unsigned __int8)v2 - 1))) != 0;
    v7 = 1LL << ((unsigned __int8)v5 - 1);
    if ( v5 <= 0x40 )
    {
      if ( ((v3 & v7) != 0) != v6 )
        return (*(_QWORD *)(v4 + 8LL * ((v2 - 1) >> 6)) & (1LL << ((unsigned __int8)v2 - 1))) == 0 ? 1 : -1;
    }
    else if ( ((*(_QWORD *)(v3 + 8LL * ((v5 - 1) >> 6)) & v7) != 0) != v6 )
    {
      return (*(_QWORD *)(v4 + 8LL * ((v2 - 1) >> 6)) & (1LL << ((unsigned __int8)v2 - 1))) == 0 ? 1 : -1;
    }
    return sub_16A98D0(v4, v3, ((unsigned __int64)v2 + 63) >> 6);
  }
  return result;
}
