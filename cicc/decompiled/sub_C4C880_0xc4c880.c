// Function: sub_C4C880
// Address: 0xc4c880
//
__int64 __fastcall sub_C4C880(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 v3; // rdi
  unsigned int v4; // r8d
  bool v5; // r10
  __int64 v6; // rsi
  __int64 v7; // rdx
  unsigned int v8; // r8d
  __int64 v10; // rdi
  __int64 v11; // rax
  _BOOL4 v12; // r8d

  v2 = *(_DWORD *)(a1 + 8);
  v3 = *(_QWORD *)a1;
  if ( v2 <= 0x40 )
  {
    v8 = 0;
    if ( !v2 )
      return v8;
    v10 = v3 << (64 - (unsigned __int8)v2) >> (64 - (unsigned __int8)v2);
    v11 = (__int64)(*(_QWORD *)a2 << (64 - (unsigned __int8)v2)) >> (64 - (unsigned __int8)v2);
    v12 = v11 < v10;
    if ( v11 > v10 )
      return -1;
    return v12;
  }
  else
  {
    v4 = *(_DWORD *)(a2 + 8);
    v5 = (*(_QWORD *)(v3 + 8LL * ((v2 - 1) >> 6)) & (1LL << ((unsigned __int8)v2 - 1))) != 0;
    v6 = *(_QWORD *)a2;
    v7 = 1LL << ((unsigned __int8)v4 - 1);
    if ( v4 <= 0x40 )
    {
      if ( ((v6 & v7) != 0) != v5 )
        return (*(_QWORD *)(v3 + 8LL * ((v2 - 1) >> 6)) & (1LL << ((unsigned __int8)v2 - 1))) == 0 ? 1 : -1;
    }
    else if ( ((*(_QWORD *)(v6 + 8LL * ((v4 - 1) >> 6)) & v7) != 0) != v5 )
    {
      return (*(_QWORD *)(v3 + 8LL * ((v2 - 1) >> 6)) & (1LL << ((unsigned __int8)v2 - 1))) == 0 ? 1 : -1;
    }
    return sub_C49940(v3, v6, ((unsigned __int64)v2 + 63) >> 6);
  }
}
