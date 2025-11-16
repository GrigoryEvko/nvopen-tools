// Function: sub_2E8A790
// Address: 0x2e8a790
//
__int64 __fastcall sub_2E8A790(__int64 a1, int a2, unsigned int a3, unsigned int a4, _QWORD *a5)
{
  __int64 v6; // rdx
  __int64 v9; // rsi
  __int64 v10; // rbx
  __int64 result; // rax
  __int64 j; // r13
  __int64 v13; // rbx
  __int64 i; // r13
  unsigned int v15; // [rsp+Ch] [rbp-34h]

  v6 = a4;
  if ( a3 - 1 > 0x3FFFFFFE )
  {
    v13 = *(_QWORD *)(a1 + 32);
    result = 5LL * (*(_DWORD *)(a1 + 40) & 0xFFFFFF);
    for ( i = v13 + 40LL * (*(_DWORD *)(a1 + 40) & 0xFFFFFF); i != v13; v13 += 40 )
    {
      if ( !*(_BYTE *)v13 && a2 == *(_DWORD *)(v13 + 8) )
      {
        v15 = v6;
        result = sub_2EAB140(v13, a3, v6, a5);
        v6 = v15;
      }
    }
  }
  else
  {
    v9 = a3;
    if ( a4 )
      v9 = (unsigned int)sub_E91CF0(a5, a3, a4);
    v10 = *(_QWORD *)(a1 + 32);
    result = 5LL * (*(_DWORD *)(a1 + 40) & 0xFFFFFF);
    for ( j = v10 + 40LL * (*(_DWORD *)(a1 + 40) & 0xFFFFFF); j != v10; v10 += 40 )
    {
      if ( !*(_BYTE *)v10 && a2 == *(_DWORD *)(v10 + 8) )
      {
        result = sub_2EAB1E0(v10, v9, a5);
        v9 = (unsigned int)v9;
      }
    }
  }
  return result;
}
