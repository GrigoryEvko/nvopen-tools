// Function: sub_266E140
// Address: 0x266e140
//
__int64 __fastcall sub_266E140(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rcx
  __int64 v5; // rsi
  _QWORD *v6; // r12
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rcx

  if ( !a1 )
    return 0;
  result = 0;
  if ( a2 == **(_QWORD **)(*(_QWORD *)(a1 + 24) + 16LL) )
  {
    v4 = *(_QWORD *)(a1 + 104);
    v5 = *(unsigned int *)(a3 + 8);
    if ( v4 == v5 )
    {
      v6 = *(_QWORD **)a3;
      if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
      {
        sub_B2C6D0(a1, v5, a3, v4);
        v7 = *(_QWORD *)(a1 + 96);
        v8 = v7 + 40LL * *(_QWORD *)(a1 + 104);
        if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
        {
          sub_B2C6D0(a1, v5, 5LL * *(_QWORD *)(a1 + 104), v9);
          v7 = *(_QWORD *)(a1 + 96);
        }
      }
      else
      {
        v7 = *(_QWORD *)(a1 + 96);
        v8 = v7 + 40 * v4;
      }
      while ( 1 )
      {
        if ( v7 == v8 )
          return 1;
        if ( *(_QWORD *)(v7 + 8) != *v6 )
          break;
        ++v6;
        v7 += 40;
      }
      return 0;
    }
  }
  return result;
}
