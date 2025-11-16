// Function: sub_1D168E0
// Address: 0x1d168e0
//
__int64 __fastcall sub_1D168E0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rsi

  result = 0;
  if ( *(_WORD *)(a1 + 24) == 104 )
  {
    v2 = *(_QWORD *)(a1 + 32);
    v3 = v2 + 40LL * *(unsigned int *)(a1 + 56);
    if ( v2 == v3 )
    {
      return 1;
    }
    else
    {
      do
      {
        result = *(_WORD *)(*(_QWORD *)v2 + 24LL) & 0xFFEF;
        LOBYTE(result) = *(_WORD *)(*(_QWORD *)v2 + 24LL) == 10 || (*(_WORD *)(*(_QWORD *)v2 + 24LL) & 0xFFEF) == 32;
        if ( !(_BYTE)result )
          break;
        v2 += 40;
      }
      while ( v3 != v2 );
    }
  }
  return result;
}
