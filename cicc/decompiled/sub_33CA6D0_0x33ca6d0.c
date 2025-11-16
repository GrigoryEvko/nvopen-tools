// Function: sub_33CA6D0
// Address: 0x33ca6d0
//
__int64 __fastcall sub_33CA6D0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rsi

  result = 0;
  if ( *(_DWORD *)(a1 + 24) == 156 )
  {
    v2 = *(_QWORD *)(a1 + 40);
    v3 = v2 + 40LL * *(unsigned int *)(a1 + 64);
    if ( v2 == v3 )
    {
      return 1;
    }
    else
    {
      do
      {
        result = *(_DWORD *)(*(_QWORD *)v2 + 24LL) & 0xFFFFFFEF;
        LOBYTE(result) = *(_DWORD *)(*(_QWORD *)v2 + 24LL) == 11 || (_DWORD)result == 35;
        if ( !(_BYTE)result )
          break;
        v2 += 40;
      }
      while ( v3 != v2 );
    }
  }
  return result;
}
