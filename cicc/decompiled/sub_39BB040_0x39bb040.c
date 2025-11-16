// Function: sub_39BB040
// Address: 0x39bb040
//
__int64 __fastcall sub_39BB040(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // r8
  __int64 v5; // rax
  unsigned int v6; // edi
  __int64 v7; // rdx
  __int64 v8; // rcx

  v3 = a2 - a1;
  v4 = a1;
  v5 = v3 >> 3;
  if ( v3 > 0 )
  {
    v6 = *(_DWORD *)(*(_QWORD *)a3 + 8LL);
    do
    {
      while ( 1 )
      {
        v7 = v5 >> 1;
        v8 = v4 + 8 * (v5 >> 1);
        if ( *(_DWORD *)(*(_QWORD *)v8 + 8LL) >= v6 )
          break;
        v4 = v8 + 8;
        v5 = v5 - v7 - 1;
        if ( v5 <= 0 )
          return v4;
      }
      v5 >>= 1;
    }
    while ( v7 > 0 );
  }
  return v4;
}
