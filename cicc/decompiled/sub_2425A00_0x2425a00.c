// Function: sub_2425A00
// Address: 0x2425a00
//
__int64 __fastcall sub_2425A00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // r8
  __int64 v5; // rax
  unsigned int v6; // edi
  __int64 v7; // rcx

  v3 = a2 - a1;
  v4 = a1;
  v5 = v3 >> 3;
  if ( v3 > 0 )
  {
    v6 = *(_DWORD *)(*(_QWORD *)a3 + 32LL);
    do
    {
      v7 = v4 + 8 * (v5 >> 1);
      if ( *(_DWORD *)(*(_QWORD *)v7 + 32LL) == v6 )
      {
        if ( *(_DWORD *)(*(_QWORD *)v7 + 36LL) >= *(_DWORD *)(*(_QWORD *)a3 + 36LL) )
        {
LABEL_9:
          v5 >>= 1;
          continue;
        }
      }
      else if ( *(_DWORD *)(*(_QWORD *)v7 + 32LL) >= v6 )
      {
        goto LABEL_9;
      }
      v4 = v7 + 8;
      v5 = v5 - (v5 >> 1) - 1;
    }
    while ( v5 > 0 );
  }
  return v4;
}
