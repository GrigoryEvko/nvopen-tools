// Function: sub_29584C0
// Address: 0x29584c0
//
__int64 __fastcall sub_29584C0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  int v6; // edi
  __int64 v7; // rax
  __int64 v8; // rsi

  result = sub_AA5930(a1);
  if ( result != v4 )
  {
    v5 = result;
    do
    {
      v6 = *(_DWORD *)(v5 + 4) & 0x7FFFFFF;
      if ( v6 )
      {
        v7 = 0;
        do
        {
          v8 = (unsigned int)v7++;
          *(_QWORD *)(*(_QWORD *)(v5 - 8) + 32LL * *(unsigned int *)(v5 + 72) + 8 * v8) = a2;
        }
        while ( v7 != v6 );
      }
      result = *(_QWORD *)(v5 + 32);
      if ( !result )
        BUG();
      v5 = 0;
      if ( *(_BYTE *)(result - 24) == 84 )
        v5 = result - 24;
    }
    while ( v4 != v5 );
  }
  return result;
}
