// Function: sub_27A11C0
// Address: 0x27a11c0
//
__int64 __fastcall sub_27A11C0(__int64 a1, __int64 a2, unsigned int *a3)
{
  __int64 v5; // rsi
  __int64 v6; // rax
  unsigned int v7; // esi
  __int64 v8; // rcx
  __int64 v9; // rdx

  v5 = a2 - a1;
  v6 = v5 >> 5;
  if ( v5 > 0 )
  {
    v7 = *a3;
    do
    {
      while ( 1 )
      {
        v8 = v6 >> 1;
        v9 = a1 + 32 * (v6 >> 1);
        if ( *(_DWORD *)v9 < v7 || *(_DWORD *)v9 == v7 && *(_QWORD *)(v9 + 8) < *((_QWORD *)a3 + 1) )
          break;
        v6 >>= 1;
        if ( v8 <= 0 )
          return a1;
      }
      a1 = v9 + 32;
      v6 = v6 - v8 - 1;
    }
    while ( v6 > 0 );
  }
  return a1;
}
