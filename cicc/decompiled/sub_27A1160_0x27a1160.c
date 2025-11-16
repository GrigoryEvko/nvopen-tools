// Function: sub_27A1160
// Address: 0x27a1160
//
__int64 __fastcall sub_27A1160(__int64 a1, __int64 a2, unsigned int *a3)
{
  __int64 v5; // rsi
  __int64 v6; // rdx
  unsigned int v7; // esi
  __int64 v8; // rcx
  __int64 v9; // rax

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
        if ( v7 < *(_DWORD *)v9 || v7 == *(_DWORD *)v9 && *((_QWORD *)a3 + 1) < *(_QWORD *)(v9 + 8) )
          break;
        a1 = v9 + 32;
        v6 = v6 - v8 - 1;
        if ( v6 <= 0 )
          return a1;
      }
      v6 >>= 1;
    }
    while ( v8 > 0 );
  }
  return a1;
}
