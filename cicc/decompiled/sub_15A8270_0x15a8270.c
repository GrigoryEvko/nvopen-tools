// Function: sub_15A8270
// Address: 0x15a8270
//
_DWORD *__fastcall sub_15A8270(__int64 a1, unsigned int a2, unsigned int a3)
{
  _DWORD *v6; // r8
  __int64 v7; // rax
  __int64 v8; // rdx
  _DWORD *v9; // rcx
  unsigned int v10; // esi

  v6 = *(_DWORD **)(a1 + 48);
  v7 = *(unsigned int *)(a1 + 56);
  if ( v7 )
  {
    do
    {
      while ( 1 )
      {
        v8 = v7 >> 1;
        v9 = &v6[2 * (v7 >> 1)];
        v10 = *(unsigned __int8 *)v9;
        if ( a2 > v10 || a2 == v10 && a3 > *v9 >> 8 )
          break;
        v7 >>= 1;
        if ( v8 <= 0 )
          return v6;
      }
      v6 = v9 + 2;
      v7 = v7 - v8 - 1;
    }
    while ( v7 > 0 );
  }
  return v6;
}
