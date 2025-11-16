// Function: sub_22A79B0
// Address: 0x22a79b0
//
__int64 __fastcall sub_22A79B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // r14
  __int64 v5; // rbx
  __int64 v7; // r12
  __int64 v8; // r13
  unsigned __int8 v9; // al
  unsigned int v11; // edx
  unsigned int v12; // eax
  unsigned int v13; // ecx
  unsigned int v14; // ecx
  unsigned int v15; // eax
  unsigned int v16; // eax

  v3 = a2 - a1;
  v4 = a1;
  v5 = 0x6DB6DB6DB6DB6DB7LL * (v3 >> 3);
  if ( v3 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v7 = v5 >> 1;
        v8 = v4 + 56 * (v5 >> 1);
        v9 = *(_BYTE *)(v8 + 10);
        if ( *(_BYTE *)(a3 + 10) >= v9 )
          break;
LABEL_6:
        v5 >>= 1;
        if ( v7 <= 0 )
          return v4;
      }
      if ( *(_BYTE *)(a3 + 10) == v9 )
      {
        v11 = *(_DWORD *)(a3 + 16);
        v12 = *(_DWORD *)(v8 + 16);
        if ( v11 < v12 )
          goto LABEL_6;
        if ( v11 == v12 )
        {
          v13 = *(_DWORD *)(v8 + 20);
          if ( *(_DWORD *)(a3 + 20) < v13 )
            goto LABEL_6;
          if ( *(_DWORD *)(a3 + 20) == v13 )
          {
            v14 = *(_DWORD *)(v8 + 24);
            if ( *(_DWORD *)(a3 + 24) < v14
              || *(_DWORD *)(a3 + 24) == v14 && *(_DWORD *)(a3 + 28) < *(_DWORD *)(v8 + 28) )
            {
              goto LABEL_6;
            }
          }
        }
        if ( v11 <= v12 )
        {
          v15 = *(_DWORD *)(a3 + 20);
          if ( *(_DWORD *)(v8 + 20) >= v15 )
          {
            if ( *(_DWORD *)(v8 + 20) != v15
              || (v16 = *(_DWORD *)(a3 + 24), *(_DWORD *)(v8 + 24) >= v16)
              && (*(_DWORD *)(v8 + 24) != v16 || *(_DWORD *)(v8 + 28) >= *(_DWORD *)(a3 + 28)) )
            {
              if ( (unsigned __int8)sub_22A6F20(a3, (__int64 *)(v4 + 56 * (v5 >> 1))) )
                goto LABEL_6;
              sub_22A6F20(v4 + 56 * (v5 >> 1), (__int64 *)a3);
            }
          }
        }
      }
      v4 = v8 + 56;
      v5 = v5 - v7 - 1;
    }
    while ( v5 > 0 );
  }
  return v4;
}
