// Function: sub_8EEDF0
// Address: 0x8eedf0
//
__int64 __fastcall sub_8EEDF0(__int64 a1, __int64 a2)
{
  __int64 v2; // r10
  unsigned int v3; // r11d
  unsigned int v5; // edi
  unsigned int v6; // esi
  __int64 v7; // r8
  unsigned int v8; // eax
  unsigned int v9; // r9d
  unsigned int v10; // ecx
  signed int v11; // eax
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rsi
  __int64 v16; // rax
  _DWORD *v17; // rax
  __int64 v18; // rsi
  __int64 v19; // r10

  v2 = a1;
  v3 = 0;
  v5 = *(_DWORD *)(a1 + 2088);
  v6 = *(_DWORD *)(a2 + 2088);
  if ( v5 < v6 )
    return v3;
  v7 = v5 - 1;
  v8 = *(_DWORD *)(v2 + 4 * v7 + 8);
  v9 = *(_DWORD *)(a2 + 4LL * (v6 - 1) + 8);
  v10 = v5 - 1;
  if ( v5 == v6 )
  {
    if ( v8 < v9 )
      return v3;
    v11 = v8 / (v9 + 1);
    if ( !v11 )
      goto LABEL_9;
  }
  else
  {
    v11 = v8 / (v9 + 1);
    if ( !v11 )
      goto LABEL_9;
  }
  if ( v6 )
  {
    v12 = v11;
    v13 = 0;
    v14 = 0;
    do
    {
      v15 = 0;
      v16 = v14 + *(unsigned int *)(v2 + 4 * v13 + 8) - v12 * *(unsigned int *)(a2 + 4 * v13 + 8);
      if ( v16 < 0 )
      {
        v16 += 0x100000000LL;
        v15 = -1;
      }
      *(_DWORD *)(v2 + 4 * v13++ + 8) = v16;
      v14 = (v16 >> 32) + v15;
    }
    while ( *(_DWORD *)(a2 + 2088) > (unsigned int)v13 );
  }
LABEL_9:
  v17 = (_DWORD *)(v2 + 4 * v7 + 8);
  if ( v5 )
  {
    while ( !*v17 )
    {
      *(_DWORD *)(v2 + 2088) = v10;
      --v17;
      if ( !v10 )
        goto LABEL_16;
      --v10;
    }
    v18 = a2;
    if ( (int)sub_8EECF0(v2, a2) >= 0 )
      goto LABEL_15;
  }
  else
  {
LABEL_16:
    while ( 1 )
    {
      v18 = a2;
      if ( (int)sub_8EECF0(v2, a2) < 0 )
        break;
LABEL_15:
      sub_8EED40(v19, v18);
    }
  }
  return v3;
}
