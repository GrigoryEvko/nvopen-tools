// Function: sub_2AF69D0
// Address: 0x2af69d0
//
__int64 __fastcall sub_2AF69D0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v5; // rax
  int v6; // r13d
  __int64 v7; // rsi
  __int64 v8; // rdx
  _QWORD *v9; // rax
  unsigned int v10; // eax
  unsigned int v11; // r8d
  unsigned __int64 v12; // rdx
  _QWORD *v13; // rdi
  int v14; // esi
  unsigned int v15; // eax

  if ( (unsigned __int8)(*(_BYTE *)a2 - 61) <= 1u )
  {
    _BitScanReverse64(&v2, 1LL << (*(_WORD *)(a2 + 2) >> 1));
    return (unsigned int)(63 - (v2 ^ 0x3F));
  }
  if ( *(_BYTE *)a2 != 85 )
    return (unsigned int)-1;
  v5 = *(_QWORD *)(a2 - 32);
  if ( !v5 || *(_BYTE *)v5 || *(_QWORD *)(v5 + 24) != *(_QWORD *)(a2 + 80) )
    BUG();
  v6 = *(_DWORD *)(v5 + 36);
  if ( v6 != 9567 && v6 != 8975 )
  {
    if ( v6 == 8937 )
    {
      v13 = (_QWORD *)(a2 + 72);
      v14 = 1;
    }
    else
    {
      if ( v6 != 9549 )
        goto LABEL_12;
      v13 = (_QWORD *)(a2 + 72);
      v14 = 2;
    }
    LOWORD(v15) = sub_A74840(v13, v14);
    v11 = 0;
    if ( BYTE1(v15) )
      return v15;
    return v11;
  }
  v8 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v9 = *(_QWORD **)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v9 = (_QWORD *)*v9;
  v10 = (int)pow(2.0, (double)(int)((((unsigned int)v9 >> 13) & 0x1F) - 1));
  if ( v10 )
  {
    _BitScanReverse64(&v12, v10);
    return 63 - ((unsigned int)v12 ^ 0x3F);
  }
  if ( v6 == 8975 )
  {
    v7 = *(_QWORD *)(a2 + 8);
    return (unsigned int)sub_AE5020(*(_QWORD *)(a1 + 48), v7);
  }
LABEL_12:
  v7 = *(_QWORD *)(*(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 8LL);
  return (unsigned int)sub_AE5020(*(_QWORD *)(a1 + 48), v7);
}
