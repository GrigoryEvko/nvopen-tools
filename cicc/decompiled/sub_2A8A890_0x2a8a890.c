// Function: sub_2A8A890
// Address: 0x2a8a890
//
unsigned __int16 __fastcall sub_2A8A890(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // rax
  __int16 v3; // r12
  unsigned int v4; // eax
  __int64 v5; // rdx
  _QWORD *v6; // rax
  unsigned int v7; // eax
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rsi
  unsigned __int16 result; // ax
  unsigned __int64 v12; // rdx

  v2 = *(unsigned __int8 **)(a2 - 32);
  if ( !v2 )
    goto LABEL_27;
  v3 = *v2;
  if ( (_BYTE)v3 || *((_QWORD *)v2 + 3) != *(_QWORD *)(a2 + 80) )
    goto LABEL_27;
  v4 = *((_DWORD *)v2 + 9);
  if ( v4 == 8975 )
  {
LABEL_8:
    v5 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v6 = *(_QWORD **)(v5 + 24);
    if ( *(_DWORD *)(v5 + 32) > 0x40u )
      v6 = (_QWORD *)*v6;
    v7 = (int)pow(2.0, (double)(int)((((unsigned int)v6 >> 13) & 0x1F) - 1));
    if ( v7 )
    {
      _BitScanReverse64(&v12, v7);
      return 63 - (v12 ^ 0x3F);
    }
    v8 = *(_QWORD *)(a2 - 32);
    if ( v8 && !*(_BYTE *)v8 && *(_QWORD *)(v8 + 24) == *(_QWORD *)(a2 + 80) )
    {
      v9 = *(_QWORD *)(a1 + 48);
      if ( *(_DWORD *)(v8 + 36) == 8975 )
        v10 = *(_QWORD *)(a2 + 8);
      else
        v10 = *(_QWORD *)(*(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 8LL);
      return sub_AE5020(v9, v10);
    }
LABEL_27:
    BUG();
  }
  if ( v4 <= 0x230F )
  {
    if ( v4 - 8937 > 1 )
      goto LABEL_27;
    result = sub_A74840((_QWORD *)(a2 + 72), 1);
    if ( !HIBYTE(result) )
      return v3;
  }
  else
  {
    if ( v4 != 9553 )
    {
      if ( v4 == 9567 )
        goto LABEL_8;
      if ( v4 != 9549 )
        goto LABEL_27;
    }
    result = sub_A74840((_QWORD *)(a2 + 72), 2);
    if ( !HIBYTE(result) )
      return v3;
  }
  return result;
}
