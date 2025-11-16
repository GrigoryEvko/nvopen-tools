// Function: sub_27D81A0
// Address: 0x27d81a0
//
__int64 __fastcall sub_27D81A0(
        __int64 a1,
        char *a2,
        __int64 (__fastcall *a3)(__int64, __int64, _QWORD, _QWORD),
        __int64 a4)
{
  char v4; // al
  unsigned int v5; // r8d
  __int64 v10; // r15
  unsigned __int16 v11; // cx
  __int64 v12; // rsi
  unsigned __int64 v13; // rax
  unsigned int v14; // r13d
  unsigned int v15; // eax
  unsigned __int8 v16; // al
  unsigned __int64 v17; // rax

  v4 = *a2;
  v5 = 0;
  if ( (unsigned __int8)*a2 > 0x1Cu )
  {
    if ( v4 == 61 )
    {
      v10 = *((_QWORD *)a2 - 4);
      if ( !v10 )
        return v5;
      v11 = *((_WORD *)a2 + 1);
      v12 = *((_QWORD *)a2 + 1);
      _BitScanReverse64(&v13, 1LL << (v11 >> 1));
      v14 = 63 - (v13 ^ 0x3F);
    }
    else
    {
      if ( v4 != 62 )
        return v5;
      v10 = *((_QWORD *)a2 - 4);
      if ( !v10 )
        return v5;
      _BitScanReverse64(&v17, 1LL << (*((_WORD *)a2 + 1) >> 1));
      v14 = 63 - (v17 ^ 0x3F);
      v12 = *(_QWORD *)(*((_QWORD *)a2 - 8) + 8LL);
    }
    v15 = sub_AE5260(a1, v12);
    v16 = a3(a4, v10, v14, v15);
    v5 = 0;
    if ( (unsigned __int8)v14 < v16 )
    {
      v5 = 1;
      *((_WORD *)a2 + 1) = *((_WORD *)a2 + 1) & 0xFF81 | (2 * v16);
    }
    return v5;
  }
  return 0;
}
