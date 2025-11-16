// Function: sub_2AF6B80
// Address: 0x2af6b80
//
_QWORD *__fastcall sub_2AF6B80(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned int v7; // r9d
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v13; // rdx
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax

  v7 = a4 >> 3;
  LODWORD(v8) = a3 * (a4 >> 3);
  LODWORD(v9) = v8 & 0xFFFFFFFC;
  if ( (((_BYTE)a3 * (_BYTE)v7) & 3) == 0 || !(_DWORD)v9 )
  {
    LODWORD(v9) = 0;
    if ( !(_DWORD)v8 )
      goto LABEL_4;
    v13 = (unsigned int)v8;
    LODWORD(v8) = 0;
    v14 = v13 - 1;
    if ( !v14 )
      goto LABEL_4;
    _BitScanReverse64(&v15, v14);
    v9 = (unsigned __int64)(1LL << (64 - ((unsigned __int8)v15 ^ 0x3Fu))) >> 1;
  }
  LODWORD(v8) = (unsigned int)v9 / v7;
LABEL_4:
  if ( v7 > (unsigned int)v9 )
  {
    v10 = 8;
    v8 = 1;
  }
  else
  {
    v8 = (unsigned int)v8;
    v10 = 8LL * (unsigned int)v8;
  }
  v11 = a3 - v8;
  *a1 = a2;
  a1[1] = v8;
  a1[2] = v10 + a2;
  a1[3] = v11;
  return a1;
}
