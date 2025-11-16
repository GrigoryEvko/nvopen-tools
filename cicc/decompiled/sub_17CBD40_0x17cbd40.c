// Function: sub_17CBD40
// Address: 0x17cbd40
//
__int64 __fastcall sub_17CBD40(
        _QWORD *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rax
  char v11; // r8
  __int64 v12; // r13
  __int64 v13; // r14
  __int64 v14; // rbx
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rax
  unsigned __int8 v21; // [rsp+7h] [rbp-39h]

  v10 = a1[26];
  if ( a1[27] != v10 )
    a1[27] = v10;
  v11 = 0;
  v12 = *(_QWORD *)(a2 + 80);
  v13 = a2 + 72;
  if ( v12 == a2 + 72 )
    return 0;
  do
  {
    if ( !v12 )
      BUG();
    v14 = *(_QWORD *)(v12 + 24);
    v15 = v12 + 16;
LABEL_6:
    while ( v15 != v14 )
    {
      while ( 1 )
      {
        v16 = v14;
        v14 = *(_QWORD *)(v14 + 8);
        v17 = v16 - 24;
        if ( *(_BYTE *)(v16 - 8) != 78 )
          break;
        v18 = *(_QWORD *)(v16 - 48);
        if ( !*(_BYTE *)(v18 + 16)
          && (*(_BYTE *)(v18 + 33) & 0x20) != 0
          && (*(_DWORD *)(v18 + 36) == 111 || (*(_BYTE *)(v18 + 33) & 0x20) != 0 && *(_DWORD *)(v18 + 36) == 110) )
        {
          sub_17CB560((__int64)a1, v17, a3, a4, a5, a6, a7, a8, a9, a10);
          v11 = 1;
          goto LABEL_6;
        }
        if ( *(_BYTE *)(v18 + 16) || (*(_BYTE *)(v18 + 33) & 0x20) == 0 || *(_DWORD *)(v18 + 36) != 112 )
          goto LABEL_6;
        sub_17C5C60((__int64)a1, v17, a3, a4, a5, a6, a7, a8, a9, a10);
        v11 = 1;
        if ( v15 == v14 )
          goto LABEL_14;
      }
    }
LABEL_14:
    v12 = *(_QWORD *)(v12 + 8);
  }
  while ( v13 != v12 );
  v21 = v11;
  if ( !v11 )
  {
    return 0;
  }
  else
  {
    sub_17C9AE0(a1, a2);
    return v21;
  }
}
