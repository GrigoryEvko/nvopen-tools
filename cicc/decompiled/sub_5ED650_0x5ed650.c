// Function: sub_5ED650
// Address: 0x5ed650
//
__int64 __fastcall sub_5ED650(_QWORD *a1, _QWORD *a2, _QWORD *a3, _QWORD *a4)
{
  __int64 v4; // r8
  __int64 v6; // rsi
  unsigned int v8; // r11d
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rax

  v4 = a1[2];
  v6 = a3[2];
  v8 = 0;
  if ( ((*(_BYTE *)(v6 + 96) ^ *(_BYTE *)(v4 + 96)) & 2) == 0 )
  {
    v8 = dword_4F07588;
    while ( 1 )
    {
      v9 = *(_QWORD *)(v4 + 40);
      v10 = *(_QWORD *)(v6 + 40);
      if ( v9 != v10 )
      {
        if ( !v9 || !v10 )
          return 0;
        if ( !dword_4F07588 )
          return v8;
        v11 = *(_QWORD *)(v9 + 32);
        if ( *(_QWORD *)(v10 + 32) != v11 || !v11 )
          return 0;
      }
      if ( a2 == a1 )
        break;
      if ( a3 == a4 )
        return 0;
      a1 = (_QWORD *)*a1;
      a3 = (_QWORD *)*a3;
      v4 = a1[2];
      v6 = a3[2];
    }
    v8 = 0;
    if ( a3 == a4 )
      return (((unsigned __int8)(*(_BYTE *)(v6 + 96) ^ *(_BYTE *)(v4 + 96)) >> 1) ^ 1) & 1;
  }
  return v8;
}
