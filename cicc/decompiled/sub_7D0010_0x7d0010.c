// Function: sub_7D0010
// Address: 0x7d0010
//
_QWORD *__fastcall sub_7D0010(__int64 a1, int a2)
{
  char v2; // r13
  __int64 v4; // rdi
  _QWORD *v5; // r12
  __int64 v6; // rdx
  __int64 v8; // rdx
  __int64 v9; // rsi
  _QWORD *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rsi
  char v14; // al
  __int64 v15[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)a1;
  v5 = *(_QWORD **)(v4 + 40);
  if ( v5 )
  {
    while ( 1 )
    {
      if ( (*((_BYTE *)v5 + 84) & 1) != 0 )
      {
        v6 = v5[11];
        if ( (*(_BYTE *)(v6 + 177) & 1) == a2
          && *(_BYTE *)(*(_QWORD *)(v6 + 192) + 80LL) == *(_BYTE *)(a1 + 80)
          && unk_4D03FF0 == sub_880F80(v5)
          && ((*((_BYTE *)v5 + 81) ^ *(_BYTE *)(a1 + 81)) & 0x10) == 0
          && *(_QWORD *)(a1 + 64) == v5[8] )
        {
          return v5;
        }
      }
      v5 = (_QWORD *)v5[1];
      if ( !v5 )
      {
        v4 = *(_QWORD *)a1;
        break;
      }
    }
  }
  v8 = *(_QWORD *)(a1 + 64);
  v9 = 0;
  if ( (*(_BYTE *)(a1 + 81) & 0x10) != 0 )
  {
    v9 = *(_QWORD *)(a1 + 64);
    v8 = 0;
  }
  v10 = (_QWORD *)sub_7CE9E0(v4, v9, v8, v2, v15);
  v11 = v15[0];
  v5 = v10;
  *(_QWORD *)(v15[0] + 192) = a1;
  v12 = *v10;
  v5[1] = *(_QWORD *)(v12 + 40);
  *(_QWORD *)(v12 + 40) = v5;
  if ( *(_BYTE *)(a1 + 80) != 19 )
    return v5;
  v13 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 104LL);
  v14 = *(_BYTE *)(v13 + 90) & 0x10 | *(_BYTE *)(v11 + 90) & 0xEF;
  *(_BYTE *)(v11 + 90) = v14;
  *(_BYTE *)(v11 + 90) = *(_BYTE *)(v13 + 90) & 0x20 | v14 & 0xDF;
  return v5;
}
