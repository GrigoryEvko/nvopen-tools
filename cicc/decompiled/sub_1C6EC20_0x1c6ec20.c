// Function: sub_1C6EC20
// Address: 0x1c6ec20
//
__int64 __fastcall sub_1C6EC20(__int64 a1, unsigned __int64 a2)
{
  char v3; // al
  __int64 v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // rax
  __int64 v9; // rdi
  _QWORD *v10; // rsi

  if ( !*(_QWORD *)(a2 + 8) )
    return 0;
  v3 = *(_BYTE *)(a2 + 32) & 0xF;
  if ( ((v3 + 14) & 0xFu) <= 3 )
    return 0;
  if ( ((v3 + 7) & 0xFu) <= 1 )
    return 0;
  if ( (unsigned __int8)sub_15E3650(a2, 0) )
    return 0;
  v6 = *(_QWORD *)(a2 + 24);
  if ( *(_DWORD *)(v6 + 8) >> 8 )
    return 0;
  v7 = **(_QWORD **)(v6 + 16);
  if ( *(_BYTE *)(v7 + 8) != 15 || *(_DWORD *)(v7 + 8) >> 8 )
    return sub_1C6EAF0(a1, a2);
  v8 = *(_QWORD **)(a1 + 24);
  v9 = a1 + 16;
  if ( !v8 )
    return 1;
  v10 = (_QWORD *)(a1 + 16);
  do
  {
    if ( v8[4] < a2 )
    {
      v8 = (_QWORD *)v8[3];
    }
    else
    {
      v10 = v8;
      v8 = (_QWORD *)v8[2];
    }
  }
  while ( v8 );
  if ( (_QWORD *)v9 != v10 && v10[4] <= a2 )
    return sub_1C6EAF0(a1, a2);
  else
    return 1;
}
