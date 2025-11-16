// Function: sub_2CBA650
// Address: 0x2cba650
//
__int64 __fastcall sub_2CBA650(__int64 a1, unsigned __int64 a2)
{
  char v3; // al
  char v6; // al
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // rdi
  _QWORD *v13; // rsi
  __int64 v14; // [rsp-20h] [rbp-20h]

  if ( !*(_QWORD *)(a2 + 16) )
    return 0;
  v3 = *(_BYTE *)(a2 + 32) & 0xF;
  if ( ((v3 + 14) & 0xFu) <= 3 )
    return 0;
  if ( ((v3 + 7) & 0xFu) <= 1 )
    return 0;
  v6 = sub_B2DDD0(a2, 0, 0, 1, 0, 0, 0);
  v7 = v14;
  if ( v6 )
    return 0;
  v8 = *(_QWORD *)(a2 + 24);
  v9 = *(_DWORD *)(v8 + 8) >> 8;
  if ( (_DWORD)v9 )
    return 0;
  v10 = **(_QWORD **)(v8 + 16);
  if ( *(_BYTE *)(v10 + 8) != 14 || *(_DWORD *)(v10 + 8) >> 8 )
    return sub_2CBA520(a1, a2, v9, v7);
  v11 = *(_QWORD **)(a1 + 24);
  v12 = a1 + 16;
  if ( !v11 )
    return 1;
  v13 = (_QWORD *)(a1 + 16);
  do
  {
    v7 = v11[2];
    v9 = v11[3];
    if ( v11[4] < a2 )
    {
      v11 = (_QWORD *)v11[3];
    }
    else
    {
      v13 = v11;
      v11 = (_QWORD *)v11[2];
    }
  }
  while ( v11 );
  if ( (_QWORD *)v12 != v13 && v13[4] <= a2 )
    return sub_2CBA520(a1, a2, v9, v7);
  else
    return 1;
}
