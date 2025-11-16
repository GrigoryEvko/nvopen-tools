// Function: sub_777CA0
// Address: 0x777ca0
//
_BOOL8 __fastcall sub_777CA0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 v10; // rax
  char i; // dl
  FILE *v13; // r13
  unsigned __int64 v14; // rax
  unsigned int v15; // edx
  __int64 v16; // rax
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rsi
  _QWORD *v21; // r8
  int v22; // [rsp+8h] [rbp-28h] BYREF
  int v23[9]; // [rsp+Ch] [rbp-24h] BYREF

  v7 = *(_QWORD *)(a3 + 72);
  if ( v7 && (v8 = *(_QWORD *)(v7 + 16)) != 0 && (v9 = *(__int64 **)(v8 + 16)) != 0 )
  {
    v10 = *v9;
    for ( i = *(_BYTE *)(v10 + 140); i == 12; i = *(_BYTE *)(v10 + 140) )
      v10 = *(_QWORD *)(v10 + 160);
    v13 = (FILE *)(a3 + 28);
    if ( i == 2 )
    {
      v14 = sub_620EE0(*(_WORD **)(a4 + 8), byte_4B6DF90[*(unsigned __int8 *)(v10 + 160)], &v22);
      v15 = v14;
      if ( v22 || v14 > 0xFFFFFF )
      {
        if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
        {
          sub_67E440(0xBADu, v13, v14, (_QWORD *)(a1 + 96));
          sub_770D30(a1);
        }
        return 0;
      }
      v16 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2[5] + 32) + 168LL) + 168LL);
      if ( v16 && !*(_BYTE *)(v16 + 8) )
        return sub_777910(a1, *(_QWORD *)(v16 + 32), v15, 1, v13, a5, v23) != 0;
    }
    if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
      return 0;
    v18 = a2[19];
    v19 = *a2;
    v21 = (_QWORD *)(a1 + 96);
    v20 = a3 + 28;
  }
  else
  {
    if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
      return 0;
    v18 = a2[19];
    v19 = *a2;
    v20 = a3 + 28;
    v21 = (_QWORD *)(a1 + 96);
  }
  sub_687670(0xBB4u, v20, v19, v18, v21);
  sub_770D30(a1);
  return 0;
}
