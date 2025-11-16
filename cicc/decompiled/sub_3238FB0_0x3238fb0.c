// Function: sub_3238FB0
// Address: 0x3238fb0
//
__int64 __fastcall sub_3238FB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v5; // r13
  __int64 v6; // rbx
  unsigned __int8 v7; // al
  __int64 v8; // rsi
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 result; // rax
  unsigned __int8 v12; // al
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r13

  v3 = a2;
  v5 = *(_QWORD *)(a3 + 8);
  v6 = v5 - 16;
  if ( !*(_BYTE *)(a1 + 3769) || (unsigned __int8)sub_321F6A0(a1, a2) )
  {
    v12 = *(_BYTE *)(v5 - 16);
    if ( (v12 & 2) != 0 )
      v8 = *(_QWORD *)(*(_QWORD *)(v5 - 32) + 40LL);
    else
      v8 = *(_QWORD *)(v6 - 8LL * ((v12 >> 2) & 0xF) + 40);
  }
  else
  {
    v7 = *(_BYTE *)(v5 - 16);
    if ( (v7 & 2) != 0 )
    {
      v8 = *(_QWORD *)(*(_QWORD *)(v5 - 32) + 40LL);
      if ( !*(_BYTE *)(v8 + 41) )
      {
LABEL_5:
        v9 = a3;
        v10 = v3;
        return sub_37404B0(v10, v9);
      }
    }
    else
    {
      v8 = *(_QWORD *)(v6 - 8LL * ((v7 >> 2) & 0xF) + 40);
      if ( !*(_BYTE *)(v8 + 41) )
        goto LABEL_5;
    }
  }
  v13 = sub_3238860(a1, v8);
  v14 = *(_QWORD *)(v13 + 408);
  v15 = v13;
  if ( !v14 )
    return sub_37404B0(v13, a3);
  if ( (unsigned __int8)sub_321F6A0(a1, v8) )
    v3 = v15;
  sub_37404B0(v3, a3);
  result = *(_QWORD *)(v15 + 80);
  v9 = a3;
  v10 = v14;
  if ( *(_BYTE *)(result + 41) )
    return sub_37404B0(v10, v9);
  return result;
}
