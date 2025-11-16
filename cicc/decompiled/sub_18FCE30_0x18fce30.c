// Function: sub_18FCE30
// Address: 0x18fce30
//
unsigned __int64 __fastcall sub_18FCE30(__int64 a1)
{
  unsigned __int8 v1; // cl
  unsigned int v2; // r12d
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned __int8 v9; // cl
  _QWORD v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 == 78 )
  {
    v2 = sub_1560260((_QWORD *)(a1 + 56), -1, 36);
    if ( (_BYTE)v2 )
      goto LABEL_3;
    if ( *(char *)(a1 + 23) >= 0 )
      goto LABEL_17;
    v4 = sub_1648A40(a1);
    v6 = v4 + v5;
    v7 = 0;
    if ( *(char *)(a1 + 23) < 0 )
      v7 = sub_1648A40(a1);
    if ( !(unsigned int)((v6 - v7) >> 4) )
    {
LABEL_17:
      v8 = *(_QWORD *)(a1 - 24);
      if ( !*(_BYTE *)(v8 + 16) )
      {
        v10[0] = *(_QWORD *)(v8 + 112);
        if ( (unsigned __int8)sub_1560260(v10, -1, 36) )
LABEL_3:
          LOBYTE(v2) = *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 0;
      }
    }
    return v2;
  }
  v2 = 1;
  if ( (unsigned int)v1 - 60 <= 0xC )
    return v2;
  v9 = v1 - 35;
  v2 = 0;
  if ( v9 > 0x34u )
    return v2;
  return (0x1F13000023FFFFuLL >> v9) & 1;
}
