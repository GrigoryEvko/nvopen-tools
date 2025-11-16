// Function: sub_147AD20
// Address: 0x147ad20
//
__int64 __fastcall sub_147AD20(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  __int64 v8; // rax
  __int64 v9; // rax

  if ( a2 != 36 )
    return 0;
  v4 = *(unsigned __int8 *)(a1 + 489);
  if ( (_BYTE)v4 )
    return 0;
  *(_BYTE *)(a1 + 489) = 1;
  if ( (unsigned __int8)sub_1477BC0(a1, a4) )
  {
    v8 = sub_1456040(a3);
    v9 = sub_145CF80(a1, v8, 0, 0);
    if ( (unsigned __int8)sub_147A340(a1, 0x27u, a3, v9) )
      v4 = sub_147A340(a1, 0x28u, a3, a4);
  }
  *(_BYTE *)(a1 + 489) = 0;
  return v4;
}
