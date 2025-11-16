// Function: sub_1360B10
// Address: 0x1360b10
//
__int64 __fastcall sub_1360B10(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned __int64 v4; // rbx
  _QWORD *v5; // r13
  unsigned __int64 v6; // rdi
  char v7; // r15
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v12; // rax
  __int64 v13; // rsi
  _QWORD v14[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v5 = *(_QWORD **)(a1 + 24);
  v6 = (a2 & 0xFFFFFFFFFFFFFFF8LL) + 56;
  v7 = (a2 >> 2) & 1;
  if ( v7 )
  {
    if ( !(unsigned __int8)sub_1560290(v6, a3, 57) )
    {
      v8 = *(_QWORD *)(v4 - 24);
      if ( *(_BYTE *)(v8 + 16) )
      {
LABEL_6:
        if ( *(_BYTE *)(v8 + 16) || !(unsigned __int8)sub_149CB50(*v5, v8, v14) )
          goto LABEL_20;
        goto LABEL_8;
      }
      v14[0] = *(_QWORD *)(v8 + 112);
      if ( !(unsigned __int8)sub_1560290(v14, a3, 57) )
      {
        v8 = *(_QWORD *)(v4 - 24);
        goto LABEL_6;
      }
    }
    return 6;
  }
  if ( (unsigned __int8)sub_1560290(v6, a3, 57) )
    return 6;
  v13 = *(_QWORD *)(v4 - 72);
  if ( !*(_BYTE *)(v13 + 16) )
  {
    v14[0] = *(_QWORD *)(v13 + 112);
    if ( (unsigned __int8)sub_1560290(v14, a3, 57) )
      return 6;
    v13 = *(_QWORD *)(v4 - 72);
  }
  if ( *(_BYTE *)(v13 + 16) || !(unsigned __int8)sub_149CB50(*v5, v13, v14) )
    goto LABEL_10;
LABEL_8:
  if ( LODWORD(v14[0]) == 295 && (*(_BYTE *)(*v5 + 73LL) & 0xC0) != 0 && !a3 )
    return 6;
  if ( !v7 )
  {
LABEL_10:
    if ( !(unsigned __int8)sub_1560290(v4 + 56, a3, 37) )
    {
      v9 = *(_QWORD *)(v4 - 72);
      if ( *(_BYTE *)(v9 + 16) || (v14[0] = *(_QWORD *)(v9 + 112), !(unsigned __int8)sub_1560290(v14, a3, 37)) )
      {
        if ( !(unsigned __int8)sub_1560290(v4 + 56, a3, 36) )
        {
          v10 = *(_QWORD *)(v4 - 72);
          if ( *(_BYTE *)(v10 + 16) )
            return 7;
          goto LABEL_15;
        }
        return 4;
      }
    }
    return 5;
  }
LABEL_20:
  if ( (unsigned __int8)sub_1560290(v4 + 56, a3, 37) )
    return 5;
  v12 = *(_QWORD *)(v4 - 24);
  if ( !*(_BYTE *)(v12 + 16) )
  {
    v14[0] = *(_QWORD *)(v12 + 112);
    if ( (unsigned __int8)sub_1560290(v14, a3, 37) )
      return 5;
  }
  if ( !(unsigned __int8)sub_1560290(v4 + 56, a3, 36) )
  {
    v10 = *(_QWORD *)(v4 - 24);
    if ( *(_BYTE *)(v10 + 16) )
      return 7;
LABEL_15:
    v14[0] = *(_QWORD *)(v10 + 112);
    if ( !(unsigned __int8)sub_1560290(v14, a3, 36) )
      return 7;
  }
  return 4;
}
