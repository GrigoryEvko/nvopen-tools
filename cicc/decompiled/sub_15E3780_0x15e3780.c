// Function: sub_15E3780
// Address: 0x15e3780
//
__int64 __fastcall sub_15E3780(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // rbx
  __int64 i; // r13
  unsigned __int8 v5; // dl
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r15
  _QWORD *v9; // rdi
  __int64 v10; // rax
  _QWORD v11[7]; // [rsp+8h] [rbp-38h] BYREF

  v1 = a1 + 72;
  v2 = *(_QWORD *)(a1 + 80);
  if ( a1 + 72 == v2 )
    return 0;
  if ( !v2 )
    BUG();
  while ( 1 )
  {
    i = *(_QWORD *)(v2 + 24);
    if ( i != v2 + 16 )
      break;
    v2 = *(_QWORD *)(v2 + 8);
    if ( v1 == v2 )
      return 0;
    if ( !v2 )
      BUG();
  }
  if ( v1 == v2 )
    return 0;
  while ( 1 )
  {
    if ( !i )
      BUG();
    v5 = *(_BYTE *)(i - 8);
    v6 = i - 24;
    if ( v5 > 0x17u )
    {
      if ( v5 == 78 )
      {
        v7 = v6 | 4;
      }
      else
      {
        if ( v5 != 29 )
          goto LABEL_13;
        v7 = v6 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v8 = v7 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v7 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        break;
    }
LABEL_13:
    for ( i = *(_QWORD *)(i + 8); i == v2 - 24 + 40; i = *(_QWORD *)(v2 + 24) )
    {
      v2 = *(_QWORD *)(v2 + 8);
      if ( v1 == v2 )
        return 0;
      if ( !v2 )
        BUG();
    }
    if ( v1 == v2 )
      return 0;
  }
  v9 = (_QWORD *)(v8 + 56);
  if ( (v7 & 4) != 0 )
  {
    if ( (unsigned __int8)sub_1560260(v9, -1, 39) )
      return 1;
    v10 = *(_QWORD *)(v8 - 24);
    if ( *(_BYTE *)(v10 + 16) )
      goto LABEL_13;
    goto LABEL_25;
  }
  if ( !(unsigned __int8)sub_1560260(v9, -1, 39) )
  {
    v10 = *(_QWORD *)(v8 - 72);
    if ( *(_BYTE *)(v10 + 16) )
      goto LABEL_13;
LABEL_25:
    v11[0] = *(_QWORD *)(v10 + 112);
    if ( (unsigned __int8)sub_1560260(v11, -1, 39) )
      return 1;
    goto LABEL_13;
  }
  return 1;
}
