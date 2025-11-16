// Function: sub_13FBCD0
// Address: 0x13fbcd0
//
__int64 __fastcall sub_13FBCD0(__int64 a1)
{
  _QWORD *v1; // r12
  __int64 v2; // r15
  __int64 v3; // rbx
  __int64 v4; // r15
  unsigned __int8 v5; // al
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // rdi
  __int64 v11; // rax
  _QWORD *v12; // [rsp+8h] [rbp-48h]
  _QWORD v13[7]; // [rsp+18h] [rbp-38h] BYREF

  v1 = *(_QWORD **)(a1 + 32);
  v12 = *(_QWORD **)(a1 + 40);
  if ( v12 != v1 )
  {
    while ( 1 )
    {
      v2 = *v1;
      if ( *(_BYTE *)(sub_157EBA0(*v1) + 16) == 28 )
        return 0;
      v3 = *(_QWORD *)(v2 + 48);
      v4 = v2 + 40;
      if ( v4 != v3 )
        break;
LABEL_18:
      if ( v12 == ++v1 )
        return 1;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v3 )
          BUG();
        v5 = *(_BYTE *)(v3 - 8);
        v6 = v3 - 24;
        if ( v5 <= 0x17u )
          goto LABEL_6;
        if ( v5 == 78 )
        {
          v7 = v6 | 4;
        }
        else
        {
          if ( v5 != 29 )
            goto LABEL_6;
          v7 = v6 & 0xFFFFFFFFFFFFFFFBLL;
        }
        v8 = v7 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v7 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          goto LABEL_6;
        v9 = v8 + 56;
        if ( (v7 & 4) != 0 )
          break;
        if ( (unsigned __int8)sub_1560260(v9, 0xFFFFFFFFLL, 24) )
          return 0;
        v11 = *(_QWORD *)(v8 - 72);
        if ( !*(_BYTE *)(v11 + 16) )
          goto LABEL_16;
LABEL_6:
        v3 = *(_QWORD *)(v3 + 8);
        if ( v4 == v3 )
          goto LABEL_18;
      }
      if ( (unsigned __int8)sub_1560260(v9, 0xFFFFFFFFLL, 24) )
        return 0;
      v11 = *(_QWORD *)(v8 - 24);
      if ( *(_BYTE *)(v11 + 16) )
        goto LABEL_6;
LABEL_16:
      v13[0] = *(_QWORD *)(v11 + 112);
      if ( (unsigned __int8)sub_1560260(v13, 0xFFFFFFFFLL, 24) )
        return 0;
      v3 = *(_QWORD *)(v3 + 8);
      if ( v4 == v3 )
        goto LABEL_18;
    }
  }
  return 1;
}
