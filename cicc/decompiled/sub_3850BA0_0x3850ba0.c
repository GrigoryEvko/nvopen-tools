// Function: sub_3850BA0
// Address: 0x3850ba0
//
__int64 __fastcall sub_3850BA0(__int64 a1)
{
  char v1; // al
  __int64 v2; // rbx
  char v3; // r13
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 v6; // r14
  unsigned __int8 v7; // dl
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // r11
  int v12; // eax
  __int64 v13; // rdx
  char v14; // si
  unsigned int v15; // eax
  __int64 *v17; // r11
  __int64 v18; // rax
  unsigned __int64 v19; // [rsp+10h] [rbp-50h]
  unsigned __int64 v20; // [rsp+18h] [rbp-48h]
  __int64 v21[7]; // [rsp+28h] [rbp-38h] BYREF

  v1 = sub_1560180(a1 + 112, 39);
  v2 = *(_QWORD *)(a1 + 80);
  v3 = v1;
  if ( v2 != a1 + 72 )
  {
    while ( 1 )
    {
      v4 = 0;
      if ( v2 )
        v4 = v2 - 24;
      if ( *(_BYTE *)(sub_157EBA0(v4) + 16) == 28 || *(_WORD *)(v4 + 18) )
        return 0;
      v5 = *(_QWORD *)(v4 + 48);
      v6 = v4 + 40;
      if ( v6 != v5 )
        break;
LABEL_28:
      v2 = *(_QWORD *)(v2 + 8);
      if ( a1 + 72 == v2 )
        return 1;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v5 )
          BUG();
        v7 = *(_BYTE *)(v5 - 8);
        v8 = v5 - 24;
        if ( v7 <= 0x17u )
          goto LABEL_9;
        if ( v7 == 78 )
        {
          v9 = v8 | 4;
        }
        else
        {
          if ( v7 != 29 )
            goto LABEL_9;
          v9 = v8 & 0xFFFFFFFFFFFFFFFBLL;
        }
        v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v9 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          v11 = v10 - 72;
          v12 = (v9 >> 2) & 1;
          if ( v12 )
            v11 = v10 - 24;
          v13 = *(_QWORD *)v11;
          v14 = *(_BYTE *)(*(_QWORD *)v11 + 16LL);
          if ( a1 == *(_QWORD *)v11 && !v14 )
            return 0;
          if ( !v3 && (_BYTE)v12 )
          {
            v19 = v11;
            v20 = v10;
            if ( (unsigned __int8)sub_1560260((_QWORD *)(v10 + 56), -1, 39) )
              return 0;
            v17 = (__int64 *)v19;
            v18 = *(_QWORD *)(v20 - 24);
            if ( !*(_BYTE *)(v18 + 16) )
            {
              v21[0] = *(_QWORD *)(v18 + 112);
              if ( (unsigned __int8)sub_1560260(v21, -1, 39) )
                return 0;
              v17 = (__int64 *)v19;
            }
            v13 = *v17;
            v14 = *(_BYTE *)(*v17 + 16);
          }
          if ( !v14 )
            break;
        }
LABEL_9:
        v5 = *(_QWORD *)(v5 + 8);
        if ( v6 == v5 )
          goto LABEL_28;
      }
      v15 = *(_DWORD *)(v13 + 36);
      if ( v15 == 120 )
        return 0;
      if ( v15 <= 0x78 )
      {
        if ( v15 == 108 )
          return 0;
        goto LABEL_9;
      }
      if ( v15 - 213 <= 1 )
        return 0;
      v5 = *(_QWORD *)(v5 + 8);
      if ( v6 == v5 )
        goto LABEL_28;
    }
  }
  return 1;
}
