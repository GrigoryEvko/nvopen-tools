// Function: sub_1951E40
// Address: 0x1951e40
//
__int64 __fastcall sub_1951E40(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r14
  unsigned int v4; // r15d
  __int64 v5; // r13
  unsigned __int8 v6; // al
  __int64 v7; // rdx
  char v8; // al
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  char v14; // al
  unsigned int v15; // r9d
  unsigned int v16; // [rsp+Ch] [rbp-54h]
  unsigned int v17; // [rsp+18h] [rbp-48h]
  _QWORD v19[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = sub_157ED20(a1);
  if ( v3 )
    v3 += 24;
  v16 = 0;
  if ( a2 == sub_157EBA0(a1) )
  {
    v14 = *(_BYTE *)(a2 + 16);
    if ( v14 == 27 )
    {
      a3 += 6;
      v16 = 6;
    }
    else if ( v14 == 28 )
    {
      a3 += 8;
      v16 = 8;
    }
  }
  v4 = 0;
  while ( 1 )
  {
    v5 = 0;
    if ( v3 )
      v5 = v3 - 24;
    if ( v5 == a2 )
      break;
    if ( v4 > a3 )
      return v4;
    v6 = *(_BYTE *)(v5 + 16);
    if ( v6 == 78 )
    {
      v12 = *(_QWORD *)(v5 - 24);
      if ( *(_BYTE *)(v12 + 16) || (*(_BYTE *)(v12 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v12 + 36) - 35) > 3 )
        return (unsigned int)-1;
    }
    else
    {
      if ( (unsigned int)v6 - 25 > 9 )
        return (unsigned int)-1;
      v7 = *(_QWORD *)v5;
      if ( v6 == 71 )
      {
        v8 = *(_BYTE *)(v7 + 8);
        if ( v8 == 15 )
          goto LABEL_7;
        if ( v8 != 10 )
        {
LABEL_6:
          ++v4;
          goto LABEL_7;
        }
      }
      else if ( *(_BYTE *)(v7 + 8) != 10 )
      {
        goto LABEL_6;
      }
      if ( (unsigned __int8)sub_15F2E00(v5, a1) )
        return (unsigned int)-1;
      v17 = v4 + 1;
      if ( *(_BYTE *)(v5 + 16) != 78 )
        goto LABEL_28;
      if ( (unsigned __int8)sub_1560260((_QWORD *)(v5 + 56), -1, 24) )
        return (unsigned int)-1;
      v9 = *(_QWORD *)(v5 - 24);
      if ( !*(_BYTE *)(v9 + 16) )
      {
        v19[0] = *(_QWORD *)(v9 + 112);
        if ( (unsigned __int8)sub_1560260(v19, -1, 24) )
          return (unsigned int)-1;
      }
      if ( (unsigned __int8)sub_1560260((_QWORD *)(v5 + 56), -1, 8) )
        return (unsigned int)-1;
      v10 = *(_QWORD *)(v5 - 24);
      if ( !*(_BYTE *)(v10 + 16) )
      {
        v19[0] = *(_QWORD *)(v10 + 112);
        if ( (unsigned __int8)sub_1560260(v19, -1, 8) )
          return (unsigned int)-1;
        v11 = *(_QWORD *)(v5 - 24);
        if ( !*(_BYTE *)(v11 + 16) && (*(_BYTE *)(v11 + 33) & 0x20) != 0 )
        {
          v4 += 2;
          if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) != 16 )
            goto LABEL_7;
LABEL_28:
          v4 = v17;
          goto LABEL_7;
        }
      }
      v4 += 4;
    }
LABEL_7:
    v3 = *(_QWORD *)(v3 + 8);
  }
  v15 = 0;
  if ( v16 < v4 )
    return v4 - v16;
  return v15;
}
