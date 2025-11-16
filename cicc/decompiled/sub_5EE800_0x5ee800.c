// Function: sub_5EE800
// Address: 0x5ee800
//
__int64 __fastcall sub_5EE800(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  char v7; // al
  __int64 v8; // r12
  int v9; // r15d
  char v11; // al
  __int64 v12; // rdx
  _DWORD v13[13]; // [rsp+Ch] [rbp-34h] BYREF

  v6 = *(unsigned __int8 *)(a1 + 80);
  if ( (_BYTE)v6 == 2 || (_BYTE)v6 == 20 )
    return 0;
  v7 = *(_BYTE *)(a2 + 80);
  v8 = a2;
  v9 = 0;
  if ( v7 != 17 )
    goto LABEL_4;
  v8 = *(_QWORD *)(a2 + 88);
  if ( !v8 )
    return 0;
  v7 = *(_BYTE *)(v8 + 80);
  v9 = 1;
LABEL_4:
  if ( (_BYTE)v6 == v7 )
    goto LABEL_9;
LABEL_5:
  if ( v7 != 16 )
    goto LABEL_28;
  v11 = *(_BYTE *)(v8 + 96);
  if ( (v11 & 4) == 0 )
    goto LABEL_28;
  v12 = **(_QWORD **)(v8 + 88);
  if ( *(_BYTE *)(v12 + 80) == 24 )
    v12 = *(_QWORD *)(v12 + 88);
  if ( a1 != v12 )
  {
LABEL_28:
    while ( v9 )
    {
      v8 = *(_QWORD *)(v8 + 8);
      if ( !v8 )
        break;
      v7 = *(_BYTE *)(v8 + 80);
      v6 = *(unsigned __int8 *)(a1 + 80);
      if ( (_BYTE)v6 != v7 )
        goto LABEL_5;
LABEL_9:
      if ( (!dword_4F077BC || (_BYTE)v6 != 11 || !sub_5E9060(*(_QWORD *)(a1 + 88), v8, v6, dword_4F077BC, a5))
        && (unsigned int)sub_5E9110(v8, a1, v13) )
      {
        if ( !v13[0] )
          return 1;
        sub_686C60(735, a3, a1, v8);
        return 1;
      }
    }
    return 0;
  }
  if ( (v11 & 3) != (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C5C + 5) & 3) )
    sub_6854C0(720, a3, a1);
  return 1;
}
