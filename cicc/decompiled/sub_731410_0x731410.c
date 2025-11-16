// Function: sub_731410
// Address: 0x731410
//
__int64 __fastcall sub_731410(__int64 a1, int a2)
{
  __int64 v2; // r12
  char v3; // al
  __int64 v5; // rdx
  char v6; // al
  char v7; // al
  __int64 v8; // rax
  char v9; // si
  char v10; // cl
  char v11; // cl
  char v12; // dl
  char v13; // cl
  char v14; // dl
  __int64 v16; // rax

  v2 = a1;
  v3 = *(_BYTE *)(a1 + 24);
  if ( v3 == 1 )
  {
    v5 = a1;
    do
    {
      while ( 1 )
      {
        v7 = *(_BYTE *)(v5 + 56);
        if ( v7 == 91 )
          break;
        if ( v7 != 94 && v7 != 25 )
          goto LABEL_23;
        v5 = *(_QWORD *)(v5 + 72);
        v6 = *(_BYTE *)(v5 + 24);
        if ( v6 != 1 )
          goto LABEL_8;
      }
      v5 = *(_QWORD *)(*(_QWORD *)(v5 + 72) + 16LL);
      v6 = *(_BYTE *)(v5 + 24);
    }
    while ( v6 == 1 );
LABEL_8:
    if ( (unsigned __int8)(v6 - 5) <= 1u )
    {
      v8 = a1;
      if ( v5 == a1 )
        goto LABEL_25;
      while ( 1 )
      {
        v11 = *(_BYTE *)(v8 + 25);
        if ( a2 )
        {
          v9 = *(_BYTE *)(v8 + 56);
          v10 = v11 | 2;
          *(_BYTE *)(v8 + 25) = v10;
          if ( v9 != 91 )
            goto LABEL_12;
LABEL_17:
          *(_BYTE *)(v8 + 58) |= 1u;
          v8 = *(_QWORD *)(*(_QWORD *)(v8 + 72) + 16LL);
          if ( v5 == v8 )
            goto LABEL_18;
        }
        else
        {
          v9 = *(_BYTE *)(v8 + 56);
          v10 = v11 | 1;
          *(_BYTE *)(v8 + 25) = v10;
          if ( v9 == 91 )
            goto LABEL_17;
LABEL_12:
          if ( v9 != 94 && v9 != 25 )
            goto LABEL_19;
          v8 = *(_QWORD *)(v8 + 72);
          if ( v5 == v8 )
          {
LABEL_18:
            v10 = *(_BYTE *)(v8 + 25);
            goto LABEL_19;
          }
        }
      }
    }
LABEL_23:
    v16 = sub_6EAFA0(3u);
    *(_QWORD *)(v16 + 56) = a1;
    v8 = sub_6EC670(*(_QWORD *)a1, v16, 1, 0);
    v10 = *(_BYTE *)(v8 + 25) & 0xFE;
    *(_BYTE *)(v8 + 25) = v10;
    v2 = v8;
    *(_QWORD *)(v8 + 28) = *(_QWORD *)(a1 + 28);
    goto LABEL_19;
  }
  if ( (unsigned __int8)(v3 - 5) > 1u )
    goto LABEL_23;
  v8 = a1;
LABEL_25:
  v10 = *(_BYTE *)(a1 + 25);
LABEL_19:
  v12 = v10;
  v13 = v10 | 2;
  v14 = v12 | 1;
  if ( !a2 )
    v13 = v14;
  *(_BYTE *)(v8 + 25) = v13;
  return v2;
}
