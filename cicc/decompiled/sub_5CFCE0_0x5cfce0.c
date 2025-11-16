// Function: sub_5CFCE0
// Address: 0x5cfce0
//
__int64 __fastcall sub_5CFCE0(__int64 a1, char a2, __int64 a3)
{
  __int64 v4; // r12
  char v6; // bl
  unsigned __int8 v7; // r15
  char v8; // al
  unsigned int v9; // r14d
  char v10; // al
  char v11; // al
  __int64 v12; // rax
  char v13; // dl
  unsigned int v14; // ebx
  unsigned int v15; // eax
  unsigned __int8 v16; // al
  __int64 v17; // [rsp+8h] [rbp-38h]

  v4 = a1;
  switch ( a2 )
  {
    case 0:
      return a1;
    case 1:
      v17 = 1;
      v6 = 2;
      v7 = 2;
      goto LABEL_4;
    case 2:
      v17 = 2;
      v7 = a2;
      v6 = a2;
      goto LABEL_4;
    case 3:
      v17 = 4;
      v6 = 2;
      v7 = 2;
      goto LABEL_4;
    case 4:
      v17 = 8;
      v6 = 2;
      v7 = 2;
      goto LABEL_4;
    case 5:
      v17 = 16;
      v6 = 2;
      v7 = 2;
      goto LABEL_4;
    case 6:
      v17 = 4;
      v6 = 3;
      v7 = 2;
      goto LABEL_4;
    case 7:
      v17 = 8;
      v6 = 3;
      v7 = 2;
      goto LABEL_4;
    case 8:
      v6 = 3;
      v7 = unk_4B6D467;
      v17 = 0;
      goto LABEL_4;
    case 9:
      v6 = 3;
      v7 = unk_4B6D466;
      v17 = 0;
      goto LABEL_4;
    case 10:
      v17 = 4;
      v6 = 5;
      v7 = 2;
      goto LABEL_4;
    case 11:
      v17 = 8;
      v6 = 5;
      v7 = 2;
      goto LABEL_4;
    case 12:
      v6 = 5;
      v7 = unk_4B6D467;
      v17 = 0;
      goto LABEL_4;
    case 13:
      v6 = 5;
      v7 = unk_4B6D466;
      v17 = 0;
LABEL_4:
      v8 = *(_BYTE *)(a1 + 140);
      v9 = 0;
      if ( (v8 & 0xFB) != 8 || (v9 = sub_8D4C10(a1, unk_4F077C4 != 2), v8 = *(_BYTE *)(a1 + 140), v8 != 12) )
      {
        if ( v6 == v8 )
          goto LABEL_25;
LABEL_6:
        if ( a3 )
          sub_685360(1088, a3);
        goto LABEL_8;
      }
      do
      {
        v4 = *(_QWORD *)(v4 + 160);
        v10 = *(_BYTE *)(v4 + 140);
      }
      while ( v10 == 12 );
      if ( v6 != v10 )
        goto LABEL_6;
LABEL_25:
      if ( v6 == 2 )
      {
        v15 = sub_8D27E0(v4);
        v16 = sub_622A90((unsigned int)(unk_4F06BA0 * v17), v15);
        if ( v16 != 13 )
        {
          v4 = sub_72BA30(v16);
          return sub_73C570(v4, v9, -1);
        }
        if ( a3 )
          sub_6851C0(1089, a3);
LABEL_8:
        v4 = sub_72C930();
        return sub_73C570(v4, v9, -1);
      }
      if ( v7 == 2 )
      {
        v14 = 0;
        while ( 1 )
        {
          v7 = v14;
          if ( (((_BYTE)v14 - 3) & 0xFD) != 0 )
          {
            a1 = v14;
            if ( *(_QWORD *)(sub_72C610(v14) + 128) == v17 )
              break;
          }
          if ( ++v14 == 9 )
          {
            if ( a3 )
            {
              a1 = 1089;
              sub_6851C0(1089, a3);
            }
            v7 = 9;
            v4 = sub_72C930();
            break;
          }
        }
      }
      v11 = *(_BYTE *)(v4 + 140);
      if ( v11 == 12 )
      {
        v12 = v4;
        do
        {
          v12 = *(_QWORD *)(v12 + 160);
          v13 = *(_BYTE *)(v12 + 140);
        }
        while ( v13 == 12 );
        if ( v13 )
LABEL_31:
          sub_721090(a1);
      }
      else if ( v11 )
      {
        if ( v11 == 3 )
        {
          v4 = sub_72C610(v7);
        }
        else
        {
          if ( v11 != 5 )
            goto LABEL_31;
          v4 = sub_72C6F0(v7);
        }
      }
      return sub_73C570(v4, v9, -1);
    default:
      goto LABEL_31;
  }
}
