// Function: sub_8CB6C0
// Address: 0x8cb6c0
//
void __fastcall sub_8CB6C0(unsigned __int8 a1, __int64 a2)
{
  __int64 *v2; // r13
  _BYTE *v3; // r15
  char v5; // al
  __int64 v6; // rsi
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  char v15; // al
  __int64 v16; // rcx
  __int64 v17; // rdx
  int v18; // [rsp+Ch] [rbp-34h]

  v2 = *(__int64 **)(a2 + 32);
  v3 = (_BYTE *)*v2;
  if ( a2 == *v2 )
    return;
  v18 = sub_8C6530(a1, a2);
  if ( v18 <= (int)sub_8C6530(a1, (__int64)v3) )
    return;
  switch ( a1 )
  {
    case 0xBu:
      v15 = *(_BYTE *)(a2 + 195);
      if ( (v15 & 1) != 0 && (v3[195] & 1) != 0 && (v15 & 0xA) == 0 )
        sub_899FE0(a2, (__int64)v3);
      break;
    case 0x3Bu:
      v9 = *(_QWORD *)v3;
      v10 = *(_QWORD *)a2;
      switch ( *(_BYTE *)(*(_QWORD *)v3 + 80LL) )
      {
        case 4:
        case 5:
          v16 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 80LL);
          break;
        case 6:
          v16 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 32LL);
          break;
        case 9:
        case 0xA:
          v16 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 56LL);
          break;
        case 0x13:
        case 0x14:
        case 0x15:
        case 0x16:
          v16 = *(_QWORD *)(v9 + 88);
          break;
        default:
          v16 = 0;
          break;
      }
      switch ( *(_BYTE *)(v10 + 80) )
      {
        case 4:
        case 5:
          v17 = *(_QWORD *)(*(_QWORD *)(v10 + 96) + 80LL);
          break;
        case 6:
          v17 = *(_QWORD *)(*(_QWORD *)(v10 + 96) + 32LL);
          break;
        case 9:
        case 0xA:
          v17 = *(_QWORD *)(*(_QWORD *)(v10 + 96) + 56LL);
          break;
        case 0x13:
        case 0x14:
        case 0x15:
        case 0x16:
          v17 = *(_QWORD *)(v10 + 88);
          break;
        default:
          v17 = 0;
          break;
      }
      v6 = *(_QWORD *)(v16 + 112);
      if ( v6 )
      {
        v7 = *(_QWORD **)(v17 + 112);
        if ( v7 )
        {
          do
          {
            v8 = v7;
            v7 = (_QWORD *)*v7;
          }
          while ( v7 );
          *v8 = v6;
        }
        else
        {
          *(_QWORD *)(v17 + 112) = v6;
        }
        *(_QWORD *)(v16 + 112) = 0;
      }
      break;
    case 7u:
      if ( (*(_BYTE *)(a2 + 89) & 4) != 0 )
        v11 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 32LL) + 177LL) >> 7;
      else
        v11 = (*(_BYTE *)(a2 + 170) & 0x20) != 0;
      if ( !v11 && (*(_BYTE *)(a2 + 170) & 0x10) != 0 )
      {
        v12 = *(_QWORD *)(*(_QWORD *)v3 + 96LL);
        if ( *(_QWORD *)(*(_QWORD *)a2 + 96LL) )
        {
          if ( v12 && *(_QWORD *)(v12 + 16) )
            sub_89A000(a2, (__int64)v3);
        }
      }
      break;
    default:
      if ( (*(v3 - 8) & 2) == 0 )
        goto LABEL_7;
      goto LABEL_30;
  }
  if ( (*(v3 - 8) & 2) == 0 )
  {
    *v2 = a2;
    return;
  }
LABEL_30:
  v13 = qword_4F60248;
  if ( qword_4F60248 )
    qword_4F60248 = *(_QWORD *)qword_4F60248;
  else
    v13 = sub_823970(24);
  v14 = qword_4F60250;
  *(_BYTE *)(v13 + 8) = a1;
  qword_4F60250 = v13;
  *(_QWORD *)v13 = v14;
  *(_QWORD *)(v13 + 16) = v3;
LABEL_7:
  *v2 = a2;
  if ( a1 == 6 )
  {
    v5 = v3[140];
    if ( (unsigned __int8)(v5 - 9) <= 2u )
    {
      if ( (v3[141] & 0x20) == 0 )
        sub_8CAE10((__int64)v3);
    }
    else if ( v5 == 2 && (v3[161] & 8) != 0 && (v3[141] & 0x20) == 0 )
    {
      sub_8CA420((__int64)v3);
    }
  }
}
