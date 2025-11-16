// Function: sub_39F5E90
// Address: 0x39f5e90
//
char *__fastcall sub_39F5E90(_QWORD *a1, char a2, char *a3, unsigned __int64 *a4)
{
  unsigned __int8 v7; // r8
  char *v8; // r10
  _QWORD **v9; // rax
  unsigned __int64 v10; // rdx
  char *result; // rax
  unsigned int v12; // ecx
  char v13; // bl
  unsigned __int64 v14; // r11
  int v15; // ecx
  char v16; // bl
  unsigned __int64 v17; // r11

  if ( a2 == -1 )
    goto LABEL_37;
  v7 = a2 & 0x70;
  if ( (a2 & 0x70) != 0x30 )
  {
    if ( v7 > 0x30u )
    {
      if ( v7 == 64 )
      {
        v8 = (char *)a1[23];
        if ( a2 == 80 )
          goto LABEL_9;
        goto LABEL_6;
      }
      if ( v7 == 80 )
        goto LABEL_13;
    }
    else
    {
      if ( v7 == 32 )
      {
        v8 = (char *)a1[21];
        if ( a2 != 80 )
          goto LABEL_6;
LABEL_9:
        v9 = (_QWORD **)((unsigned __int64)(a3 + 7) & 0xFFFFFFFFFFFFFFF8LL);
        v10 = (unsigned __int64)*v9;
        result = (char *)(v9 + 1);
LABEL_10:
        *a4 = v10;
        return result;
      }
      if ( v7 <= 0x20u && (a2 & 0x60) == 0 )
      {
LABEL_13:
        v8 = 0;
        goto LABEL_14;
      }
    }
LABEL_37:
    abort();
  }
  v8 = (char *)a1[22];
LABEL_14:
  if ( a2 == 80 )
    goto LABEL_9;
LABEL_6:
  switch ( a2 & 0xF )
  {
    case 0:
    case 4:
    case 0xC:
      v10 = *(_QWORD *)a3;
      result = a3 + 8;
      goto LABEL_19;
    case 1:
      result = a3;
      v10 = 0;
      v15 = 0;
      do
      {
        v16 = *result++;
        v17 = (unsigned __int64)(v16 & 0x7F) << v15;
        v15 += 7;
        v10 |= v17;
      }
      while ( v16 < 0 );
      goto LABEL_19;
    case 2:
      v10 = *(unsigned __int16 *)a3;
      result = a3 + 2;
      goto LABEL_19;
    case 3:
      v10 = *(unsigned int *)a3;
      result = a3 + 4;
      goto LABEL_19;
    case 9:
      result = a3;
      v10 = 0;
      v12 = 0;
      do
      {
        v13 = *result++;
        v14 = (unsigned __int64)(v13 & 0x7F) << v12;
        v12 += 7;
        v10 |= v14;
      }
      while ( v13 < 0 );
      if ( v12 > 0x3F || (v13 & 0x40) == 0 )
        goto LABEL_19;
      v10 |= -1LL << v12;
      goto LABEL_20;
    case 0xA:
      v10 = *(__int16 *)a3;
      result = a3 + 2;
      goto LABEL_19;
    case 0xB:
      v10 = *(int *)a3;
      result = a3 + 4;
LABEL_19:
      if ( !v10 )
        goto LABEL_10;
LABEL_20:
      if ( v7 == 16 )
        v8 = a3;
      v10 += (unsigned __int64)v8;
      if ( a2 >= 0 )
        goto LABEL_10;
      *a4 = *(_QWORD *)v10;
      break;
    default:
      goto LABEL_37;
  }
  return result;
}
