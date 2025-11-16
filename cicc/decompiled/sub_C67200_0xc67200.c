// Function: sub_C67200
// Address: 0xc67200
//
__int64 *__fastcall sub_C67200(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdx
  int v3; // ebx
  __int64 v4; // rsi
  __int64 v5; // r13
  char v6; // al
  char v8; // al

  *a1 = (__int64)(a1 + 2);
  sub_C66B70(a1, *(_BYTE **)a2, *(_QWORD *)a2 + *(_QWORD *)(a2 + 8));
  v2 = a1[1];
  if ( v2 )
  {
    v3 = 0;
    v4 = 0;
    while ( 1 )
    {
      v5 = (unsigned int)(v3 + 1);
      v6 = *(_BYTE *)(*a1 + v4);
      if ( v6 == 34 )
        goto LABEL_15;
      if ( v6 > 34 )
        break;
      if ( v6 == 9 )
      {
        sub_2240FD0(a1, v4, 0, 1, 32);
        v4 = (unsigned int)(v3 + 2);
        v3 += 2;
        *(_BYTE *)(*a1 + v5) = 32;
        v2 = a1[1];
        goto LABEL_6;
      }
      if ( v6 == 10 )
      {
        sub_2240FD0(a1, v4, 0, 1, 92);
        v4 = (unsigned int)(v3 + 2);
        v3 += 2;
        *(_BYTE *)(*a1 + v5) = 110;
        v2 = a1[1];
LABEL_6:
        if ( v2 == v4 )
          return a1;
      }
      else
      {
LABEL_12:
        v4 = (unsigned int)v5;
        ++v3;
        if ( v2 == (unsigned int)v5 )
          return a1;
      }
    }
    if ( v6 == 92 )
    {
      if ( (unsigned int)v5 != v2 )
      {
        v8 = *(_BYTE *)(*a1 + (unsigned int)v5);
        if ( v8 == 108 )
        {
          ++v3;
          v4 = (unsigned int)v5;
          goto LABEL_6;
        }
        if ( (unsigned __int8)(v8 - 123) <= 2u )
        {
          ++v3;
          sub_2240CE0(a1, v4, 1);
          v2 = a1[1];
          v4 = (unsigned int)v5;
          goto LABEL_6;
        }
      }
    }
    else if ( v6 <= 92 )
    {
      if ( (v6 & 0xFD) != 0x3C )
        goto LABEL_12;
    }
    else if ( (unsigned __int8)(v6 - 123) > 2u )
    {
      goto LABEL_12;
    }
LABEL_15:
    sub_2240FD0(a1, v4, 0, 1, 92);
    v4 = (unsigned int)(v3 + 2);
    v2 = a1[1];
    v3 += 2;
    goto LABEL_6;
  }
  return a1;
}
