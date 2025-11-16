// Function: sub_2450F80
// Address: 0x2450f80
//
char *__fastcall sub_2450F80(char *a1, char *a2)
{
  char *result; // rax
  __int64 v3; // rdx
  __int64 v4; // rsi
  __int64 v5; // rdx
  char *v6; // rsi
  unsigned __int64 v7; // rdx
  __int64 v8; // rcx
  unsigned __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned __int64 v11; // rdx
  __int64 v12; // rcx
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rdx

  result = a2;
  v3 = a2 - a1;
  v4 = (a2 - a1) >> 5;
  v5 = v3 >> 3;
  if ( v4 <= 0 )
  {
LABEL_21:
    if ( v5 != 2 )
    {
      if ( v5 != 3 )
      {
        if ( v5 != 1 )
          return result;
LABEL_35:
        v16 = *(_QWORD *)(*(_QWORD *)a1 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v16 != *(_QWORD *)a1 + 48LL && v16 && (unsigned int)*(unsigned __int8 *)(v16 - 24) - 30 <= 0xA )
        {
          if ( *(_BYTE *)(v16 - 24) == 39 )
            return a1;
          return result;
        }
LABEL_45:
        BUG();
      }
      v14 = *(_QWORD *)(*(_QWORD *)a1 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v14 == *(_QWORD *)a1 + 48LL || !v14 || (unsigned int)*(unsigned __int8 *)(v14 - 24) - 30 > 0xA )
        goto LABEL_45;
      if ( *(_BYTE *)(v14 - 24) == 39 )
        return a1;
      a1 += 8;
    }
    v15 = *(_QWORD *)(*(_QWORD *)a1 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v15 == *(_QWORD *)a1 + 48LL || !v15 || (unsigned int)*(unsigned __int8 *)(v15 - 24) - 30 > 0xA )
      goto LABEL_45;
    if ( *(_BYTE *)(v15 - 24) != 39 )
    {
      a1 += 8;
      goto LABEL_35;
    }
    return a1;
  }
  v6 = &a1[32 * v4];
  while ( 1 )
  {
    v7 = *(_QWORD *)(*(_QWORD *)a1 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v7 == *(_QWORD *)a1 + 48LL || !v7 || (unsigned int)*(unsigned __int8 *)(v7 - 24) - 30 > 0xA )
      goto LABEL_45;
    if ( *(_BYTE *)(v7 - 24) == 39 )
      return a1;
    v8 = *((_QWORD *)a1 + 1);
    v9 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v9 == v8 + 48 || !v9 || (unsigned int)*(unsigned __int8 *)(v9 - 24) - 30 > 0xA )
      goto LABEL_45;
    if ( *(_BYTE *)(v9 - 24) == 39 )
      return a1 + 8;
    v10 = *((_QWORD *)a1 + 2);
    v11 = *(_QWORD *)(v10 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v11 == v10 + 48 || !v11 || (unsigned int)*(unsigned __int8 *)(v11 - 24) - 30 > 0xA )
      goto LABEL_45;
    if ( *(_BYTE *)(v11 - 24) == 39 )
      return a1 + 16;
    v12 = *((_QWORD *)a1 + 3);
    v13 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v13 == v12 + 48 || !v13 || (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA )
      goto LABEL_45;
    if ( *(_BYTE *)(v13 - 24) == 39 )
      return a1 + 24;
    a1 += 32;
    if ( v6 == a1 )
    {
      v5 = (result - a1) >> 3;
      goto LABEL_21;
    }
  }
}
