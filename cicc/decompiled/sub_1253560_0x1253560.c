// Function: sub_1253560
// Address: 0x1253560
//
_BYTE *__fastcall sub_1253560(unsigned int *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v7; // rbx
  char v8; // si
  char v9; // al
  char *v10; // rax
  unsigned __int64 v11; // rbx
  char v12; // si
  char v13; // al
  char *v14; // rax
  _BYTE **v15; // rbx
  _BYTE *result; // rax
  unsigned __int64 v17; // r15
  char v18; // si
  char v19; // al
  _BYTE **v20; // [rsp+8h] [rbp-38h]

  v7 = *a1;
  do
  {
    while ( 1 )
    {
      v8 = v7 & 0x7F;
      v9 = v7 & 0x7F | 0x80;
      v7 >>= 7;
      if ( v7 )
        v8 = v9;
      v10 = *(char **)(a3 + 32);
      if ( (unsigned __int64)v10 >= *(_QWORD *)(a3 + 24) )
        break;
      *(_QWORD *)(a3 + 32) = v10 + 1;
      *v10 = v8;
      if ( !v7 )
        goto LABEL_7;
    }
    sub_CB5D20(a3, v8);
  }
  while ( v7 );
LABEL_7:
  v11 = a1[4];
  do
  {
    while ( 1 )
    {
      v12 = v11 & 0x7F;
      v13 = v11 & 0x7F | 0x80;
      v11 >>= 7;
      if ( v11 )
        v12 = v13;
      v14 = *(char **)(a3 + 32);
      if ( (unsigned __int64)v14 >= *(_QWORD *)(a3 + 24) )
        break;
      *(_QWORD *)(a3 + 32) = v14 + 1;
      *v14 = v12;
      if ( !v11 )
        goto LABEL_13;
    }
    sub_CB5D20(a3, v12);
  }
  while ( v11 );
LABEL_13:
  v15 = (_BYTE **)*((_QWORD *)a1 + 1);
  result = &v15[a1[4]];
  v20 = (_BYTE **)result;
  if ( result != (_BYTE *)v15 )
  {
    do
    {
      v17 = sub_1070C50(a4, *v15, a2, a4);
      do
      {
        while ( 1 )
        {
          v18 = v17 & 0x7F;
          v19 = v17 & 0x7F | 0x80;
          v17 >>= 7;
          if ( v17 )
            v18 = v19;
          result = *(_BYTE **)(a3 + 32);
          if ( (unsigned __int64)result >= *(_QWORD *)(a3 + 24) )
            break;
          *(_QWORD *)(a3 + 32) = result + 1;
          *result = v18;
          if ( !v17 )
            goto LABEL_20;
        }
        result = (_BYTE *)sub_CB5D20(a3, v18);
      }
      while ( v17 );
LABEL_20:
      ++v15;
    }
    while ( v20 != v15 );
  }
  return result;
}
