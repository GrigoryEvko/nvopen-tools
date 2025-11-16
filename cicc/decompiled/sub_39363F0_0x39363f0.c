// Function: sub_39363F0
// Address: 0x39363f0
//
__int64 *__fastcall sub_39363F0(unsigned int *a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  unsigned __int64 v7; // rbx
  char v8; // si
  char v9; // al
  char *v10; // rax
  unsigned __int64 v11; // rbx
  char v12; // si
  char v13; // al
  char *v14; // rax
  __int64 *v15; // rbx
  __int64 *result; // rax
  unsigned __int64 v17; // r15
  char v18; // si
  char v19; // al
  __int64 *i; // [rsp+8h] [rbp-38h]

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
      v10 = *(char **)(a2 + 24);
      if ( (unsigned __int64)v10 >= *(_QWORD *)(a2 + 16) )
        break;
      *(_QWORD *)(a2 + 24) = v10 + 1;
      *v10 = v8;
      if ( !v7 )
        goto LABEL_7;
    }
    sub_16E7DE0(a2, v8);
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
      v14 = *(char **)(a2 + 24);
      if ( (unsigned __int64)v14 >= *(_QWORD *)(a2 + 16) )
        break;
      *(_QWORD *)(a2 + 24) = v14 + 1;
      *v14 = v12;
      if ( !v11 )
        goto LABEL_13;
    }
    sub_16E7DE0(a2, v12);
  }
  while ( v11 );
LABEL_13:
  v15 = (__int64 *)*((_QWORD *)a1 + 1);
  result = &v15[a1[4]];
  for ( i = result; i != v15; ++v15 )
  {
    v17 = sub_3913FA0(a3, *v15, a4);
    do
    {
      while ( 1 )
      {
        v18 = v17 & 0x7F;
        v19 = v17 & 0x7F | 0x80;
        v17 >>= 7;
        if ( v17 )
          v18 = v19;
        result = *(__int64 **)(a2 + 24);
        if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 16) )
          break;
        *(_QWORD *)(a2 + 24) = (char *)result + 1;
        *(_BYTE *)result = v18;
        if ( !v17 )
          goto LABEL_20;
      }
      result = (__int64 *)sub_16E7DE0(a2, v18);
    }
    while ( v17 );
LABEL_20:
    ;
  }
  return result;
}
