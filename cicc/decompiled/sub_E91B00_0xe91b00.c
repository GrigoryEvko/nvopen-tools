// Function: sub_E91B00
// Address: 0xe91b00
//
__int64 __fastcall sub_E91B00(char *a1, char *a2, __int64 a3)
{
  __int64 result; // rax
  char *v5; // r14
  __int64 v6; // rbx
  char *v7; // r10
  char *v8; // r12
  unsigned __int16 v9; // si
  unsigned __int16 v10; // r8
  unsigned __int16 v11; // cx
  char *v12; // rax
  unsigned __int16 v13; // dx
  unsigned __int16 v14; // dx
  unsigned __int16 *v15; // rdi
  char *v16; // r13
  char *v17; // rcx
  char *v18; // rax
  __int64 v19; // r12
  __int64 i; // rbx
  unsigned __int16 *v21; // r14
  unsigned __int16 v22; // cx
  __int64 v23; // rbx
  unsigned __int16 *v24; // [rsp-40h] [rbp-40h]

  result = a2 - a1;
  if ( a2 - a1 <= 32 )
    return result;
  v5 = a2;
  v6 = a3;
  if ( !a3 )
    goto LABEL_24;
  v7 = a2;
  v8 = a1 + 2;
  v24 = (unsigned __int16 *)(a1 + 4);
  while ( 2 )
  {
    v9 = *((_WORD *)a1 + 1);
    v10 = *(_WORD *)a1;
    --v6;
    v11 = *((_WORD *)v7 - 1);
    v12 = &a1[(((v7 - a1) >> 1) + ((unsigned __int64)(v7 - a1) >> 63)) & 0xFFFFFFFFFFFFFFFELL];
    v13 = *(_WORD *)v12;
    if ( v9 >= *(_WORD *)v12 )
    {
      if ( v11 > v9 )
        goto LABEL_7;
      if ( v11 > v13 )
      {
LABEL_18:
        *(_WORD *)a1 = v11;
        v14 = v10;
        *((_WORD *)v7 - 1) = v10;
        v9 = *(_WORD *)a1;
        v10 = *((_WORD *)a1 + 1);
        goto LABEL_8;
      }
LABEL_23:
      *(_WORD *)a1 = v13;
      *(_WORD *)v12 = v10;
      v10 = *((_WORD *)a1 + 1);
      v9 = *(_WORD *)a1;
      v14 = *((_WORD *)v7 - 1);
      goto LABEL_8;
    }
    if ( v11 > v13 )
      goto LABEL_23;
    if ( v11 > v9 )
      goto LABEL_18;
LABEL_7:
    *(_WORD *)a1 = v9;
    *((_WORD *)a1 + 1) = v10;
    v14 = *((_WORD *)v7 - 1);
LABEL_8:
    v15 = v24;
    v16 = v8;
    v17 = v7;
    while ( 1 )
    {
      v5 = v16;
      if ( v9 > v10 )
        goto LABEL_15;
      if ( v9 >= v14 )
      {
        v17 -= 2;
      }
      else
      {
        v18 = v17 - 4;
        do
        {
          v17 = v18;
          v14 = *(_WORD *)v18;
          v18 -= 2;
        }
        while ( v9 < v14 );
      }
      if ( v16 >= v17 )
        break;
      *(_WORD *)v16 = v14;
      v14 = *((_WORD *)v17 - 1);
      *(_WORD *)v17 = v10;
      v9 = *(_WORD *)a1;
LABEL_15:
      v10 = *v15;
      v16 += 2;
      ++v15;
    }
    sub_E91B00(v16, v7, v6);
    result = v16 - a1;
    if ( v16 - a1 > 32 )
    {
      if ( v6 )
      {
        v7 = v16;
        continue;
      }
LABEL_24:
      v19 = result >> 1;
      for ( i = ((result >> 1) - 2) >> 1; ; --i )
      {
        sub_E91910((__int64)a1, i, v19, *(_WORD *)&a1[2 * i]);
        if ( !i )
          break;
      }
      v21 = (unsigned __int16 *)(v5 - 2);
      do
      {
        v22 = *v21;
        v23 = (char *)v21-- - a1;
        v21[1] = *(_WORD *)a1;
        result = sub_E91910((__int64)a1, 0, v23 >> 1, v22);
      }
      while ( v23 > 2 );
    }
    return result;
  }
}
