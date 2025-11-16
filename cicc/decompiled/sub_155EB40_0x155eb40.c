// Function: sub_155EB40
// Address: 0x155eb40
//
__int64 __fastcall sub_155EB40(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v4; // r14
  __int64 *v5; // r12
  __int64 *v6; // r13
  char v7; // al
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 *v10; // r12
  __int64 *v11; // r15
  __int64 *v12; // r13
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // r14
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // [rsp+0h] [rbp-40h]
  __int64 v22; // [rsp+8h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v22 = a3;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  v4 = a2;
  if ( !a3 )
  {
    v12 = a2;
    goto LABEL_19;
  }
  v21 = a1 + 1;
  while ( 2 )
  {
    --v22;
    v5 = v4 - 1;
    v6 = &a1[result >> 4];
    v7 = sub_155E9A0(v21, *v6);
    v8 = *(v4 - 1);
    if ( !v7 )
    {
      if ( !sub_155E9A0(v21, v8) )
      {
        if ( sub_155E9A0(v6, *v5) )
        {
          v20 = *a1;
          *a1 = *(v4 - 1);
          *(v4 - 1) = v20;
          goto LABEL_7;
        }
        goto LABEL_6;
      }
LABEL_17:
      v15 = a1[1];
      a1[1] = *a1;
      *a1 = v15;
      goto LABEL_7;
    }
    if ( !sub_155E9A0(v6, v8) )
    {
      if ( sub_155E9A0(v21, *v5) )
      {
        v19 = *a1;
        *a1 = *(v4 - 1);
        *(v4 - 1) = v19;
        goto LABEL_7;
      }
      goto LABEL_17;
    }
LABEL_6:
    v9 = *a1;
    *a1 = *v6;
    *v6 = v9;
LABEL_7:
    v10 = a1 + 1;
    v11 = v4;
    while ( 1 )
    {
      v12 = v10;
      if ( sub_155E9A0(v10, *a1) )
        goto LABEL_8;
      do
        v13 = *--v11;
      while ( sub_155E9A0(a1, v13) );
      if ( v10 >= v11 )
        break;
      v14 = *v10;
      *v10 = *v11;
      *v11 = v14;
LABEL_8:
      ++v10;
    }
    sub_155EB40(v10, v4, v22);
    result = (char *)v10 - (char *)a1;
    if ( (char *)v10 - (char *)a1 > 128 )
    {
      if ( v22 )
      {
        v4 = v10;
        continue;
      }
LABEL_19:
      v16 = result >> 3;
      v17 = ((result >> 3) - 2) >> 1;
      sub_155E9D0((__int64)a1, v17, result >> 3, a1[v17]);
      do
      {
        --v17;
        sub_155E9D0((__int64)a1, v17, v16, a1[v17]);
      }
      while ( v17 );
      do
      {
        v18 = *--v12;
        *v12 = *a1;
        result = sub_155E9D0((__int64)a1, 0, v12 - a1, v18);
      }
      while ( (char *)v12 - (char *)a1 > 8 );
    }
    return result;
  }
}
