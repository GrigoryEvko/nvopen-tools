// Function: sub_F06D00
// Address: 0xf06d00
//
__int64 __fastcall sub_F06D00(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v5; // r14
  __int64 *v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 *v10; // r12
  __int64 *v11; // r15
  __int64 *v12; // r13
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // r14
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 *v19; // [rsp+0h] [rbp-40h]
  __int64 v20; // [rsp+8h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v20 = a3;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  v5 = a2;
  if ( !a3 )
  {
    v12 = a2;
    goto LABEL_19;
  }
  v19 = a1 + 1;
  while ( 2 )
  {
    --v20;
    v6 = &a1[result >> 4];
    if ( !sub_B445A0(*v6, a1[1]) )
    {
      if ( !sub_B445A0(*(v5 - 1), a1[1]) )
      {
        if ( !sub_B445A0(*(v5 - 1), *v6) )
          goto LABEL_6;
LABEL_24:
        v18 = *a1;
        *a1 = *(v5 - 1);
        *(v5 - 1) = v18;
        v8 = *a1;
        v9 = a1[1];
        goto LABEL_7;
      }
LABEL_17:
      v9 = *a1;
      v8 = a1[1];
      a1[1] = *a1;
      *a1 = v8;
      goto LABEL_7;
    }
    if ( !sub_B445A0(*(v5 - 1), *v6) )
    {
      if ( sub_B445A0(*(v5 - 1), a1[1]) )
        goto LABEL_24;
      goto LABEL_17;
    }
LABEL_6:
    v7 = *a1;
    *a1 = *v6;
    *v6 = v7;
    v8 = *a1;
    v9 = a1[1];
LABEL_7:
    v10 = v19;
    v11 = v5;
    while ( 1 )
    {
      v12 = v10;
      if ( sub_B445A0(v8, v9) )
        goto LABEL_8;
      do
        v13 = *--v11;
      while ( sub_B445A0(v13, *a1) );
      if ( v10 >= v11 )
        break;
      v14 = *v10;
      *v10 = *v11;
      *v11 = v14;
LABEL_8:
      v8 = *a1;
      v9 = v10[1];
      ++v10;
    }
    sub_F06D00(v10, v5, v20);
    result = (char *)v10 - (char *)a1;
    if ( (char *)v10 - (char *)a1 > 128 )
    {
      if ( v20 )
      {
        v5 = v10;
        continue;
      }
LABEL_19:
      v15 = result >> 3;
      v16 = ((result >> 3) - 2) >> 1;
      sub_F066F0((__int64)a1, v16, result >> 3, a1[v16]);
      do
      {
        --v16;
        sub_F066F0((__int64)a1, v16, v15, a1[v16]);
      }
      while ( v16 );
      do
      {
        v17 = *--v12;
        *v12 = *a1;
        result = sub_F066F0((__int64)a1, 0, v12 - a1, v17);
      }
      while ( (char *)v12 - (char *)a1 > 8 );
    }
    return result;
  }
}
