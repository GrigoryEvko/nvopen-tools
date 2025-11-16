// Function: sub_11D29F0
// Address: 0x11d29f0
//
__int64 __fastcall sub_11D29F0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v5; // r14
  __int64 *v6; // r12
  bool v7; // al
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 *v12; // r12
  __int64 *v13; // r15
  __int64 *v14; // r13
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // r14
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 *v21; // [rsp+0h] [rbp-40h]
  __int64 v22; // [rsp+8h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v22 = a3;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  v5 = a2;
  if ( !a3 )
  {
    v14 = a2;
    goto LABEL_19;
  }
  v21 = a1 + 1;
  while ( 2 )
  {
    --v22;
    v6 = &a1[result >> 4];
    v7 = sub_B445A0(a1[1], *v6);
    v8 = *(v5 - 1);
    if ( !v7 )
    {
      if ( !sub_B445A0(a1[1], v8) )
      {
        if ( !sub_B445A0(*v6, *(v5 - 1)) )
          goto LABEL_6;
LABEL_24:
        v20 = *a1;
        *a1 = *(v5 - 1);
        *(v5 - 1) = v20;
        v10 = *a1;
        v11 = a1[1];
        goto LABEL_7;
      }
LABEL_17:
      v11 = *a1;
      v10 = a1[1];
      a1[1] = *a1;
      *a1 = v10;
      goto LABEL_7;
    }
    if ( !sub_B445A0(*v6, v8) )
    {
      if ( sub_B445A0(a1[1], *(v5 - 1)) )
        goto LABEL_24;
      goto LABEL_17;
    }
LABEL_6:
    v9 = *a1;
    *a1 = *v6;
    *v6 = v9;
    v10 = *a1;
    v11 = a1[1];
LABEL_7:
    v12 = v21;
    v13 = v5;
    while ( 1 )
    {
      v14 = v12;
      if ( sub_B445A0(v11, v10) )
        goto LABEL_8;
      do
        v15 = *--v13;
      while ( sub_B445A0(*a1, v15) );
      if ( v12 >= v13 )
        break;
      v16 = *v12;
      *v12 = *v13;
      *v13 = v16;
LABEL_8:
      v10 = *a1;
      v11 = v12[1];
      ++v12;
    }
    sub_11D29F0(v12, v5, v22);
    result = (char *)v12 - (char *)a1;
    if ( (char *)v12 - (char *)a1 > 128 )
    {
      if ( v22 )
      {
        v5 = v12;
        continue;
      }
LABEL_19:
      v17 = result >> 3;
      v18 = ((result >> 3) - 2) >> 1;
      sub_11D27D0((__int64)a1, v18, result >> 3, a1[v18]);
      do
      {
        --v18;
        sub_11D27D0((__int64)a1, v18, v17, a1[v18]);
      }
      while ( v18 );
      do
      {
        v19 = *--v14;
        *v14 = *a1;
        result = sub_11D27D0((__int64)a1, 0, v14 - a1, v19);
      }
      while ( (char *)v14 - (char *)a1 > 8 );
    }
    return result;
  }
}
