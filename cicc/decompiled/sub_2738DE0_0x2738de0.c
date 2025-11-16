// Function: sub_2738DE0
// Address: 0x2738de0
//
__int64 __fastcall sub_2738DE0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v6; // r12
  bool v7; // zf
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 *v12; // r12
  __int64 *v13; // r14
  __int64 *v14; // r13
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // r14
  __int64 v19; // rcx
  __int64 *v20; // r14
  __int64 v21; // [rsp+10h] [rbp-40h]
  __int64 *v22; // [rsp+18h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v22 = a2;
  v21 = a3;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v14 = a2;
    goto LABEL_19;
  }
  while ( 2 )
  {
    --v21;
    v6 = &a1[result >> 4];
    v7 = (unsigned __int8)sub_B19DB0(a4, a1[1], *v6) == 0;
    v8 = *(v22 - 1);
    if ( v7 )
    {
      if ( !(unsigned __int8)sub_B19DB0(a4, a1[1], v8) )
      {
        v20 = v22;
        v7 = (unsigned __int8)sub_B19DB0(a4, *v6, *(v22 - 1)) == 0;
        v9 = *a1;
        if ( v7 )
          goto LABEL_6;
LABEL_25:
        *a1 = *(v20 - 1);
        *(v20 - 1) = v9;
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
    if ( !(unsigned __int8)sub_B19DB0(a4, *v6, v8) )
    {
      v20 = v22;
      if ( (unsigned __int8)sub_B19DB0(a4, a1[1], *(v22 - 1)) )
      {
        v9 = *a1;
        goto LABEL_25;
      }
      goto LABEL_17;
    }
    v9 = *a1;
LABEL_6:
    *a1 = *v6;
    *v6 = v9;
    v10 = *a1;
    v11 = a1[1];
LABEL_7:
    v12 = a1 + 1;
    v13 = v22;
    while ( 1 )
    {
      v14 = v12;
      if ( (unsigned __int8)sub_B19DB0(a4, v11, v10) )
        goto LABEL_8;
      do
        v15 = *--v13;
      while ( (unsigned __int8)sub_B19DB0(a4, *a1, v15) );
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
    sub_2738DE0(v12, v22, v21, a4);
    result = (char *)v12 - (char *)a1;
    if ( (char *)v12 - (char *)a1 > 128 )
    {
      if ( v21 )
      {
        v22 = v12;
        continue;
      }
LABEL_19:
      v17 = result >> 3;
      v18 = ((result >> 3) - 2) >> 1;
      sub_2737EB0((__int64)a1, v18, result >> 3, a1[v18], a4);
      do
      {
        --v18;
        sub_2737EB0((__int64)a1, v18, v17, a1[v18], a4);
      }
      while ( v18 );
      do
      {
        v19 = *--v14;
        *v14 = *a1;
        result = sub_2737EB0((__int64)a1, 0, v14 - a1, v19, a4);
      }
      while ( (char *)v14 - (char *)a1 > 8 );
    }
    return result;
  }
}
