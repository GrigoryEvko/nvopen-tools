// Function: sub_2894BE0
// Address: 0x2894be0
//
__int64 __fastcall sub_2894BE0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
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
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // r14
  __int64 v18; // rcx
  __int64 *v19; // r14
  __int64 v20; // [rsp+10h] [rbp-40h]
  __int64 *v21; // [rsp+18h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v21 = a2;
  v20 = a3;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v14 = a2;
    goto LABEL_19;
  }
  while ( 2 )
  {
    v6 = &a1[result >> 4];
    --v20;
    v7 = (unsigned __int8)sub_B19DB0(*(_QWORD *)(a4 + 40), a1[1], *v6) == 0;
    v8 = *(v21 - 1);
    if ( v7 )
    {
      if ( !(unsigned __int8)sub_B19DB0(*(_QWORD *)(a4 + 40), a1[1], v8) )
      {
        v19 = v21;
        v7 = (unsigned __int8)sub_B19DB0(*(_QWORD *)(a4 + 40), *v6, *(v21 - 1)) == 0;
        v9 = *a1;
        if ( v7 )
          goto LABEL_6;
LABEL_25:
        *a1 = *(v19 - 1);
        *(v19 - 1) = v9;
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
    if ( !(unsigned __int8)sub_B19DB0(*(_QWORD *)(a4 + 40), *v6, v8) )
    {
      v19 = v21;
      if ( (unsigned __int8)sub_B19DB0(*(_QWORD *)(a4 + 40), a1[1], *(v21 - 1)) )
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
    v13 = v21;
    while ( 1 )
    {
      v14 = v12;
      if ( (unsigned __int8)sub_B19DB0(*(_QWORD *)(a4 + 40), v11, v10) )
        goto LABEL_8;
      do
        --v13;
      while ( (unsigned __int8)sub_B19DB0(*(_QWORD *)(a4 + 40), *a1, *v13) );
      if ( v12 >= v13 )
        break;
      v15 = *v12;
      *v12 = *v13;
      *v13 = v15;
LABEL_8:
      v10 = *a1;
      v11 = v12[1];
      ++v12;
    }
    sub_2894BE0(v12, v21, v20, a4);
    result = (char *)v12 - (char *)a1;
    if ( (char *)v12 - (char *)a1 > 128 )
    {
      if ( v20 )
      {
        v21 = v12;
        continue;
      }
LABEL_19:
      v16 = result >> 3;
      v17 = ((result >> 3) - 2) >> 1;
      sub_28942B0((__int64)a1, v17, result >> 3, a1[v17], a4);
      do
      {
        --v17;
        sub_28942B0((__int64)a1, v17, v16, a1[v17], a4);
      }
      while ( v17 );
      do
      {
        v18 = *--v14;
        *v14 = *a1;
        result = sub_28942B0((__int64)a1, 0, v14 - a1, v18, a4);
      }
      while ( (char *)v14 - (char *)a1 > 8 );
    }
    return result;
  }
}
