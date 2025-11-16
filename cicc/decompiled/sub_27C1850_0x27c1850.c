// Function: sub_27C1850
// Address: 0x27c1850
//
__int64 __fastcall sub_27C1850(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // r12
  __int64 *v7; // r13
  __int64 v8; // r14
  char v9; // al
  __int64 v10; // r14
  __int64 v11; // r12
  char v12; // al
  __int64 v13; // rax
  __int64 v14; // r10
  __int64 *v15; // r13
  char v16; // al
  __int64 *v17; // r12
  __int64 v18; // r13
  char v19; // al
  __int64 v20; // rax
  char v21; // al
  char v22; // al
  __int64 v23; // r14
  char v24; // al
  __int64 v25; // r12
  __int64 v26; // r13
  __int64 v27; // rcx
  char v28; // al
  __int64 v29; // r14
  char v30; // al
  __int64 v31; // rax
  char v32; // al
  __int64 v33; // r12
  char v34; // al
  char v35; // al
  char v36; // al
  char v37; // al
  __int64 v38; // rax
  __int64 v39; // [rsp+8h] [rbp-58h]
  __int64 *v40; // [rsp+10h] [rbp-50h]
  __int64 *v41; // [rsp+18h] [rbp-48h]
  __int64 v42; // [rsp+20h] [rbp-40h]
  __int64 *i; // [rsp+28h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v40 = a2;
  v39 = a3;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v41 = a2;
    goto LABEL_28;
  }
  while ( 2 )
  {
    v6 = a1[1];
    --v39;
    v7 = &a1[result >> 4];
    v8 = *v7;
    if ( *v7 == v6 )
    {
      v23 = *(v40 - 1);
      if ( v6 == v23 )
        goto LABEL_48;
LABEL_25:
      sub_B196A0(*(_QWORD *)(a4 + 16), v6, v23);
      if ( v24 )
      {
        v14 = *a1;
        v10 = a1[1];
        a1[1] = *a1;
        *a1 = v10;
        goto LABEL_8;
      }
      sub_B196A0(*(_QWORD *)(a4 + 16), v23, v6);
      if ( !v36 )
        goto LABEL_49;
      v6 = *(v40 - 1);
LABEL_34:
      v29 = *v7;
      if ( *v7 != v6 )
      {
        sub_B196A0(*(_QWORD *)(a4 + 16), *v7, v6);
        if ( v30 )
          goto LABEL_36;
        sub_B196A0(*(_QWORD *)(a4 + 16), v6, v29);
        if ( !v37 )
LABEL_49:
          BUG();
        v6 = *v7;
      }
LABEL_48:
      v38 = *a1;
      *a1 = v6;
      *v7 = v38;
      v10 = *a1;
      v14 = a1[1];
      goto LABEL_8;
    }
    sub_B196A0(*(_QWORD *)(a4 + 16), v6, *v7);
    if ( !v9 )
    {
      sub_B196A0(*(_QWORD *)(a4 + 16), v8, v6);
      if ( !v28 )
        goto LABEL_49;
      v6 = a1[1];
      v23 = *(v40 - 1);
      if ( v23 == v6 )
        goto LABEL_34;
      goto LABEL_25;
    }
    v10 = *v7;
    v11 = *(v40 - 1);
    if ( v11 == *v7 )
    {
LABEL_39:
      v33 = a1[1];
      if ( v33 == v10 )
      {
LABEL_43:
        v14 = *a1;
        *a1 = v10;
        a1[1] = v14;
        goto LABEL_8;
      }
      sub_B196A0(*(_QWORD *)(a4 + 16), a1[1], v10);
      if ( !v34 )
      {
        sub_B196A0(*(_QWORD *)(a4 + 16), v10, v33);
        if ( !v35 )
          goto LABEL_49;
        v10 = a1[1];
        goto LABEL_43;
      }
LABEL_36:
      v31 = *a1;
      *a1 = *(v40 - 1);
      *(v40 - 1) = v31;
      v10 = *a1;
      v14 = a1[1];
      goto LABEL_8;
    }
    sub_B196A0(*(_QWORD *)(a4 + 16), *v7, *(v40 - 1));
    if ( !v12 )
    {
      sub_B196A0(*(_QWORD *)(a4 + 16), v11, v10);
      if ( !v32 )
        goto LABEL_49;
      v10 = *(v40 - 1);
      goto LABEL_39;
    }
    v13 = *a1;
    *a1 = *v7;
    *v7 = v13;
    v10 = *a1;
    v14 = a1[1];
LABEL_8:
    v15 = v40;
    for ( i = a1 + 1; ; ++i )
    {
      v41 = i;
      if ( v14 != v10 )
      {
        v42 = v14;
        sub_B196A0(*(_QWORD *)(a4 + 16), v14, v10);
        if ( v16 )
          goto LABEL_10;
        sub_B196A0(*(_QWORD *)(a4 + 16), v10, v42);
        if ( !v22 )
          goto LABEL_49;
        v10 = *a1;
      }
      v17 = v15 - 1;
      v18 = *(v15 - 1);
      if ( v18 != v10 )
      {
        while ( 1 )
        {
          sub_B196A0(*(_QWORD *)(a4 + 16), v10, v18);
          if ( !v19 )
            break;
          v10 = *a1;
          v18 = *--v17;
          if ( v18 == *a1 )
            goto LABEL_15;
        }
        sub_B196A0(*(_QWORD *)(a4 + 16), v18, v10);
        if ( !v21 )
          goto LABEL_49;
      }
LABEL_15:
      if ( v17 <= i )
        break;
      v15 = v17;
      v20 = *i;
      *i = *v17;
      *v17 = v20;
LABEL_10:
      v10 = *a1;
      v14 = i[1];
    }
    sub_27C1850(i, v40, v39, a4);
    result = (char *)i - (char *)a1;
    if ( (char *)i - (char *)a1 <= 128 )
      return result;
    if ( v39 )
    {
      v40 = i;
      continue;
    }
    break;
  }
LABEL_28:
  v25 = result >> 3;
  v26 = ((result >> 3) - 2) >> 1;
  sub_27C08A0((__int64)a1, v26, result >> 3, a1[v26], a4);
  do
  {
    --v26;
    sub_27C08A0((__int64)a1, v26, v25, a1[v26], a4);
  }
  while ( v26 );
  do
  {
    v27 = *--v41;
    *v41 = *a1;
    result = sub_27C08A0((__int64)a1, 0, v41 - a1, v27, a4);
  }
  while ( (char *)v41 - (char *)a1 > 8 );
  return result;
}
