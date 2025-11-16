// Function: sub_31AFAC0
// Address: 0x31afac0
//
__int64 __fastcall sub_31AFAC0(char *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v4; // r14
  char *v5; // r12
  bool v6; // al
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 *v11; // r12
  __int64 *v12; // r15
  __int64 *v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // r14
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // [rsp+8h] [rbp-38h]

  result = (char *)a2 - a1;
  v20 = a3;
  if ( (char *)a2 - a1 <= 128 )
    return result;
  v4 = a2;
  if ( !a3 )
  {
    v13 = a2;
    goto LABEL_18;
  }
  while ( 2 )
  {
    --v20;
    v5 = &a1[8 * (result >> 4)];
    v6 = sub_B445A0(*(_QWORD *)(*((_QWORD *)a1 + 1) + 16LL), *(_QWORD *)(*(_QWORD *)v5 + 16LL));
    v7 = *(_QWORD *)(*(v4 - 1) + 16);
    if ( !v6 )
    {
      if ( !sub_B445A0(*(_QWORD *)(*((_QWORD *)a1 + 1) + 16LL), v7) )
      {
        if ( !sub_B445A0(*(_QWORD *)(*(_QWORD *)v5 + 16LL), *(_QWORD *)(*(v4 - 1) + 16)) )
          goto LABEL_5;
LABEL_23:
        v19 = *(_QWORD *)a1;
        *(_QWORD *)a1 = *(v4 - 1);
        *(v4 - 1) = v19;
        v9 = *(_QWORD *)a1;
        v10 = *((_QWORD *)a1 + 1);
        goto LABEL_6;
      }
LABEL_16:
      v10 = *(_QWORD *)a1;
      v9 = *((_QWORD *)a1 + 1);
      *((_QWORD *)a1 + 1) = *(_QWORD *)a1;
      *(_QWORD *)a1 = v9;
      goto LABEL_6;
    }
    if ( !sub_B445A0(*(_QWORD *)(*(_QWORD *)v5 + 16LL), v7) )
    {
      if ( sub_B445A0(*(_QWORD *)(*((_QWORD *)a1 + 1) + 16LL), *(_QWORD *)(*(v4 - 1) + 16)) )
        goto LABEL_23;
      goto LABEL_16;
    }
LABEL_5:
    v8 = *(_QWORD *)a1;
    *(_QWORD *)a1 = *(_QWORD *)v5;
    *(_QWORD *)v5 = v8;
    v9 = *(_QWORD *)a1;
    v10 = *((_QWORD *)a1 + 1);
LABEL_6:
    v11 = (__int64 *)(a1 + 8);
    v12 = v4;
    while ( 1 )
    {
      v13 = v11;
      if ( sub_B445A0(*(_QWORD *)(v10 + 16), *(_QWORD *)(v9 + 16)) )
        goto LABEL_7;
      do
        v14 = *--v12;
      while ( sub_B445A0(*(_QWORD *)(*(_QWORD *)a1 + 16LL), *(_QWORD *)(v14 + 16)) );
      if ( v11 >= v12 )
        break;
      v15 = *v11;
      *v11 = *v12;
      *v12 = v15;
LABEL_7:
      v9 = *(_QWORD *)a1;
      v10 = v11[1];
      ++v11;
    }
    sub_31AFAC0(v11, v4, v20);
    result = (char *)v11 - a1;
    if ( (char *)v11 - a1 > 128 )
    {
      if ( v20 )
      {
        v4 = v11;
        continue;
      }
LABEL_18:
      v16 = result >> 3;
      v17 = ((result >> 3) - 2) >> 1;
      sub_31AF930((__int64)a1, v17, result >> 3, *(_QWORD *)&a1[8 * v17]);
      do
      {
        --v17;
        sub_31AF930((__int64)a1, v17, v16, *(_QWORD *)&a1[8 * v17]);
      }
      while ( v17 );
      do
      {
        v18 = *--v13;
        *v13 = *(_QWORD *)a1;
        result = sub_31AF930((__int64)a1, 0, ((char *)v13 - a1) >> 3, v18);
      }
      while ( (char *)v13 - a1 > 8 );
    }
    return result;
  }
}
