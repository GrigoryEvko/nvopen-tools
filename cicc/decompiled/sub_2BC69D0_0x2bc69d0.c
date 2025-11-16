// Function: sub_2BC69D0
// Address: 0x2bc69d0
//
__int64 __fastcall sub_2BC69D0(char *a1, __int64 *a2, __int64 a3, unsigned __int8 (__fastcall *a4)(__int64, __int64))
{
  __int64 result; // rax
  __int64 *v7; // r13
  bool v8; // zf
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 *v13; // r13
  __int64 *v14; // r15
  __int64 *v15; // r14
  __int64 v16; // rax
  __int64 *v17; // r14
  __int64 v18; // r13
  __int64 v19; // r15
  __int64 v20; // rcx
  __int64 *v21; // [rsp+8h] [rbp-48h]
  __int64 v22; // [rsp+10h] [rbp-40h]
  __int64 *v23; // [rsp+18h] [rbp-38h]

  result = (char *)a2 - a1;
  v23 = a2;
  v22 = a3;
  if ( (char *)a2 - a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v15 = a2;
    goto LABEL_23;
  }
  v21 = (__int64 *)(a1 + 8);
  while ( 2 )
  {
    --v22;
    v7 = (__int64 *)&a1[8 * (result >> 4)];
    v8 = a4(*((_QWORD *)a1 + 1), *v7) == 0;
    v9 = *(v23 - 1);
    if ( v8 )
    {
      if ( a4(*((_QWORD *)a1 + 1), v9) )
      {
LABEL_21:
        v12 = *(_QWORD *)a1;
        v11 = *((_QWORD *)a1 + 1);
        *((_QWORD *)a1 + 1) = *(_QWORD *)a1;
        *(_QWORD *)a1 = v11;
        goto LABEL_8;
      }
      v17 = v23;
      v8 = a4(*v7, *(v23 - 1)) == 0;
      v10 = *(_QWORD *)a1;
      if ( v8 )
        goto LABEL_7;
LABEL_19:
      *(_QWORD *)a1 = *(v17 - 1);
      *(v17 - 1) = v10;
      v11 = *(_QWORD *)a1;
      v12 = *((_QWORD *)a1 + 1);
      goto LABEL_8;
    }
    if ( !a4(*v7, v9) )
    {
      v17 = v23;
      if ( !a4(*((_QWORD *)a1 + 1), *(v23 - 1)) )
        goto LABEL_21;
      v10 = *(_QWORD *)a1;
      goto LABEL_19;
    }
    v10 = *(_QWORD *)a1;
LABEL_7:
    *(_QWORD *)a1 = *v7;
    *v7 = v10;
    v11 = *(_QWORD *)a1;
    v12 = *((_QWORD *)a1 + 1);
LABEL_8:
    v13 = v21;
    v14 = v23;
    while ( 1 )
    {
      v15 = v13;
      if ( a4(v12, v11) )
        goto LABEL_9;
      do
        --v14;
      while ( a4(*(_QWORD *)a1, *v14) );
      if ( v13 >= v14 )
        break;
      v16 = *v13;
      *v13 = *v14;
      *v14 = v16;
LABEL_9:
      v11 = *(_QWORD *)a1;
      v12 = v13[1];
      ++v13;
    }
    sub_2BC69D0(v13, v23, v22, a4);
    result = (char *)v13 - a1;
    if ( (char *)v13 - a1 > 128 )
    {
      if ( v22 )
      {
        v23 = v13;
        continue;
      }
LABEL_23:
      v18 = result >> 3;
      v19 = ((result >> 3) - 2) >> 1;
      sub_2BC6860((__int64)a1, v19, result >> 3, *(_QWORD *)&a1[8 * v19], a4);
      do
      {
        --v19;
        sub_2BC6860((__int64)a1, v19, v18, *(_QWORD *)&a1[8 * v19], a4);
      }
      while ( v19 );
      do
      {
        v20 = *--v15;
        *v15 = *(_QWORD *)a1;
        result = sub_2BC6860((__int64)a1, 0, ((char *)v15 - a1) >> 3, v20, a4);
      }
      while ( (char *)v15 - a1 > 8 );
    }
    return result;
  }
}
