// Function: sub_27C0B00
// Address: 0x27c0b00
//
__int64 __fastcall sub_27C0B00(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // rsi
  __int64 *v7; // r12
  char v8; // al
  __int64 v9; // rdx
  char v10; // al
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // r11
  __int64 v14; // rsi
  __int64 *v15; // r14
  __int64 *v16; // r13
  __int64 v17; // rax
  char v18; // al
  __int64 v19; // rdx
  __int64 *v20; // r12
  char v21; // al
  __int64 v22; // rsi
  __int64 v23; // rdx
  char v24; // al
  __int64 v25; // rax
  __int64 v26; // r13
  __int64 v27; // r14
  __int64 v28; // rcx
  char v29; // al
  __int64 v30; // rax
  __int64 v31; // [rsp+8h] [rbp-48h]
  __int64 *v32; // [rsp+10h] [rbp-40h]
  __int64 *v33; // [rsp+18h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v32 = a2;
  v31 = a3;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v33 = a2;
    goto LABEL_31;
  }
  while ( 2 )
  {
    v6 = a1[1];
    --v31;
    v7 = &a1[result >> 4];
    if ( *v7 != v6 )
    {
      sub_B196A0(*(_QWORD *)(a4 + 16), v6, *v7);
      if ( v8 )
      {
        v22 = *v7;
        v23 = *(v32 - 1);
        if ( v23 != *v7 )
        {
          sub_B196A0(*(_QWORD *)(a4 + 16), v22, v23);
          if ( v24 )
          {
            v25 = *a1;
            *a1 = *v7;
            *v7 = v25;
            v13 = *a1;
            v14 = a1[1];
            goto LABEL_13;
          }
          v22 = *(v32 - 1);
        }
        v13 = a1[1];
        if ( v22 != v13 )
        {
          sub_B196A0(*(_QWORD *)(a4 + 16), a1[1], v22);
          if ( v29 )
            goto LABEL_38;
          v13 = a1[1];
        }
        v14 = *a1;
        *a1 = v13;
        a1[1] = v14;
        goto LABEL_13;
      }
      v6 = a1[1];
    }
    v9 = *(v32 - 1);
    if ( v6 != v9 )
    {
      sub_B196A0(*(_QWORD *)(a4 + 16), v6, v9);
      if ( v10 )
      {
        v14 = *a1;
        v13 = a1[1];
        a1[1] = *a1;
        *a1 = v13;
        goto LABEL_13;
      }
      v9 = *(v32 - 1);
    }
    if ( *v7 == v9 )
      goto LABEL_12;
    sub_B196A0(*(_QWORD *)(a4 + 16), *v7, v9);
    if ( v11 )
    {
LABEL_38:
      v30 = *a1;
      *a1 = *(v32 - 1);
      *(v32 - 1) = v30;
      v13 = *a1;
      v14 = a1[1];
      goto LABEL_13;
    }
    v9 = *v7;
LABEL_12:
    v12 = *a1;
    *a1 = v9;
    *v7 = v12;
    v13 = *a1;
    v14 = a1[1];
LABEL_13:
    v15 = a1 + 1;
    v16 = v32;
    while ( 1 )
    {
      v33 = v15;
      if ( v13 == v14 )
        break;
      sub_B196A0(*(_QWORD *)(a4 + 16), v14, v13);
      if ( !v18 )
      {
        v14 = *a1;
        break;
      }
LABEL_16:
      v13 = *a1;
      v14 = v15[1];
      ++v15;
    }
    v19 = *(v16 - 1);
    v20 = v16 - 1;
    if ( v19 != v14 )
    {
      while ( 1 )
      {
        sub_B196A0(*(_QWORD *)(a4 + 16), v14, v19);
        if ( !v21 )
          break;
        v14 = *a1;
        v19 = *--v20;
        if ( v19 == *a1 )
        {
          if ( v20 > v15 )
            goto LABEL_15;
          goto LABEL_24;
        }
      }
    }
    if ( v20 > v15 )
    {
LABEL_15:
      v17 = *v15;
      v16 = v20;
      *v15 = *v20;
      *v20 = v17;
      goto LABEL_16;
    }
LABEL_24:
    sub_27C0B00(v15, v32, v31, a4);
    result = (char *)v15 - (char *)a1;
    if ( (char *)v15 - (char *)a1 > 128 )
    {
      if ( v31 )
      {
        v32 = v15;
        continue;
      }
LABEL_31:
      v26 = result >> 3;
      v27 = ((result >> 3) - 2) >> 1;
      sub_27C04A0((__int64)a1, v27, result >> 3, a1[v27], a4);
      do
      {
        --v27;
        sub_27C04A0((__int64)a1, v27, v26, a1[v27], a4);
      }
      while ( v27 );
      do
      {
        v28 = *--v33;
        *v33 = *a1;
        result = sub_27C04A0((__int64)a1, 0, v33 - a1, v28, a4);
      }
      while ( (char *)v33 - (char *)a1 > 8 );
    }
    return result;
  }
}
