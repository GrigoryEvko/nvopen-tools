// Function: sub_31B1840
// Address: 0x31b1840
//
void __fastcall sub_31B1840(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi
  __int64 v3; // rsi
  __int64 v5; // r15
  __int64 *v6; // rdx
  bool v7; // al
  __int64 *v8; // rbx
  __int64 v9; // r13
  __int64 v10; // r9
  __int64 v11; // r12
  __int64 v12; // rsi
  unsigned __int8 v13; // di
  unsigned __int8 v14; // al
  unsigned __int8 v15; // al
  __int64 v16; // rbx
  __int64 v17; // r13
  __int64 *v18; // r12
  __int64 v19; // rdi
  unsigned __int8 v20; // r8
  unsigned __int8 v21; // al
  unsigned __int8 v22; // al
  bool v23; // al
  bool v24; // [rsp-79h] [rbp-79h]
  __int64 v25; // [rsp-78h] [rbp-78h]
  __int64 v26; // [rsp-70h] [rbp-70h]
  __int64 v27; // [rsp-60h] [rbp-60h]
  __int64 v28; // [rsp-60h] [rbp-60h]
  __int64 v29; // [rsp-58h] [rbp-58h]
  unsigned __int8 v30; // [rsp-50h] [rbp-50h]
  unsigned __int8 v31; // [rsp-50h] [rbp-50h]
  __int64 v32; // [rsp-48h] [rbp-48h]
  __int64 v33; // [rsp-40h] [rbp-40h]

  v2 = a2 - a1;
  if ( v2 <= 8 )
    return;
  v3 = v2 >> 3;
  v25 = (v3 - 2) / 2;
  v24 = (v3 & 1) == 0;
  v32 = (v3 - 1) >> 1;
  v5 = v25;
  while ( 2 )
  {
    v6 = (__int64 *)(a1 + 8 * v5);
    v33 = *v6;
    if ( v5 >= v32 )
    {
      if ( v5 == v25 )
      {
        v16 = v5;
        if ( v24 )
        {
LABEL_33:
          v16 = 2 * v16 + 1;
          *v6 = *(_QWORD *)(a1 + 8 * v16);
          v6 = (__int64 *)(a1 + 8 * v16);
          goto LABEL_15;
        }
      }
LABEL_22:
      *v6 = v33;
      if ( !v5 )
        return;
      goto LABEL_23;
    }
    v26 = v5;
    while ( 1 )
    {
      v10 = 2 * (v5 + 1);
      v11 = a1 + 16 * (v5 + 1);
      v9 = v10 - 1;
      v8 = (__int64 *)(a1 + 8 * (v10 - 1));
      v12 = *(_QWORD *)(*v8 + 8);
      v13 = (unsigned int)**(unsigned __int8 **)(*(_QWORD *)(*(_QWORD *)v11 + 8LL) + 16LL) - 30 <= 0xA;
      v14 = (unsigned int)**(unsigned __int8 **)(v12 + 16) - 30 <= 0xA;
      if ( v13 != v14 )
      {
        v7 = v13 > v14;
        goto LABEL_6;
      }
      v27 = *(_QWORD *)(*(_QWORD *)v11 + 8LL);
      v30 = sub_318B700(v27);
      v15 = sub_318B700(v12);
      v10 = 2 * (v5 + 1);
      if ( v30 != v15 )
      {
        v7 = v30 < v15;
LABEL_6:
        if ( v7 )
          goto LABEL_8;
LABEL_7:
        v8 = (__int64 *)(a1 + 16 * (v5 + 1));
        v9 = v10;
        goto LABEL_8;
      }
      v23 = sub_B445A0(*(_QWORD *)(v12 + 16), *(_QWORD *)(v27 + 16));
      v10 = 2 * (v5 + 1);
      if ( !v23 )
        goto LABEL_7;
LABEL_8:
      *(_QWORD *)(a1 + 8 * v5) = *v8;
      if ( v9 >= v32 )
        break;
      v5 = v9;
    }
    v6 = v8;
    v5 = v26;
    v16 = v9;
    if ( v25 == v9 && v24 )
      goto LABEL_33;
LABEL_15:
    v17 = (v16 - 1) >> 1;
    if ( v5 >= v16 )
    {
      *v6 = v33;
      goto LABEL_23;
    }
    while ( 1 )
    {
      v18 = (__int64 *)(a1 + 8 * v17);
      v19 = *v18;
      v20 = (unsigned int)**(unsigned __int8 **)(*(_QWORD *)(*v18 + 8) + 16LL) - 30 <= 0xA;
      v21 = (unsigned int)**(unsigned __int8 **)(*(_QWORD *)(v33 + 8) + 16LL) - 30 <= 0xA;
      if ( v20 == v21 )
      {
        v28 = *(_QWORD *)(*v18 + 8);
        v29 = *(_QWORD *)(v33 + 8);
        v31 = sub_318B700(v28);
        v22 = sub_318B700(v29);
        if ( v31 == v22 )
        {
          if ( !sub_B445A0(*(_QWORD *)(v29 + 16), *(_QWORD *)(v28 + 16)) )
            goto LABEL_21;
        }
        else if ( v31 >= v22 )
        {
LABEL_21:
          v6 = (__int64 *)(a1 + 8 * v16);
          goto LABEL_22;
        }
        v19 = *v18;
      }
      else if ( v20 <= v21 )
      {
        goto LABEL_21;
      }
      *(_QWORD *)(a1 + 8 * v16) = v19;
      v16 = v17;
      if ( v5 >= v17 )
        break;
      v17 = (v17 - 1) / 2;
    }
    *v18 = v33;
    if ( v5 )
    {
LABEL_23:
      --v5;
      continue;
    }
    break;
  }
}
