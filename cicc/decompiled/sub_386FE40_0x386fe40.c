// Function: sub_386FE40
// Address: 0x386fe40
//
__int64 __fastcall sub_386FE40(__int64 a1, __int64 **a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v4; // r13
  __int64 **v5; // r14
  __int64 *v6; // r15
  __int64 v7; // r8
  __int64 v8; // r10
  char v9; // al
  __int64 *v10; // r12
  bool v11; // al
  __int64 v12; // r8
  __int64 *v13; // r9
  __int64 v14; // rsi
  __int64 *v15; // rcx
  __int64 v16; // r15
  __int64 **v17; // r14
  __int64 **v18; // r12
  char v19; // si
  __int64 **v20; // r13
  char v21; // al
  unsigned int v22; // eax
  unsigned int v23; // eax
  unsigned int v24; // r12d
  unsigned int v25; // eax
  __int64 *v26; // r12
  bool v27; // al
  __int64 *v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // r12
  __int64 v31; // r14
  __int64 *v32; // rcx
  __int64 v33; // [rsp+8h] [rbp-68h]
  __int64 **v34; // [rsp+18h] [rbp-58h]
  __int64 *v35; // [rsp+20h] [rbp-50h]
  __int64 *v36; // [rsp+28h] [rbp-48h]
  __int64 *v37; // [rsp+28h] [rbp-48h]
  __int64 v38; // [rsp+30h] [rbp-40h]
  __int64 *v39; // [rsp+30h] [rbp-40h]
  __int64 v40; // [rsp+30h] [rbp-40h]
  __int64 v41; // [rsp+30h] [rbp-40h]
  __int64 v42; // [rsp+30h] [rbp-40h]
  __int64 v43; // [rsp+38h] [rbp-38h]
  unsigned int v44; // [rsp+38h] [rbp-38h]
  unsigned int v45; // [rsp+38h] [rbp-38h]
  __int64 v46; // [rsp+38h] [rbp-38h]
  __int64 v47; // [rsp+38h] [rbp-38h]
  __int64 *v48; // [rsp+38h] [rbp-38h]
  __int64 *v49; // [rsp+38h] [rbp-38h]

  result = (__int64)a2 - a1;
  v34 = a2;
  v33 = a3;
  if ( (__int64)a2 - a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v20 = a2;
    goto LABEL_30;
  }
  while ( 2 )
  {
    v4 = *(__int64 **)(a1 + 8);
    --v33;
    v5 = (__int64 **)(a1 + 8 * (result >> 4));
    v6 = *v5;
    v7 = *v4;
    v8 = **v5;
    v9 = *(_BYTE *)(v8 + 8);
    if ( *(_BYTE *)(*v4 + 8) == 11 )
    {
      if ( v9 != 11 )
        goto LABEL_5;
      v41 = **v5;
      v46 = *v4;
      v24 = sub_1643030(v41);
      v25 = sub_1643030(v46);
      v7 = v46;
      v8 = v41;
      if ( v24 >= v25 )
        goto LABEL_5;
    }
    else if ( v9 != 11 )
    {
LABEL_5:
      v38 = v8;
      v10 = *(v34 - 1);
      v43 = *v10;
      v11 = sub_386EDD0(v7, *v10);
      v13 = *(__int64 **)a1;
      v14 = v43;
      if ( v11 )
      {
        *(_QWORD *)a1 = v4;
        *(_QWORD *)(a1 + 8) = v13;
        v15 = *(v34 - 1);
      }
      else
      {
        v49 = *(__int64 **)a1;
        if ( sub_386EDD0(v38, v14) )
        {
          *(_QWORD *)a1 = v10;
          v15 = v49;
          *(v34 - 1) = v49;
          v4 = *(__int64 **)a1;
          v13 = *(__int64 **)(a1 + 8);
        }
        else
        {
          *(_QWORD *)a1 = v6;
          *v5 = v49;
          v4 = *(__int64 **)a1;
          v13 = *(__int64 **)(a1 + 8);
          v15 = *(v34 - 1);
        }
      }
      goto LABEL_7;
    }
    v42 = v7;
    v26 = *(v34 - 1);
    v47 = *v26;
    v27 = sub_386EDD0(v8, *v26);
    v28 = *(__int64 **)a1;
    v29 = v47;
    v12 = v42;
    if ( v27 )
    {
      *(_QWORD *)a1 = v6;
      *v5 = v28;
      v4 = *(__int64 **)a1;
      v13 = *(__int64 **)(a1 + 8);
      v15 = *(v34 - 1);
    }
    else
    {
      v48 = *(__int64 **)a1;
      if ( sub_386EDD0(v42, v29) )
      {
        *(_QWORD *)a1 = v26;
        v15 = v48;
        *(v34 - 1) = v48;
        v4 = *(__int64 **)a1;
        v13 = *(__int64 **)(a1 + 8);
      }
      else
      {
        *(_QWORD *)a1 = v4;
        v13 = v48;
        *(_QWORD *)(a1 + 8) = v48;
        v15 = *(v34 - 1);
      }
    }
LABEL_7:
    v16 = *v4;
    v17 = (__int64 **)(a1 + 16);
    v18 = v34;
    v19 = *(_BYTE *)(*v4 + 8);
    while ( 1 )
    {
      v20 = v17 - 1;
      if ( *(_BYTE *)(*v13 + 8) == 11 )
        break;
      if ( v19 != 11 )
        goto LABEL_14;
LABEL_11:
      v13 = *v17++;
    }
    if ( v19 == 11 )
    {
      v35 = v15;
      v37 = v13;
      v40 = *v13;
      v45 = sub_1643030(v16);
      v23 = sub_1643030(v40);
      v13 = v37;
      v15 = v35;
      v19 = 11;
      if ( v45 < v23 )
        goto LABEL_11;
    }
LABEL_14:
    for ( --v18; ; v15 = *v18 )
    {
      v21 = *(_BYTE *)(*v15 + 8);
      if ( v19 == 11 )
        break;
      if ( v21 != 11 )
        goto LABEL_9;
LABEL_18:
      --v18;
    }
    if ( v21 == 11 )
    {
      v36 = v15;
      v39 = v13;
      v44 = sub_1643030(*v15);
      v22 = sub_1643030(v16);
      v13 = v39;
      v15 = v36;
      if ( v44 < v22 )
        goto LABEL_18;
    }
LABEL_9:
    if ( v18 > v20 )
    {
      *(v17 - 1) = v15;
      v15 = *(v18 - 1);
      *v18 = v13;
      v16 = **(_QWORD **)a1;
      v19 = *(_BYTE *)(v16 + 8);
      goto LABEL_11;
    }
    sub_386FE40(v17 - 1, v34, v33, v15, v12, v13);
    result = (__int64)v20 - a1;
    if ( (__int64)v20 - a1 > 128 )
    {
      if ( v33 )
      {
        v34 = v17 - 1;
        continue;
      }
LABEL_30:
      v30 = result >> 3;
      v31 = ((result >> 3) - 2) >> 1;
      sub_386F200(a1, v31, result >> 3, *(__int64 **)(a1 + 8 * v31));
      do
      {
        --v31;
        sub_386F200(a1, v31, v30, *(__int64 **)(a1 + 8 * v31));
      }
      while ( v31 );
      do
      {
        v32 = *--v20;
        *v20 = *(__int64 **)a1;
        result = sub_386F200(a1, 0, ((__int64)v20 - a1) >> 3, v32);
      }
      while ( (__int64)v20 - a1 > 8 );
    }
    return result;
  }
}
