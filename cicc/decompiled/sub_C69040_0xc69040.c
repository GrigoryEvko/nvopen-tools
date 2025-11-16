// Function: sub_C69040
// Address: 0xc69040
//
__int64 __fastcall sub_C69040(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v5; // r9
  __int64 v6; // r8
  __int64 v7; // r14
  __int64 *v8; // rbx
  __int64 v9; // r12
  __int64 v10; // r10
  __int64 *v11; // r11
  __int64 v12; // rcx
  size_t v13; // r13
  size_t v14; // r15
  size_t v15; // rdx
  int v16; // eax
  __int64 v17; // r14
  __int64 v18; // r8
  __int64 v19; // r15
  __int64 v20; // rcx
  __int64 *v21; // r13
  size_t v22; // rbx
  __int64 v23; // r9
  size_t v24; // rdx
  size_t v25; // r12
  int v26; // eax
  __int64 v29; // [rsp+8h] [rbp-78h]
  __int64 v31; // [rsp+18h] [rbp-68h]
  __int64 v32; // [rsp+28h] [rbp-58h]
  __int64 v33; // [rsp+30h] [rbp-50h]
  __int64 *v34; // [rsp+38h] [rbp-48h]
  __int64 v35; // [rsp+38h] [rbp-48h]
  __int64 v36; // [rsp+40h] [rbp-40h]
  __int64 v37; // [rsp+48h] [rbp-38h]
  __int64 v38; // [rsp+48h] [rbp-38h]

  v4 = a1;
  v29 = a3 & 1;
  v31 = (a3 - 1) / 2;
  if ( a2 >= v31 )
  {
    v8 = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_25;
    v9 = a2;
    goto LABEL_28;
  }
  v5 = a2;
  v6 = a1;
  while ( 1 )
  {
    v9 = 2 * (v5 + 1);
    v10 = v9 - 1;
    v8 = (__int64 *)(v6 + 16 * (v5 + 1));
    v11 = (__int64 *)(v6 + 8 * (v9 - 1));
    v7 = *v8;
    v12 = *v11;
    v13 = *(_QWORD *)(*v8 + 16);
    v14 = *(_QWORD *)(*v11 + 16);
    v15 = v14;
    if ( v13 <= v14 )
      v15 = *(_QWORD *)(*v8 + 16);
    if ( !v15 )
      break;
    v32 = v6;
    v33 = v5;
    v34 = (__int64 *)(v6 + 8 * (v9 - 1));
    v37 = *v11;
    v16 = memcmp(*(const void **)(v7 + 8), *(const void **)(v12 + 8), v15);
    v12 = v37;
    v10 = v9 - 1;
    v11 = v34;
    v5 = v33;
    v6 = v32;
    if ( !v16 )
      break;
    if ( v16 < 0 )
    {
      v7 = v37;
      v8 = v34;
      --v9;
    }
    *(_QWORD *)(v32 + 8 * v33) = v7;
    if ( v9 >= v31 )
      goto LABEL_15;
LABEL_7:
    v5 = v9;
  }
  if ( v13 != v14 && v13 < v14 )
  {
    v7 = v12;
    v8 = v11;
    v9 = v10;
  }
  *(_QWORD *)(v6 + 8 * v5) = v7;
  if ( v9 < v31 )
    goto LABEL_7;
LABEL_15:
  v4 = v6;
  if ( !v29 )
  {
LABEL_28:
    if ( (a3 - 2) / 2 == v9 )
    {
      v9 = 2 * v9 + 1;
      *v8 = *(_QWORD *)(v4 + 8 * v9);
      v8 = (__int64 *)(v4 + 8 * v9);
    }
  }
  v17 = (v9 - 1) / 2;
  if ( v9 > a2 )
  {
    v18 = a4;
    v19 = v9;
    v20 = v4;
    while ( 1 )
    {
      v21 = (__int64 *)(v20 + 8 * v17);
      v22 = *(_QWORD *)(v18 + 16);
      v23 = *v21;
      v24 = v22;
      v25 = *(_QWORD *)(*v21 + 16);
      if ( v25 <= v22 )
        v24 = *(_QWORD *)(*v21 + 16);
      if ( v24
        && (v35 = v20,
            v36 = v18,
            v38 = *v21,
            v26 = memcmp(*(const void **)(v23 + 8), *(const void **)(v18 + 8), v24),
            v23 = v38,
            v18 = v36,
            v20 = v35,
            v26) )
      {
        if ( v26 >= 0 )
          goto LABEL_24;
      }
      else if ( v25 == v22 || v25 >= v22 )
      {
LABEL_24:
        v8 = (__int64 *)(v20 + 8 * v19);
        goto LABEL_25;
      }
      *(_QWORD *)(v20 + 8 * v19) = v23;
      v19 = v17;
      if ( a2 >= v17 )
        break;
      v17 = (v17 - 1) / 2;
    }
    v8 = v21;
  }
LABEL_25:
  *v8 = a4;
  return a4;
}
