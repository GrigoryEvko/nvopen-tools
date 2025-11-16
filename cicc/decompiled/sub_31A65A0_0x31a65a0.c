// Function: sub_31A65A0
// Address: 0x31a65a0
//
bool __fastcall sub_31A65A0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rbx
  __int64 v3; // rax
  _QWORD *v4; // r13
  signed __int64 v5; // rax
  __int64 v6; // rax
  _QWORD *v7; // r15
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r15
  bool result; // al
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 *v19; // r12
  __int64 *v20; // r8
  __int64 v21; // [rsp+0h] [rbp-50h]
  __int64 v22; // [rsp+0h] [rbp-50h]
  __int64 v23; // [rsp+0h] [rbp-50h]
  __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+8h] [rbp-48h]
  __int64 v27; // [rsp+8h] [rbp-48h]
  __int64 *v28; // [rsp+10h] [rbp-40h]
  __int64 *v29; // [rsp+10h] [rbp-40h]
  __int64 *v30; // [rsp+10h] [rbp-40h]
  __int64 *v31; // [rsp+10h] [rbp-40h]
  __int64 v32; // [rsp+10h] [rbp-40h]
  __int64 v33; // [rsp+10h] [rbp-40h]
  _QWORD *v34; // [rsp+18h] [rbp-38h]
  __int64 *v35; // [rsp+18h] [rbp-38h]
  __int64 *v36; // [rsp+18h] [rbp-38h]
  __int64 v37; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD **)(a1 + 112);
  v3 = 23LL * *(unsigned int *)(a1 + 120);
  v4 = &v2[v3];
  v5 = 0xD37A6F4DE9BD37A7LL * ((v3 * 8) >> 3);
  if ( v5 >> 2 )
  {
    v34 = &v2[92 * (v5 >> 2)];
    while ( 1 )
    {
      v10 = v2[1];
      if ( v10 )
      {
        v11 = *(_QWORD *)(v10 - 32);
        if ( v11 == a2 )
          break;
        v24 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 112LL);
        v28 = sub_DD8400(v24, a2);
        if ( v28 == sub_DD8400(v24, v11) )
          break;
      }
      v6 = v2[24];
      v7 = v2 + 23;
      if ( v6 )
      {
        if ( *(_QWORD *)(v6 - 32) == a2 )
          return v4 != v7;
        v21 = *(_QWORD *)(v6 - 32);
        v25 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 112LL);
        v29 = sub_DD8400(v25, a2);
        if ( v29 == sub_DD8400(v25, v21) )
          return v4 != v7;
      }
      v8 = v2[47];
      v7 = v2 + 46;
      if ( v8 )
      {
        if ( *(_QWORD *)(v8 - 32) == a2 )
          return v4 != v7;
        v22 = *(_QWORD *)(v8 - 32);
        v26 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 112LL);
        v30 = sub_DD8400(v26, a2);
        if ( v30 == sub_DD8400(v26, v22) )
          return v4 != v7;
      }
      v9 = v2[70];
      v7 = v2 + 69;
      if ( v9 )
      {
        if ( *(_QWORD *)(v9 - 32) == a2 )
          return v4 != v7;
        v23 = *(_QWORD *)(v9 - 32);
        v27 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 112LL);
        v31 = sub_DD8400(v27, a2);
        if ( v31 == sub_DD8400(v27, v23) )
          return v4 != v7;
      }
      v2 += 92;
      if ( v2 == v34 )
      {
        v5 = 0xD37A6F4DE9BD37A7LL * (v4 - v2);
        goto LABEL_20;
      }
    }
    return v4 != v2;
  }
LABEL_20:
  if ( v5 != 2 )
  {
    if ( v5 != 3 )
    {
      if ( v5 != 1 )
        return 0;
      goto LABEL_32;
    }
    v13 = v2[1];
    if ( v13 )
    {
      v14 = *(_QWORD *)(v13 - 32);
      if ( v14 == a2 )
        return v4 != v2;
      v32 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 112LL);
      v35 = sub_DD8400(v32, a2);
      if ( v35 == sub_DD8400(v32, v14) )
        return v4 != v2;
    }
    v2 += 23;
  }
  v15 = v2[1];
  if ( v15 )
  {
    v16 = *(_QWORD *)(v15 - 32);
    if ( v16 == a2 )
      return v4 != v2;
    v33 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 112LL);
    v36 = sub_DD8400(v33, a2);
    if ( v36 == sub_DD8400(v33, v16) )
      return v4 != v2;
  }
  v2 += 23;
LABEL_32:
  v17 = v2[1];
  if ( !v17 )
    return 0;
  if ( *(_QWORD *)(v17 - 32) == a2 )
    return v4 != v2;
  v37 = *(_QWORD *)(v17 - 32);
  v18 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 112LL);
  v19 = sub_DD8400(v18, a2);
  v20 = sub_DD8400(v18, v37);
  result = 0;
  if ( v19 == v20 )
    return v4 != v2;
  return result;
}
