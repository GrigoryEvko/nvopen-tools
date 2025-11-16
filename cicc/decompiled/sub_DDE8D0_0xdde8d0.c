// Function: sub_DDE8D0
// Address: 0xdde8d0
//
char __fastcall sub_DDE8D0(__int64 *a1, unsigned __int64 a2, _BYTE *a3, _BYTE *a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r12
  __int64 v10; // r10
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  __int64 v13; // rax
  __int64 v15; // r10
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  __int64 v18; // rax
  _BYTE *v19; // r8
  _BYTE *v20; // r9
  __int64 v21; // [rsp+8h] [rbp-48h]
  __int64 v22; // [rsp+8h] [rbp-48h]
  __int64 v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+8h] [rbp-48h]
  __int64 v25; // [rsp+10h] [rbp-40h]
  __int64 v26; // [rsp+10h] [rbp-40h]
  __int64 v27; // [rsp+10h] [rbp-40h]
  __int64 v28; // [rsp+10h] [rbp-40h]
  __int64 v29; // [rsp+18h] [rbp-38h]
  __int64 v30; // [rsp+18h] [rbp-38h]
  __int64 v31; // [rsp+18h] [rbp-38h]
  __int64 v32; // [rsp+18h] [rbp-38h]

  if ( !a7 )
    return 0;
  v7 = *(_QWORD *)(a7 + 40);
  if ( *(_WORD *)(a5 + 24) != 8 )
  {
    if ( *(_WORD *)(a6 + 24) == 8 )
    {
      v15 = *(_QWORD *)(a6 + 48);
      if ( *(_BYTE *)(v15 + 84) )
      {
        v16 = *(_QWORD **)(v15 + 64);
        v17 = &v16[*(unsigned int *)(v15 + 76)];
        if ( v16 == v17 )
          return 0;
        while ( v7 != *v16 )
        {
          if ( v17 == ++v16 )
            return 0;
        }
      }
      else
      {
        v23 = a6;
        v27 = a5;
        v31 = *(_QWORD *)(a6 + 48);
        if ( !sub_C8CA60(v15 + 56, *(_QWORD *)(a7 + 40)) )
          return 0;
        v15 = v31;
        a5 = v27;
        a6 = v23;
      }
      v22 = a6;
      v26 = a5;
      v30 = a1[5];
      v18 = sub_D47930(v15);
      if ( !(unsigned __int8)sub_B19720(v30, v7, v18) || !sub_DAEB70((__int64)a1, v26, *(_QWORD *)(v22 + 48)) )
        return 0;
      v19 = (_BYTE *)v26;
      v20 = **(_BYTE ***)(v22 + 32);
      return sub_DDB0E0(a1, a2, a3, a4, v19, v20, 0);
    }
    return 0;
  }
  v10 = *(_QWORD *)(a5 + 48);
  if ( *(_BYTE *)(v10 + 84) )
  {
    v11 = *(_QWORD **)(v10 + 64);
    v12 = &v11[*(unsigned int *)(v10 + 76)];
    if ( v11 == v12 )
      return 0;
    while ( v7 != *v11 )
    {
      if ( v12 == ++v11 )
        return 0;
    }
    goto LABEL_8;
  }
  v24 = a6;
  v28 = a5;
  v32 = *(_QWORD *)(a5 + 48);
  if ( !sub_C8CA60(v10 + 56, *(_QWORD *)(a7 + 40)) )
    return 0;
  v10 = v32;
  a5 = v28;
  a6 = v24;
LABEL_8:
  v21 = a6;
  v25 = a5;
  v29 = a1[5];
  v13 = sub_D47930(v10);
  if ( !(unsigned __int8)sub_B19720(v29, v7, v13) || !sub_DAEB70((__int64)a1, v21, *(_QWORD *)(v25 + 48)) )
    return 0;
  v20 = (_BYTE *)v21;
  v19 = **(_BYTE ***)(v25 + 32);
  return sub_DDB0E0(a1, a2, a3, a4, v19, v20, 0);
}
