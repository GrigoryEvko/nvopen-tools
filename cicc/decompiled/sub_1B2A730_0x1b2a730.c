// Function: sub_1B2A730
// Address: 0x1b2a730
//
__int64 __fastcall sub_1B2A730(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r12
  __int64 *v6; // r13
  bool v7; // zf
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 *v12; // r15
  __int64 *v13; // r14
  __int64 *v14; // r13
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // r14
  __int64 v20; // rcx
  __int64 *v21; // r15
  __int64 v23; // [rsp+10h] [rbp-40h]
  __int64 *v24; // [rsp+18h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v24 = a2;
  v23 = a3;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v14 = a2;
    goto LABEL_30;
  }
  v5 = a4 + 40;
  while ( 2 )
  {
    --v23;
    v6 = &a1[result >> 4];
    v7 = sub_1B29A30(v5, a1[1], *v6) == 0;
    v8 = *(v24 - 1);
    if ( v7 )
    {
      if ( !sub_1B29A30(v5, a1[1], v8) )
      {
        v21 = v24;
        v7 = sub_1B29A30(v5, *v6, *(v24 - 1)) == 0;
        v9 = *a1;
        if ( v7 )
          goto LABEL_7;
        goto LABEL_36;
      }
LABEL_28:
      v11 = *a1;
      v10 = a1[1];
      a1[1] = *a1;
      *a1 = v10;
      goto LABEL_8;
    }
    if ( sub_1B29A30(v5, *v6, v8) )
    {
      v9 = *a1;
LABEL_7:
      *a1 = *v6;
      *v6 = v9;
      v10 = *a1;
      v11 = a1[1];
      goto LABEL_8;
    }
    v21 = v24;
    if ( !sub_1B29A30(v5, a1[1], *(v24 - 1)) )
      goto LABEL_28;
    v9 = *a1;
LABEL_36:
    *a1 = *(v21 - 1);
    *(v21 - 1) = v9;
    v10 = *a1;
    v11 = a1[1];
LABEL_8:
    v12 = a1 + 1;
    v13 = v24;
    while ( 1 )
    {
      v14 = v12;
      if ( !sub_1B29A30(v5, v11, v10) )
        break;
LABEL_21:
      v11 = v12[1];
      v10 = *a1;
      ++v12;
    }
    v15 = *a1;
    for ( --v13; ; --v13 )
    {
      v16 = *v13;
      if ( !v15 || *(_BYTE *)(v15 + 16) != 17 )
        break;
      if ( v16 && *(_BYTE *)(v16 + 16) == 17 && *(_DWORD *)(v15 + 32) >= *(_DWORD *)(v16 + 32) )
        goto LABEL_19;
LABEL_14:
      ;
    }
    if ( (!v16 || *(_BYTE *)(v16 + 16) != 17) && sub_1B298A0(v5, v15, v16) )
    {
      v15 = *a1;
      goto LABEL_14;
    }
LABEL_19:
    if ( v12 < v13 )
    {
      v17 = *v12;
      *v12 = *v13;
      *v13 = v17;
      goto LABEL_21;
    }
    sub_1B2A730(v12, v24, v23, a4);
    result = (char *)v12 - (char *)a1;
    if ( (char *)v12 - (char *)a1 > 128 )
    {
      if ( v23 )
      {
        v24 = v12;
        continue;
      }
LABEL_30:
      v18 = result >> 3;
      v19 = ((result >> 3) - 2) >> 1;
      sub_1B29A80((__int64)a1, v19, result >> 3, a1[v19], a4);
      do
      {
        --v19;
        sub_1B29A80((__int64)a1, v19, v18, a1[v19], a4);
      }
      while ( v19 );
      do
      {
        v20 = *--v14;
        *v14 = *a1;
        result = sub_1B29A80((__int64)a1, 0, v14 - a1, v20, a4);
      }
      while ( (char *)v14 - (char *)a1 > 8 );
    }
    return result;
  }
}
