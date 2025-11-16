// Function: sub_1004B70
// Address: 0x1004b70
//
__int64 __fastcall sub_1004B70(_BYTE *a1, _BYTE *a2)
{
  char v3; // al
  __int64 result; // rax
  _BYTE *v5; // r13
  _BYTE *v6; // r13
  char v7; // al
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rax
  _BYTE *v11; // r15
  _BYTE *v12; // r15
  char v13; // al
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rax
  char v17; // al
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rax
  char v22; // al
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // [rsp+18h] [rbp-68h] BYREF
  __int64 v28; // [rsp+20h] [rbp-60h] BYREF
  __int64 v29; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v30; // [rsp+30h] [rbp-50h] BYREF
  __int64 *v31; // [rsp+38h] [rbp-48h]
  __int64 *v32; // [rsp+40h] [rbp-40h]
  __int64 *v33; // [rsp+48h] [rbp-38h]

  v3 = *a1;
  v30 = 0;
  v31 = &v27;
  v32 = &v28;
  if ( v3 == 57 )
  {
    v11 = (_BYTE *)*((_QWORD *)a1 - 8);
    if ( *v11 != 59 )
    {
LABEL_14:
      v12 = (_BYTE *)*((_QWORD *)a1 - 4);
      goto LABEL_15;
    }
    v17 = sub_995B10(&v30, *((_QWORD *)v11 - 8));
    v18 = *((_QWORD *)v11 - 4);
    if ( v17 && v18 )
    {
      *v31 = v18;
    }
    else
    {
      if ( !(unsigned __int8)sub_995B10(&v30, v18) )
        goto LABEL_14;
      v21 = *((_QWORD *)v11 - 8);
      if ( !v21 )
        goto LABEL_14;
      *v31 = v21;
    }
    v12 = (_BYTE *)*((_QWORD *)a1 - 4);
    if ( v12 )
    {
      *v32 = (__int64)v12;
      goto LABEL_28;
    }
LABEL_15:
    if ( *v12 != 59 )
      goto LABEL_16;
    v13 = sub_995B10(&v30, *((_QWORD *)v12 - 8));
    v14 = *((_QWORD *)v12 - 4);
    if ( v13 && v14 )
    {
      *v31 = v14;
    }
    else
    {
      if ( !(unsigned __int8)sub_995B10(&v30, v14) )
        goto LABEL_16;
      v15 = *((_QWORD *)v12 - 8);
      if ( !v15 )
        goto LABEL_16;
      *v31 = v15;
    }
    v16 = *((_QWORD *)a1 - 8);
    if ( !v16 )
    {
LABEL_16:
      v3 = *a1;
      goto LABEL_2;
    }
    *v32 = v16;
LABEL_28:
    if ( *a2 == 58 )
    {
      result = v27;
      v19 = *((_QWORD *)a2 - 8);
      v20 = *((_QWORD *)a2 - 4);
      if ( v27 == v19 && v28 == v20 )
        return result;
      if ( v28 == v19 && v27 == v20 )
        return result;
    }
    goto LABEL_16;
  }
LABEL_2:
  v30 = 0;
  v31 = &v27;
  v32 = &v29;
  v33 = &v28;
  if ( v3 != 58 )
    return 0;
  v5 = (_BYTE *)*((_QWORD *)a1 - 8);
  if ( *v5 != 59 )
    goto LABEL_5;
  v22 = sub_995B10(&v30, *((_QWORD *)v5 - 8));
  v23 = *((_QWORD *)v5 - 4);
  if ( v22 && v23 )
  {
    *v31 = v23;
  }
  else
  {
    if ( !(unsigned __int8)sub_995B10(&v30, v23) || (v26 = *((_QWORD *)v5 - 8)) == 0 )
    {
LABEL_5:
      v6 = (_BYTE *)*((_QWORD *)a1 - 4);
      goto LABEL_6;
    }
    *v31 = v26;
  }
  *v32 = (__int64)v5;
  v6 = (_BYTE *)*((_QWORD *)a1 - 4);
  if ( !v6 )
  {
LABEL_6:
    if ( *v6 != 59 )
      return 0;
    v7 = sub_995B10(&v30, *((_QWORD *)v6 - 8));
    v8 = *((_QWORD *)v6 - 4);
    if ( v7 && v8 )
    {
      *v31 = v8;
    }
    else
    {
      if ( !(unsigned __int8)sub_995B10(&v30, v8) )
        return 0;
      v9 = *((_QWORD *)v6 - 8);
      if ( !v9 )
        return 0;
      *v31 = v9;
    }
    *v32 = (__int64)v6;
    v10 = *((_QWORD *)a1 - 8);
    if ( !v10 )
      return 0;
    *v33 = v10;
    goto LABEL_41;
  }
  *v33 = (__int64)v6;
LABEL_41:
  if ( *a2 != 57 )
    return 0;
  v24 = *((_QWORD *)a2 - 8);
  v25 = *((_QWORD *)a2 - 4);
  if ( (v27 != v24 || v28 != v25) && (v27 != v25 || v28 != v24) )
    return 0;
  return v29;
}
