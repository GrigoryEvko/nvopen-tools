// Function: sub_10A7DB0
// Address: 0x10a7db0
//
bool __fastcall sub_10A7DB0(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // rax
  char v8; // al
  __int64 v9; // rsi
  unsigned __int8 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rcx
  char v15; // al
  char v16; // al
  __int64 v17; // rsi
  __int64 v18; // rax
  char v19; // al
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rcx
  char v23; // al
  __int64 v24; // rax
  char v25; // al
  __int64 v26; // rax
  char v27; // al
  unsigned __int8 *v28; // [rsp-20h] [rbp-20h]
  unsigned __int8 *v29; // [rsp-20h] [rbp-20h]
  unsigned __int8 *v30; // [rsp-20h] [rbp-20h]
  unsigned __int8 *v31; // [rsp-20h] [rbp-20h]

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  v5 = *(_QWORD *)(v4 + 16);
  if ( !v5 || *(_QWORD *)(v5 + 8) || *(_BYTE *)v4 != 59 )
    goto LABEL_4;
  v30 = a3;
  v16 = sub_995B10(a1, *(_QWORD *)(v4 - 64));
  v17 = *(_QWORD *)(v4 - 32);
  a3 = v30;
  if ( v16 )
  {
    v18 = *(_QWORD *)(v17 + 16);
    if ( v18 )
    {
      if ( !*(_QWORD *)(v18 + 8) && *(_BYTE *)v17 == 46 )
      {
        v24 = *(_QWORD *)(v17 - 64);
        if ( v24 )
        {
          *a1[1] = v24;
          v25 = sub_991580((__int64)(a1 + 2), *(_QWORD *)(v17 - 32));
          a3 = v30;
          if ( v25 )
            goto LABEL_29;
          v17 = *(_QWORD *)(v4 - 32);
        }
      }
    }
  }
  v31 = a3;
  v19 = sub_995B10(a1, v17);
  a3 = v31;
  if ( !v19
    || (v20 = *(_QWORD *)(v4 - 64), (v21 = *(_QWORD *)(v20 + 16)) == 0)
    || *(_QWORD *)(v21 + 8)
    || *(_BYTE *)v20 != 46
    || (v22 = *(_QWORD *)(v20 - 64)) == 0
    || (*a1[1] = v22, v23 = sub_991580((__int64)(a1 + 2), *(_QWORD *)(v20 - 32)), a3 = v31, !v23) )
  {
LABEL_4:
    v6 = *((_QWORD *)a3 - 4);
LABEL_5:
    v7 = *(_QWORD *)(v6 + 16);
    if ( v7 && !*(_QWORD *)(v7 + 8) && *(_BYTE *)v6 == 59 )
    {
      v28 = a3;
      v8 = sub_995B10(a1, *(_QWORD *)(v6 - 64));
      v9 = *(_QWORD *)(v6 - 32);
      v10 = v28;
      if ( v8 )
      {
        v11 = *(_QWORD *)(v9 + 16);
        if ( v11 )
        {
          if ( !*(_QWORD *)(v11 + 8) && *(_BYTE *)v9 == 46 )
          {
            v26 = *(_QWORD *)(v9 - 64);
            if ( v26 )
            {
              *a1[1] = v26;
              v27 = sub_991580((__int64)(a1 + 2), *(_QWORD *)(v9 - 32));
              v10 = v28;
              if ( v27 )
                return *a1[4] == *((_QWORD *)v10 - 8);
              v9 = *(_QWORD *)(v6 - 32);
            }
          }
        }
      }
      v29 = v10;
      if ( (unsigned __int8)sub_995B10(a1, v9) )
      {
        v12 = *(_QWORD *)(v6 - 64);
        v13 = *(_QWORD *)(v12 + 16);
        if ( v13 )
        {
          if ( !*(_QWORD *)(v13 + 8) && *(_BYTE *)v12 == 46 )
          {
            v14 = *(_QWORD *)(v12 - 64);
            if ( v14 )
            {
              *a1[1] = v14;
              v15 = sub_991580((__int64)(a1 + 2), *(_QWORD *)(v12 - 32));
              v10 = v29;
              if ( v15 )
                return *a1[4] == *((_QWORD *)v10 - 8);
            }
          }
        }
      }
    }
    return 0;
  }
LABEL_29:
  v6 = *((_QWORD *)a3 - 4);
  result = 1;
  if ( *a1[4] != v6 )
    goto LABEL_5;
  return result;
}
