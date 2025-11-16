// Function: sub_10D1B20
// Address: 0x10d1b20
//
__int64 __fastcall sub_10D1B20(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  _BYTE *v9; // rdi
  _BYTE *v10; // rdi
  __int64 v11; // rax
  _BYTE *v12; // rdi
  char v13; // al
  _BYTE *v14; // rdi
  char v15; // al
  unsigned __int8 *v16; // [rsp-20h] [rbp-20h]
  unsigned __int8 *v17; // [rsp-20h] [rbp-20h]
  unsigned __int8 *v18; // [rsp-20h] [rbp-20h]
  unsigned __int8 *v19; // [rsp-20h] [rbp-20h]

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = *((_QWORD *)a3 - 8);
  v6 = *(_QWORD *)(v5 + 16);
  if ( !v6 )
    goto LABEL_4;
  if ( *(_QWORD *)(v6 + 8) )
    goto LABEL_4;
  if ( *(_BYTE *)v5 != 59 )
    goto LABEL_4;
  v12 = *(_BYTE **)(v5 - 64);
  if ( !v12 )
    goto LABEL_4;
  **a1 = v12;
  if ( *v12 <= 0x15u )
  {
    if ( *v12 == 5 )
      goto LABEL_4;
    v18 = a3;
    v13 = sub_AD6CA0((__int64)v12);
    a3 = v18;
    if ( v13 )
      goto LABEL_4;
  }
  v14 = *(_BYTE **)(v5 - 32);
  if ( *v14 > 0x15u
    || (*a1[2] = v14, *v14 <= 0x15u) && (*v14 == 5 || (v19 = a3, v15 = sub_AD6CA0((__int64)v14), a3 = v19, v15)) )
  {
LABEL_4:
    v7 = *((_QWORD *)a3 - 4);
  }
  else
  {
    v7 = *((_QWORD *)a3 - 4);
    if ( v7 )
    {
      *a1[4] = v7;
      return 1;
    }
  }
  v8 = *(_QWORD *)(v7 + 16);
  if ( v8 )
  {
    if ( !*(_QWORD *)(v8 + 8) && *(_BYTE *)v7 == 59 )
    {
      v9 = *(_BYTE **)(v7 - 64);
      if ( v9 )
      {
        **a1 = v9;
        if ( *v9 <= 0x15u )
        {
          v16 = a3;
          if ( *v9 == 5 || (unsigned __int8)sub_AD6CA0((__int64)v9) )
            return 0;
          a3 = v16;
        }
        v10 = *(_BYTE **)(v7 - 32);
        if ( *v10 <= 0x15u )
        {
          *a1[2] = v10;
          if ( *v10 > 0x15u )
            goto LABEL_20;
          v17 = a3;
          if ( *v10 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v10) )
          {
            a3 = v17;
LABEL_20:
            v11 = *((_QWORD *)a3 - 8);
            if ( v11 )
            {
              *a1[4] = v11;
              return 1;
            }
          }
        }
      }
    }
  }
  return 0;
}
