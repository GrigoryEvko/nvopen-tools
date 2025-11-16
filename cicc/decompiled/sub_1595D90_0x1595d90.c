// Function: sub_1595D90
// Address: 0x1595d90
//
char __fastcall sub_1595D90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  __int64 v5; // r12
  char v6; // r13
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r13
  char v12; // al
  __int64 v13; // r13
  __int64 v14; // rdx
  char v15; // al
  __int64 v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 *v20; // rax
  char v21[8]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v22; // [rsp+8h] [rbp-28h] BYREF
  __int64 v23; // [rsp+10h] [rbp-20h]
  char v24; // [rsp+1Ah] [rbp-16h]

  v4 = *(_BYTE *)(a1 + 16);
  if ( v4 != 14 )
  {
    if ( v4 != 12 )
    {
LABEL_13:
      if ( v4 == 8 )
      {
        v9 = sub_1594B20(a1);
        v11 = v9;
        if ( v9 )
        {
          if ( *(_BYTE *)(v9 + 16) == 14 )
          {
            if ( *(_QWORD *)(v9 + 32) == sub_16982C0(a1, a2, v10, a4) )
            {
              v16 = *(_QWORD *)(v11 + 40);
              if ( (*(_BYTE *)(v16 + 26) & 7) == 3 )
              {
                v13 = v16 + 8;
LABEL_18:
                if ( (*(_BYTE *)(v13 + 18) & 8) != 0 )
                  return 1;
              }
            }
            else
            {
              v12 = *(_BYTE *)(v11 + 50);
              v13 = v11 + 32;
              if ( (v12 & 7) == 3 )
                goto LABEL_18;
            }
          }
        }
      }
      v14 = *(_QWORD *)a1;
      v15 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
      if ( v15 == 16 )
        v15 = *(_BYTE *)(**(_QWORD **)(v14 + 16) + 8LL);
      if ( (unsigned __int8)(v15 - 1) <= 5u )
        return 0;
      return sub_1593BB0(a1, a2, v14, a4);
    }
    if ( (unsigned __int8)(*(_BYTE *)(sub_1595890(a1) + 8) - 1) > 5u || (v6 = sub_1595CF0(a1)) == 0 )
    {
LABEL_12:
      v4 = *(_BYTE *)(a1 + 16);
      goto LABEL_13;
    }
    a2 = a1;
    sub_1595B70((__int64)v21, a1, 0);
    v19 = sub_16982C0(v21, a1, v17, v18);
    if ( v22 == v19 )
    {
      if ( (*(_BYTE *)(v23 + 26) & 7) == 3 )
      {
        v20 = (__int64 *)(v23 + 8);
LABEL_28:
        if ( (*((_BYTE *)v20 + 18) & 8) != 0 )
        {
          sub_127D120(&v22);
          return v6;
        }
      }
    }
    else if ( (v24 & 7) == 3 )
    {
      v20 = &v22;
      goto LABEL_28;
    }
    sub_127D120(&v22);
    goto LABEL_12;
  }
  if ( *(_QWORD *)(a1 + 32) == sub_16982C0(a1, a2, a3, a4) )
  {
    v8 = *(_QWORD *)(a1 + 40);
    if ( (*(_BYTE *)(v8 + 26) & 7) != 3 )
      return 0;
    v5 = v8 + 8;
  }
  else
  {
    v5 = a1 + 32;
    if ( (*(_BYTE *)(a1 + 50) & 7) != 3 )
      return 0;
  }
  return (*(_BYTE *)(v5 + 18) & 8) != 0;
}
