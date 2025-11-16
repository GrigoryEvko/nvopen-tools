// Function: sub_F7B350
// Address: 0xf7b350
//
void __fastcall sub_F7B350(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // rbx
  __int64 v4; // r14
  __int64 *v5; // r15
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 *v10; // rcx
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 *v15; // r15
  bool v16; // al
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // r13
  __int64 v23; // [rsp+20h] [rbp-40h]
  __int64 v24; // [rsp+20h] [rbp-40h]
  bool v25; // [rsp+2Fh] [rbp-31h]
  bool v26; // [rsp+2Fh] [rbp-31h]

  if ( a1 == a2 )
    return;
  v3 = a1 + 2;
  if ( a2 == a1 + 2 )
    return;
  do
  {
    v4 = v3[1];
    v5 = v3;
    v6 = *v3;
    v7 = a1[1];
    v23 = *a1;
    v25 = *(_BYTE *)(sub_D95540(v4) + 8) == 14;
    if ( v25 == (*(_BYTE *)(sub_D95540(v7) + 8) == 14) )
    {
      if ( v6 == v23 )
      {
        if ( sub_D969D0(v4) )
        {
          sub_D969D0(v7);
        }
        else if ( sub_D969D0(v7) )
        {
LABEL_5:
          v8 = *v3;
          v9 = v3[1];
          v10 = v3 + 2;
          v11 = (char *)v3 - (char *)a1;
          v12 = v11 >> 4;
          if ( v11 > 0 )
          {
            do
            {
              v13 = *(v5 - 2);
              v5 -= 2;
              v5[2] = v13;
              v5[3] = v5[1];
              --v12;
            }
            while ( v12 );
          }
          *a1 = v8;
          a1[1] = v9;
          goto LABEL_8;
        }
      }
      else if ( v6 != sub_F79730(v6, v23, a3) )
      {
        goto LABEL_5;
      }
    }
    else if ( *(_BYTE *)(sub_D95540(v4) + 8) == 14 )
    {
      goto LABEL_5;
    }
    v14 = v3[1];
    v15 = v3;
    v24 = *v3;
    while ( 1 )
    {
      v18 = *(v15 - 1);
      v19 = *(v15 - 2);
      v26 = *(_BYTE *)(sub_D95540(v14) + 8) == 14;
      if ( v26 != (*(_BYTE *)(sub_D95540(v18) + 8) == 14) )
      {
        v16 = *(_BYTE *)(sub_D95540(v14) + 8) == 14;
        goto LABEL_14;
      }
      if ( v24 != v19 )
      {
        v16 = v24 != sub_F79730(v24, v19, a3);
LABEL_14:
        if ( !v16 )
          goto LABEL_21;
        goto LABEL_15;
      }
      if ( sub_D969D0(v14) )
        break;
      if ( !sub_D969D0(v18) )
        goto LABEL_21;
LABEL_15:
      v17 = *(v15 - 2);
      v15 -= 2;
      v15[2] = v17;
      v15[3] = v15[1];
    }
    sub_D969D0(v18);
LABEL_21:
    v15[1] = v14;
    v10 = v3 + 2;
    *v15 = v24;
LABEL_8:
    v3 = v10;
  }
  while ( a2 != v10 );
}
