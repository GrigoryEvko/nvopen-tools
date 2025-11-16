// Function: sub_3512ED0
// Address: 0x3512ed0
//
void __fastcall sub_3512ED0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 *v7; // r14
  __int64 *v8; // r13
  __int64 v9; // r12
  __int64 *v11; // rax
  char *v12; // r10
  char *v13; // r11
  char *v14; // rax
  __int64 v15; // r13
  __int64 *v16; // rdx
  __int64 *v17; // r11
  __int64 *v18; // rax
  __int64 v19; // r12
  unsigned __int64 v20; // rbx
  __int64 v21; // rax
  __int64 *v23; // [rsp+10h] [rbp-50h]
  char *src; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+20h] [rbp-40h]
  __int64 *v26; // [rsp+20h] [rbp-40h]
  __int64 *v27; // [rsp+20h] [rbp-40h]
  __int64 v28; // [rsp+28h] [rbp-38h]
  __int64 *v29; // [rsp+28h] [rbp-38h]

  if ( a5 )
  {
    v6 = a4;
    if ( a4 )
    {
      v7 = a1;
      v8 = a2;
      v9 = a5;
      if ( a5 + a4 == 2 )
      {
        v17 = a2;
        v16 = a1;
LABEL_12:
        v27 = v17;
        v19 = *v17;
        v29 = v16;
        v20 = sub_2F06CB0(*(_QWORD *)(a6 + 536), *v16);
        if ( v20 < sub_2F06CB0(*(_QWORD *)(a6 + 536), v19) )
        {
          v21 = *v29;
          *v29 = *v27;
          *v27 = v21;
        }
      }
      else
      {
        if ( a5 >= a4 )
          goto LABEL_10;
LABEL_5:
        v25 = v6 / 2;
        v11 = sub_35116C0(v8, a3, &v7[v6 / 2], a6);
        v12 = (char *)&v7[v6 / 2];
        v13 = (char *)v11;
        v28 = v11 - v8;
        while ( 1 )
        {
          v23 = (__int64 *)v13;
          src = v12;
          v14 = sub_3512D10(v12, (char *)v8, v13);
          v15 = v25;
          v26 = (__int64 *)v14;
          sub_3512ED0(v7, src, v14, v15, v28, a6);
          v9 -= v28;
          v6 -= v15;
          if ( !v6 )
            break;
          v16 = v26;
          v17 = v23;
          if ( !v9 )
            break;
          if ( v9 + v6 == 2 )
            goto LABEL_12;
          v8 = v23;
          v7 = v26;
          if ( v9 < v6 )
            goto LABEL_5;
LABEL_10:
          v28 = v9 / 2;
          v18 = sub_3511610(v7, (__int64)v8, &v8[v9 / 2], a6);
          v13 = (char *)&v8[v9 / 2];
          v12 = (char *)v18;
          v25 = v18 - v7;
        }
      }
    }
  }
}
