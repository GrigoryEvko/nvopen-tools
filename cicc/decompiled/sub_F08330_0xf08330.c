// Function: sub_F08330
// Address: 0xf08330
//
void __fastcall sub_F08330(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 *v6; // r9
  __int64 *v7; // r15
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 *v10; // rax
  char *v11; // r10
  __int64 *v12; // r9
  char *v13; // r11
  __int64 v14; // r14
  __int64 *v15; // rdx
  __int64 *v16; // r11
  __int64 *v17; // rax
  __int64 v18; // r13
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 *v23; // [rsp+10h] [rbp-50h]
  char *v24; // [rsp+18h] [rbp-48h]
  __int64 *v25; // [rsp+20h] [rbp-40h]
  __int64 *v26; // [rsp+20h] [rbp-40h]
  __int64 *v27; // [rsp+20h] [rbp-40h]
  __int64 *srca; // [rsp+28h] [rbp-38h]
  char *srcb; // [rsp+28h] [rbp-38h]
  __int64 *srcc; // [rsp+28h] [rbp-38h]
  __int64 *src; // [rsp+28h] [rbp-38h]

  if ( a4 )
  {
    v5 = a5;
    if ( a5 )
    {
      v6 = a1;
      v7 = a2;
      v8 = a4;
      if ( a5 + a4 == 2 )
      {
        v16 = a2;
        v15 = a1;
LABEL_12:
        v18 = *v16;
        v27 = v16;
        src = v15;
        v19 = sub_B140A0(*v15);
        v20 = sub_B140A0(v18);
        if ( sub_B445A0(v19, v20) )
        {
          v21 = *src;
          *src = *v27;
          *v27 = v21;
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v25 = v6;
        v9 = v8 / 2;
        srca = &v6[v8 / 2];
        v10 = sub_F07240(v7, a3, srca);
        v11 = (char *)srca;
        v12 = v25;
        v13 = (char *)v10;
        v14 = v10 - v7;
        while ( 1 )
        {
          v26 = v12;
          v23 = (__int64 *)v13;
          v5 -= v14;
          srcb = v11;
          v24 = sub_F07A80(v11, (char *)v7, v13);
          sub_F08330(v26, srcb, v24, v9, v14);
          v8 -= v9;
          if ( !v8 )
            break;
          v15 = (__int64 *)v24;
          v16 = v23;
          if ( !v5 )
            break;
          if ( v5 + v8 == 2 )
            goto LABEL_12;
          v7 = v23;
          v6 = (__int64 *)v24;
          if ( v8 > v5 )
            goto LABEL_5;
LABEL_10:
          srcc = v6;
          v14 = v5 / 2;
          v17 = sub_F072E0(v6, (__int64)v7, &v7[v5 / 2]);
          v12 = srcc;
          v13 = (char *)&v7[v5 / 2];
          v11 = (char *)v17;
          v9 = v17 - srcc;
        }
      }
    }
  }
}
