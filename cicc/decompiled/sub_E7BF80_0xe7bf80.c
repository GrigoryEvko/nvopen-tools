// Function: sub_E7BF80
// Address: 0xe7bf80
//
void __fastcall sub_E7BF80(__int64 *a1, __int64 *a2, __int64 a3, signed __int64 a4, signed __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 v10; // rbx
  char *v11; // r13
  __int64 v12; // rax
  __int64 *v13; // r10
  __int64 v14; // r11
  char *v15; // r14
  __int64 *v16; // r9
  __int64 v17; // r15
  __int64 v18; // rax
  __int64 v19; // [rsp-58h] [rbp-58h]
  __int64 v20; // [rsp-58h] [rbp-58h]
  __int64 *v21; // [rsp-50h] [rbp-50h]
  __int64 *v22; // [rsp-48h] [rbp-48h]
  __int64 v23; // [rsp-48h] [rbp-48h]
  __int64 v24; // [rsp-40h] [rbp-40h]
  __int64 v25; // [rsp-30h] [rbp-30h]
  __int64 v26; // [rsp-20h] [rbp-20h]
  __int64 v27; // [rsp-18h] [rbp-18h]
  __int64 v28; // [rsp-10h] [rbp-10h]

  while ( a4 )
  {
    v28 = v8;
    v27 = v7;
    v26 = v6;
    v9 = a5;
    v25 = v5;
    if ( !a5 )
      break;
    v10 = a4;
    if ( a4 + a5 == 2 )
    {
      if ( (unsigned __int8)sub_E72550((__int64)a2, (__int64)a1) )
        sub_E7BDB0(a1, a2);
      return;
    }
    if ( a4 > a5 )
    {
      v23 = a3;
      v17 = a4 / 2;
      v15 = (char *)&a1[4 * (a4 / 2) + 4 * ((a4 + ((unsigned __int64)a4 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
      v18 = sub_E72840((__int64)a2, a3, (__int64)v15);
      v16 = a2;
      v14 = v23;
      v11 = (char *)v18;
      v13 = a1;
      v24 = 0xAAAAAAAAAAAAAAABLL * ((v18 - (__int64)a2) >> 5);
    }
    else
    {
      v19 = a3;
      v24 = a5 / 2;
      v11 = (char *)&a2[4 * (a5 / 2) + 4 * ((a5 + ((unsigned __int64)a5 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
      v12 = sub_E727B0((__int64)a1, (__int64)a2, (__int64)v11);
      v13 = a1;
      v14 = v19;
      v15 = (char *)v12;
      v16 = a2;
      v17 = 0xAAAAAAAAAAAAAAABLL * ((v12 - (__int64)a1) >> 5);
    }
    v20 = v14;
    v21 = v13;
    v22 = sub_E73DF0(v15, v16, v11);
    sub_E7BF80(v21, v15, v22, v17, v24);
    a4 = v10 - v17;
    a2 = (__int64 *)v11;
    v5 = v25;
    a5 = v9 - v24;
    a3 = v20;
    a1 = v22;
    v6 = v26;
    v7 = v27;
    v8 = v28;
  }
}
