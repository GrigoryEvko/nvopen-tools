// Function: sub_2FA2490
// Address: 0x2fa2490
//
_QWORD *__fastcall sub_2FA2490(__int64 *a1, _QWORD *a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  char *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rax
  char **v10; // r12
  __int64 v11; // r13
  char *v12; // rsi
  char **v13; // rbx
  __int64 v14; // r9
  __int64 v15; // rdx
  char *v17; // [rsp+8h] [rbp-78h]
  char *v18; // [rsp+10h] [rbp-70h] BYREF
  __int64 v19; // [rsp+18h] [rbp-68h]
  __int64 v20; // [rsp+20h] [rbp-60h]
  __int64 v21; // [rsp+28h] [rbp-58h]
  char *v22; // [rsp+30h] [rbp-50h] BYREF
  __int64 v23; // [rsp+38h] [rbp-48h]
  __int64 v24; // [rsp+40h] [rbp-40h]
  __int64 v25; // [rsp+48h] [rbp-38h]

  v4 = ((__int64)(a2[6] - a2[7]) >> 3) + ((((__int64)(a2[9] - a2[5]) >> 3) - 1) << 6);
  v5 = a2[4] - a2[2];
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  a1[5] = 0;
  a1[6] = 0;
  a1[7] = 0;
  a1[8] = 0;
  a1[9] = 0;
  sub_2785050(a1, v4 + (v5 >> 3));
  v6 = (char *)a1[2];
  v7 = a1[3];
  v8 = a1[4];
  v9 = a1[5];
  v10 = (char **)a2[9];
  v11 = a2[6];
  v12 = (char *)a2[2];
  v17 = (char *)a2[7];
  v13 = (char **)a2[5];
  v14 = a2[4];
  if ( v13 == v10 )
  {
    v22 = v6;
    v23 = v7;
    v24 = v8;
    v25 = v9;
    return sub_2FA2360(&v18, v12, v11, &v22);
  }
  else
  {
    v19 = v7;
    v20 = v8;
    v15 = v14;
    v18 = v6;
    v21 = v9;
    while ( 1 )
    {
      ++v13;
      sub_2FA2360(&v22, v12, v15, &v18);
      if ( v10 == v13 )
        break;
      v19 = v23;
      v20 = v24;
      v18 = v22;
      v21 = v25;
      v12 = *v13;
      v15 = (__int64)(*v13 + 512);
    }
    return sub_2FA2360(&v18, v17, v11, &v22);
  }
}
