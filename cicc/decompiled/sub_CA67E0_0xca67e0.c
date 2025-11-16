// Function: sub_CA67E0
// Address: 0xca67e0
//
char *__fastcall sub_CA67E0(char *a1, unsigned __int64 a2, _QWORD *a3, unsigned __int8 *a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned __int64 v12; // r14
  unsigned __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rsi
  char *v22; // rdx
  char v23; // al
  unsigned __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rcx
  char *v27; // rsi
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rcx
  char *v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // rax
  char v33; // al
  const void *v35; // [rsp+8h] [rbp-68h]
  char v37; // [rsp+18h] [rbp-58h]
  char *v38; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int64 v39; // [rsp+28h] [rbp-48h]
  char *v40; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int64 v41; // [rsp+38h] [rbp-38h]

  v38 = a1;
  v39 = a2;
  v9 = sub_C934D0(&v38, a4, a5, 0);
  if ( v9 == -1 )
    return v38;
  a3[1] = 0;
  v12 = v9;
  if ( v39 > a3[2] )
    sub_C8D290((__int64)a3, a3 + 3, v39, 1u, v10, v11);
  v37 = 0;
  v35 = a3 + 3;
  while ( 1 )
  {
    v22 = v38;
    v23 = v38[v12];
    if ( v23 != 10 && v23 != 13 )
      break;
    v24 = sub_C93740((__int64 *)&v38, (unsigned __int8 *)" \t", 2, v12);
    if ( v24 != -1 )
    {
LABEL_5:
      v13 = v24 + 1;
      if ( v39 <= v13 )
        v13 = v39;
      sub_CA62A0(a3, (char *)(*a3 + a3[1]), v38, &v38[v13]);
      v15 = a3[1];
      v16 = v15 + 1;
      if ( (unsigned __int64)(v15 + 1) > a3[2] )
      {
LABEL_34:
        sub_C8D290((__int64)a3, v35, v16, 1u, v14, (__int64)&v40);
        v15 = a3[1];
      }
LABEL_8:
      *(_BYTE *)(*a3 + v15) = 32;
      ++a3[1];
      v37 = 32;
      goto LABEL_9;
    }
    v15 = a3[1];
    v25 = v15;
    if ( v37 == 10 )
    {
      if ( (unsigned __int64)(v15 + 1) > a3[2] )
      {
        sub_C8D290((__int64)a3, v35, v15 + 1, 1u, v14, v15 + 1);
        v25 = a3[1];
      }
      *(_BYTE *)(*a3 + v25) = 10;
      ++a3[1];
      goto LABEL_9;
    }
    if ( v37 != 32 )
    {
LABEL_33:
      v16 = v15 + 1;
      if ( (unsigned __int64)(v15 + 1) > a3[2] )
        goto LABEL_34;
      goto LABEL_8;
    }
    v37 = 10;
    *(_BYTE *)(*a3 + v15 - 1) = 10;
LABEL_9:
    v17 = v39;
    if ( v39 >= v12 && v39 - v12 > 1 )
      v12 += *(_WORD *)&v38[v12] == 2573;
    v18 = 0;
    if ( v12 + 1 <= v39 )
    {
      v17 = v12 + 1;
      v18 = v39 - (v12 + 1);
    }
    v40 = &v38[v17];
    v41 = v18;
    v19 = sub_C935B0(&v40, (unsigned __int8 *)" \t", 2, 0);
    v20 = v41;
    v21 = 0;
    if ( v19 < v41 )
    {
      v20 = v19;
      v21 = v41 - v19;
    }
    v39 = v21;
    v38 = &v40[v20];
    v12 = sub_C934D0(&v38, a4, a5, 0);
    if ( v12 == -1 )
      goto LABEL_39;
  }
  while ( 1 )
  {
    v26 = v39;
    if ( v39 > v12 )
      v26 = v12;
    else
      v22 = v38;
    v27 = (char *)(*a3 + a3[1]);
    sub_CA62A0(a3, v27, v22, &v22[v26]);
    v28 = v39;
    v29 = 0;
    if ( v39 >= v12 )
    {
      v29 = v39 - v12;
      v28 = v12;
    }
    v41 = v29;
    v40 = &v38[v28];
    if ( !*(_QWORD *)(a6 + 16) )
      sub_4263D6(a3, v27, v38);
    v30 = (char *)(*(__int64 (__fastcall **)(__int64, char **, _QWORD *))(a6 + 24))(a6, &v40, a3);
    v39 = v31;
    v38 = v30;
    v32 = sub_C934D0(&v38, a4, a5, 0);
    v12 = v32;
    if ( v32 == -1 )
      break;
    v22 = v38;
    v33 = v38[v32];
    if ( v33 == 13 || v33 == 10 )
    {
      v24 = sub_C93740((__int64 *)&v38, (unsigned __int8 *)" \t", 2, v12);
      if ( v24 != -1 )
        goto LABEL_5;
      v15 = a3[1];
      goto LABEL_33;
    }
  }
LABEL_39:
  sub_CA62A0(a3, (char *)(*a3 + a3[1]), v38, &v38[v39]);
  return (char *)*a3;
}
