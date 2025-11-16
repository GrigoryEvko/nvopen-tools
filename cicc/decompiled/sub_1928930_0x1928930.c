// Function: sub_1928930
// Address: 0x1928930
//
void __fastcall sub_1928930(unsigned int *a1, int *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  unsigned int *v6; // r13
  unsigned int *v7; // rbx
  char v8; // al
  unsigned int *v9; // rdx
  unsigned int v10; // eax
  unsigned int v11; // edx
  unsigned int v12; // eax
  int *v13; // r13
  int *v14; // rbx
  unsigned int *v15; // r15
  char v16; // al
  int v17; // eax
  int v18; // edx
  int v19; // eax
  unsigned int v20; // edx
  unsigned int v21; // eax
  unsigned int v22; // edx
  __int64 v23; // rbx
  __int64 i; // r13
  int *v25; // r15
  __int64 v26; // rcx
  __int64 v27; // rbx
  __int64 v28; // rdx
  bool v29; // zf
  unsigned int v30; // edx
  unsigned int v31; // eax
  int *v32; // [rsp+0h] [rbp-70h]
  __int64 v33; // [rsp+8h] [rbp-68h]
  int *v34; // [rsp+10h] [rbp-60h]
  int *v37; // [rsp+28h] [rbp-48h]
  __int64 v38; // [rsp+30h] [rbp-40h] BYREF
  __int64 v39; // [rsp+38h] [rbp-38h]

  v5 = (char *)a2 - (char *)a1;
  v33 = a3;
  v34 = a2;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return;
  if ( !a3 )
  {
    v37 = a2;
    goto LABEL_22;
  }
  v32 = (int *)(a1 + 2);
  while ( 2 )
  {
    --v33;
    v6 = (unsigned int *)(v34 - 2);
    v7 = &a1[2
           * ((__int64)((((char *)v34 - (char *)a1) >> 3) + ((unsigned __int64)((char *)v34 - (char *)a1) >> 63)) >> 1)];
    v38 = a4;
    v39 = a5;
    v8 = sub_1921830(&v38, v32, v7);
    v9 = (unsigned int *)(v34 - 2);
    if ( !v8 )
    {
      if ( !(unsigned __int8)sub_1921830(&v38, v32, v9) )
      {
        v29 = (unsigned __int8)sub_1921830(&v38, (int *)v7, v6) == 0;
        v10 = *a1;
        if ( v29 )
          goto LABEL_7;
LABEL_29:
        *a1 = *(v34 - 2);
        v30 = *(v34 - 1);
        *(v34 - 2) = v10;
        v31 = a1[1];
        a1[1] = v30;
        *(v34 - 1) = v31;
        goto LABEL_8;
      }
      v10 = *a1;
LABEL_20:
      v20 = a1[2];
      a1[2] = v10;
      v21 = a1[1];
      *a1 = v20;
      v22 = a1[3];
      a1[3] = v21;
      a1[1] = v22;
      goto LABEL_8;
    }
    if ( !(unsigned __int8)sub_1921830(&v38, (int *)v7, v9) )
    {
      v29 = (unsigned __int8)sub_1921830(&v38, v32, v6) == 0;
      v10 = *a1;
      if ( !v29 )
        goto LABEL_29;
      goto LABEL_20;
    }
    v10 = *a1;
LABEL_7:
    *a1 = *v7;
    v11 = v7[1];
    *v7 = v10;
    v12 = a1[1];
    a1[1] = v11;
    v7[1] = v12;
LABEL_8:
    v13 = (int *)(a1 + 2);
    v14 = v34;
    v38 = a4;
    v39 = a5;
    while ( 1 )
    {
      v37 = v13;
      if ( (unsigned __int8)sub_1921830(&v38, v13, a1) )
        goto LABEL_14;
      v15 = (unsigned int *)(v14 - 2);
      do
      {
        v14 = (int *)v15;
        v16 = sub_1921830(&v38, (int *)a1, v15);
        v15 -= 2;
      }
      while ( v16 );
      if ( v13 >= v14 )
        break;
      v17 = *v13;
      *v13 = *v14;
      v18 = v14[1];
      *v14 = v17;
      v19 = v13[1];
      v13[1] = v18;
      v14[1] = v19;
LABEL_14:
      v13 += 2;
    }
    v5 = (char *)v13 - (char *)a1;
    sub_1928930(v13, v34, v33, a4, a5);
    if ( (char *)v13 - (char *)a1 > 128 )
    {
      if ( v33 )
      {
        v34 = v13;
        continue;
      }
LABEL_22:
      v23 = v5 >> 3;
      for ( i = (v23 - 2) >> 1; ; --i )
      {
        sub_1928730((__int64)a1, i, v23, *(_QWORD *)&a1[2 * i], a4, a5);
        if ( !i )
          break;
      }
      v25 = v37 - 2;
      do
      {
        v26 = *(_QWORD *)v25;
        v27 = (char *)v25 - (char *)a1;
        *v25 = *a1;
        v28 = (char *)v25 - (char *)a1;
        v25 -= 2;
        v25[3] = a1[1];
        sub_1928730((__int64)a1, 0, v28 >> 3, v26, a4, a5);
      }
      while ( v27 > 8 );
    }
    break;
  }
}
