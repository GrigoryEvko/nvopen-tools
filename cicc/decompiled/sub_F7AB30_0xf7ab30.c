// Function: sub_F7AB30
// Address: 0xf7ab30
//
void __fastcall sub_F7AB30(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 *v7; // r15
  __int64 *v8; // r10
  __int64 v9; // r12
  __int64 *v10; // r14
  __int64 *v11; // rax
  char *v12; // r10
  __int64 *v13; // r13
  __int64 *v14; // r11
  __int64 *v15; // rax
  __int64 v16; // r14
  __int64 v17; // r12
  __int64 v18; // rbx
  bool v19; // r15
  __int64 v20; // rax
  __int64 *v21; // r11
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  bool v26; // al
  char *v28; // [rsp+10h] [rbp-50h]
  __int64 *v29; // [rsp+10h] [rbp-50h]
  __int64 *v31; // [rsp+20h] [rbp-40h]
  __int64 v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+20h] [rbp-40h]
  __int64 v34; // [rsp+28h] [rbp-38h]
  __int64 *v35; // [rsp+28h] [rbp-38h]

  if ( !a5 )
    return;
  v6 = a4;
  if ( !a4 )
    return;
  v7 = a1;
  v8 = a2;
  v9 = a5;
  if ( a5 + a4 != 2 )
  {
    if ( a4 <= a5 )
      goto LABEL_10;
LABEL_5:
    v31 = v8;
    v34 = v6 / 2;
    v10 = &v7[2 * (v6 / 2)];
    v11 = sub_F7AA10(v8, a3, v10, a6);
    v12 = (char *)v31;
    v13 = v11;
    v32 = ((char *)v11 - (char *)v31) >> 4;
    while ( 1 )
    {
      v28 = sub_F797F0((char *)v10, v12, (char *)v13);
      sub_F7AB30(v7, v10, v28, v34, v32, a6);
      v9 -= v32;
      v6 -= v34;
      if ( !v6 )
        break;
      v14 = (__int64 *)v28;
      if ( !v9 )
        break;
      if ( v9 + v6 == 2 )
        goto LABEL_12;
      v8 = v13;
      v7 = (__int64 *)v28;
      if ( v6 > v9 )
        goto LABEL_5;
LABEL_10:
      v29 = v8;
      v32 = v9 / 2;
      v13 = &v8[2 * (v9 / 2)];
      v15 = sub_F7A8F0(v7, (__int64)v8, v13, a6);
      v12 = (char *)v29;
      v10 = v15;
      v34 = ((char *)v15 - (char *)v7) >> 4;
    }
    return;
  }
  v13 = a2;
  v14 = a1;
LABEL_12:
  v16 = v13[1];
  v35 = v14;
  v17 = v14[1];
  v18 = *v13;
  v33 = *v14;
  v19 = *(_BYTE *)(sub_D95540(v16) + 8) == 14;
  if ( v19 != (*(_BYTE *)(sub_D95540(v17) + 8) == 14) )
  {
    v20 = sub_D95540(v16);
    v21 = v35;
    if ( *(_BYTE *)(v20 + 8) != 14 )
      return;
    goto LABEL_14;
  }
  if ( v18 != v33 )
  {
    v25 = sub_F79730(v18, v33, a6);
    v21 = v35;
    if ( v18 == v25 )
      return;
LABEL_14:
    v22 = *v21;
    *v21 = *v13;
    v23 = v13[1];
    *v13 = v22;
    v24 = v21[1];
    v21[1] = v23;
    v13[1] = v24;
    return;
  }
  if ( sub_D969D0(v16) )
  {
    sub_D969D0(v17);
    return;
  }
  v26 = sub_D969D0(v17);
  v21 = v35;
  if ( v26 )
    goto LABEL_14;
}
