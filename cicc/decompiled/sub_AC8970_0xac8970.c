// Function: sub_AC8970
// Address: 0xac8970
//
__int64 __fastcall sub_AC8970(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 *v4; // r14
  __int64 v5; // r13
  unsigned int v6; // edi
  __int64 v7; // rdi
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rbx
  __int64 v13; // r15
  __int64 v14; // r13
  __int64 *v15; // r15
  __int64 v16; // rax
  __int64 *v17; // r12
  __int64 v18; // r13
  _QWORD *v19; // rax
  _QWORD *v20; // r12
  __int64 v23; // r13
  __int64 v24; // r12
  __int64 v25; // rsi
  _QWORD *v26; // r13
  __int64 v27; // [rsp+10h] [rbp-B0h]
  __int64 *v29; // [rsp+38h] [rbp-88h]
  __int64 *v30; // [rsp+48h] [rbp-78h] BYREF
  _QWORD v31[4]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v32[10]; // [rsp+70h] [rbp-50h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(__int64 **)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  v7 = 32LL * v6;
  *(_QWORD *)(a1 + 8) = sub_C7D670(v7, 8);
  v8 = sub_C33690();
  v12 = sub_C33340(v7, 8, v9, v10, v11);
  if ( !v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    if ( v8 == v12 )
      sub_C3C5A0(v32, v12, 1);
    else
      sub_C36740(v32, v8, 1);
    v23 = *(_QWORD *)(a1 + 8);
    v24 = v23 + 32LL * *(unsigned int *)(a1 + 24);
    if ( v23 == v24 )
      return sub_91D830(v32);
    while ( 1 )
    {
      if ( !v23 )
        goto LABEL_54;
      if ( v32[0] == v12 )
      {
        sub_C3C790(v23, v32);
        v23 += 32;
        if ( v24 == v23 )
          return sub_91D830(v32);
      }
      else
      {
        sub_C33EB0(v23, v32);
LABEL_54:
        v23 += 32;
        if ( v24 == v23 )
          return sub_91D830(v32);
      }
    }
  }
  v27 = 32 * v5;
  *(_QWORD *)(a1 + 16) = 0;
  v29 = &v4[4 * v5];
  if ( v8 != v12 )
  {
    sub_C36740(v32, v8, 1);
    v13 = *(_QWORD *)(a1 + 8);
    v14 = v13 + 32LL * *(unsigned int *)(a1 + 24);
    if ( v13 != v14 )
      goto LABEL_9;
    sub_91D830(v32);
LABEL_13:
    sub_C36740(v31, v8, 1);
    sub_C36740(v32, v8, 2);
    goto LABEL_14;
  }
  sub_C3C5A0(v32, v8, 1);
  v13 = *(_QWORD *)(a1 + 8);
  v14 = v13 + 32LL * *(unsigned int *)(a1 + 24);
  if ( v13 == v14 )
  {
    sub_91D830(v32);
    goto LABEL_47;
  }
  do
  {
    while ( 1 )
    {
LABEL_9:
      if ( !v13 )
        goto LABEL_8;
      if ( v32[0] == v12 )
        break;
      sub_C33EB0(v13, v32);
LABEL_8:
      v13 += 32;
      if ( v14 == v13 )
        goto LABEL_12;
    }
    sub_C3C790(v13, v32);
    v13 += 32;
  }
  while ( v14 != v13 );
LABEL_12:
  sub_91D830(v32);
  if ( v8 != v12 )
    goto LABEL_13;
LABEL_47:
  sub_C3C5A0(v31, v8, 1);
  sub_C3C5A0(v32, v8, 2);
LABEL_14:
  v15 = v4;
  if ( v29 != v4 )
  {
    while ( 1 )
    {
      v16 = *v15;
      if ( *v15 == v31[0] )
      {
        if ( v16 == v12 )
        {
          if ( (unsigned __int8)sub_C3E590(v15) )
            goto LABEL_29;
        }
        else if ( (unsigned __int8)sub_C33D00(v15) )
        {
          goto LABEL_29;
        }
        v16 = *v15;
      }
      if ( v16 != v32[0] )
        break;
      if ( !(v16 == v12 ? sub_C3E590(v15) : (unsigned __int8)sub_C33D00(v15)) )
        break;
LABEL_29:
      if ( *v15 != v12 )
      {
LABEL_30:
        sub_C338F0(v15);
LABEL_31:
        v15 += 4;
        if ( v29 == v15 )
          goto LABEL_26;
        continue;
      }
LABEL_22:
      v19 = (_QWORD *)v15[1];
      if ( !v19 )
        goto LABEL_31;
      v20 = &v19[3 * *(v19 - 1)];
      if ( v19 != v20 )
      {
        do
        {
          v20 -= 3;
          sub_91D830(v20);
        }
        while ( (_QWORD *)v15[1] != v20 );
      }
      v15 += 4;
      j_j_j___libc_free_0_0(v20 - 1);
      if ( v29 == v15 )
        goto LABEL_26;
    }
    sub_AC67B0(a1, v15, &v30);
    v17 = v30;
    if ( *v30 == v12 )
    {
      if ( *v15 != v12 )
        goto LABEL_40;
      if ( v30 != v15 )
      {
        v25 = v30[1];
        if ( v25 )
        {
          v26 = (_QWORD *)(v25 + 24LL * *(_QWORD *)(v25 - 8));
          if ( (_QWORD *)v25 != v26 )
          {
            do
            {
              v26 -= 3;
              sub_91D830(v26);
            }
            while ( (_QWORD *)v17[1] != v26 );
          }
          j_j_j___libc_free_0_0(v26 - 1);
        }
        sub_C3C840(v17, v15);
        v17 = v30;
      }
    }
    else
    {
      if ( *v15 != v12 )
      {
        sub_C33870(v30, v15);
        v17 = v30;
        goto LABEL_20;
      }
LABEL_40:
      if ( v30 != v15 )
      {
        sub_91D830(v30);
        if ( *v15 == v12 )
          sub_C3C840(v17, v15);
        else
          sub_C338E0(v17, v15);
        v17 = v30;
      }
    }
LABEL_20:
    v17[3] = v15[3];
    v15[3] = 0;
    ++*(_DWORD *)(a1 + 16);
    v18 = v15[3];
    if ( v18 )
    {
      sub_91D830((_QWORD *)(v18 + 24));
      sub_BD7260(v18);
      sub_BD2DD0(v18);
      if ( *v15 != v12 )
        goto LABEL_30;
      goto LABEL_22;
    }
    goto LABEL_29;
  }
LABEL_26:
  sub_91D830(v32);
  sub_91D830(v31);
  return sub_C7D6A0(v4, v27, 8);
}
