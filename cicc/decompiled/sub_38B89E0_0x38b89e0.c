// Function: sub_38B89E0
// Address: 0x38b89e0
//
void __fastcall sub_38B89E0(__int64 a1, __int64 *a2, __int64 a3, char a4)
{
  __int64 *v4; // r15
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 *v8; // rsi
  __int64 *v9; // r12
  __int64 i; // rax
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rbx
  __int64 *v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // rdx
  __int64 *v18; // rax
  __int64 v19; // rdi
  int v20; // r8d
  int v21; // r9d
  __int64 v22; // rax
  __int64 *v23; // rax
  __int64 v24; // r8
  __int64 v25; // [rsp+8h] [rbp-C8h]
  __int64 *v26; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v27; // [rsp+18h] [rbp-B8h]
  _BYTE v28[176]; // [rsp+20h] [rbp-B0h] BYREF

  v4 = a2;
  v6 = *(_QWORD *)(a1 + 288);
  if ( !v6 )
  {
    v24 = *(_QWORD *)(a1 + 296);
    if ( !v24 )
      return;
    if ( *(_BYTE *)(a1 + 304) == 1 || a4 )
      goto LABEL_4;
LABEL_48:
    sub_15DE580(v24, a2, a3);
    return;
  }
  if ( *(_BYTE *)(a1 + 304) != 1 && !a4 )
  {
    v25 = a3;
    sub_15DC140(v6, a2, a3);
    v24 = *(_QWORD *)(a1 + 296);
    a3 = v25;
    if ( !v24 )
      return;
    goto LABEL_48;
  }
LABEL_4:
  v7 = 2 * a3;
  v8 = (__int64 *)v28;
  v9 = &v4[v7];
  v26 = (__int64 *)v28;
  v27 = 0x800000000LL;
  if ( v4 != &v4[v7] )
  {
    for ( i = 0; ; i = (unsigned int)v27 )
    {
      v11 = 16 * i;
      v12 = *v4;
      v13 = v4[1];
      v14 = &v8[(unsigned __int64)v11 / 8];
      v15 = v11 >> 4;
      v16 = v11 >> 6;
      if ( v16 )
      {
        v17 = &v8[8 * v16];
        v18 = v8;
        while ( v18[1] != v13 || *v18 != v12 )
        {
          if ( v18[3] == v13 && v18[2] == v12 )
          {
            v18 += 2;
            goto LABEL_35;
          }
          if ( v18[4] == v12 && v18[5] == v13 )
          {
            v18 += 4;
            goto LABEL_35;
          }
          if ( v18[7] == v13 && v18[6] == v12 )
          {
            v18 += 6;
            goto LABEL_35;
          }
          v18 += 8;
          if ( v17 == v18 )
          {
            v15 = ((char *)v14 - (char *)v18) >> 4;
            goto LABEL_18;
          }
        }
        goto LABEL_35;
      }
      v18 = v8;
LABEL_18:
      if ( v15 == 2 )
        goto LABEL_54;
      if ( v15 != 3 )
      {
        if ( v15 != 1 )
          goto LABEL_23;
LABEL_21:
        if ( *v18 != v12 || v18[1] != v13 )
          goto LABEL_23;
        goto LABEL_35;
      }
      if ( *v18 != v12 || v18[1] != v13 )
        break;
LABEL_35:
      if ( v14 != v18 )
      {
        v4 += 2;
        if ( v9 == v4 )
          goto LABEL_25;
        continue;
      }
LABEL_23:
      if ( (unsigned __int8)sub_38B85E0(a1, *v4, v4[1]) && !sub_38B8760(a1, v12, v13) )
      {
        v22 = (unsigned int)v27;
        if ( (unsigned int)v27 >= HIDWORD(v27) )
        {
          sub_16CD150((__int64)&v26, v28, 0, 16, v20, v21);
          v22 = (unsigned int)v27;
        }
        v23 = &v26[2 * v22];
        *v23 = v12;
        v23[1] = v13;
        LODWORD(v27) = v27 + 1;
        if ( *(_BYTE *)(a1 + 304) == 1 )
          sub_38B8770(a1, (v13 >> 2) & 1, v12, v13 & 0xFFFFFFFFFFFFFFF8LL, v20, v21);
      }
      v4 += 2;
      v8 = v26;
      if ( v9 == v4 )
      {
LABEL_25:
        if ( *(_BYTE *)(a1 + 304) != 1 )
        {
          v6 = *(_QWORD *)(a1 + 288);
          goto LABEL_27;
        }
        goto LABEL_31;
      }
    }
    v18 += 2;
LABEL_54:
    if ( *v18 != v12 || v18[1] != v13 )
    {
      v18 += 2;
      goto LABEL_21;
    }
    goto LABEL_35;
  }
  if ( *(_BYTE *)(a1 + 304) != 1 )
  {
    v8 = (__int64 *)v28;
LABEL_27:
    if ( v6 )
    {
      sub_15DC140(v6, v8, (unsigned int)v27);
      v8 = v26;
    }
    v19 = *(_QWORD *)(a1 + 296);
    if ( v19 )
    {
      sub_15DE580(v19, v8, (unsigned int)v27);
      v8 = v26;
    }
LABEL_31:
    if ( v8 != (__int64 *)v28 )
      _libc_free((unsigned __int64)v8);
  }
}
