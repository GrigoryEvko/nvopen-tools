// Function: sub_3947130
// Address: 0x3947130
//
__int64 __fastcall sub_3947130(__int64 a1, __int64 *a2, int a3, unsigned __int64 *a4)
{
  __int64 v6; // rsi
  unsigned int v8; // eax
  unsigned int v9; // r14d
  _BYTE *v10; // r8
  size_t v11; // r13
  _QWORD *v12; // rax
  unsigned __int64 i; // rdx
  unsigned __int64 v14; // rcx
  char *v15; // rax
  char *v16; // r14
  __int64 v17; // rax
  __int64 *v18; // rdi
  __int64 v19; // rcx
  size_t v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rax
  _QWORD *v23; // r12
  __int64 v24; // rsi
  const void *v25; // r12
  size_t v26; // r13
  unsigned int v27; // r8d
  __int64 *v28; // r9
  __int64 v29; // rdx
  __int64 v31; // rax
  unsigned int v32; // r8d
  __int64 *v33; // r9
  __int64 v34; // r15
  _BYTE *v35; // rdi
  __int64 *v36; // rax
  __int64 *v37; // rax
  __int64 v38; // rax
  _QWORD *v39; // rdi
  size_t v40; // rdx
  __int64 v41; // rax
  _BYTE *v42; // rax
  __int64 *v43; // [rsp+8h] [rbp-C8h]
  unsigned int src; // [rsp+10h] [rbp-C0h]
  _BYTE *srca; // [rsp+10h] [rbp-C0h]
  __int64 *srcb; // [rsp+10h] [rbp-C0h]
  __int64 *v47; // [rsp+18h] [rbp-B8h]
  unsigned int v48; // [rsp+18h] [rbp-B8h]
  unsigned int v50; // [rsp+28h] [rbp-A8h]
  _QWORD v51[2]; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD v52[2]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v53; // [rsp+50h] [rbp-80h]
  __int64 v54[2]; // [rsp+60h] [rbp-70h] BYREF
  __int16 v55; // [rsp+70h] [rbp-60h]
  _QWORD *v56; // [rsp+80h] [rbp-50h] BYREF
  size_t n; // [rsp+88h] [rbp-48h]
  _QWORD v58[8]; // [rsp+90h] [rbp-40h] BYREF

  v6 = a2[1];
  if ( !v6 )
  {
    v9 = 0;
    sub_2241130(a4, 0, a4[1], "Supplied regexp was blank", 0x19u);
    return v9;
  }
  LOBYTE(v8) = sub_16C9F50(*a2, v6);
  v9 = v8;
  if ( (_BYTE)v8 )
  {
    v25 = (const void *)*a2;
    v26 = a2[1];
    v27 = sub_16D19C0(a1, (unsigned __int8 *)*a2, v26);
    v28 = (__int64 *)(*(_QWORD *)a1 + 8LL * v27);
    v29 = *v28;
    if ( *v28 )
    {
      if ( v29 != -8 )
      {
LABEL_29:
        *(_DWORD *)(v29 + 8) = a3;
        return v9;
      }
      --*(_DWORD *)(a1 + 16);
    }
    v43 = v28;
    src = v27;
    v31 = malloc(v26 + 17);
    v32 = src;
    v33 = v43;
    v34 = v31;
    if ( !v31 )
    {
      if ( v26 == -17 )
      {
        v41 = malloc(1u);
        v32 = src;
        v33 = v43;
        if ( v41 )
        {
          v35 = (_BYTE *)(v41 + 16);
          v34 = v41;
          goto LABEL_54;
        }
      }
      srcb = v33;
      v48 = v32;
      sub_16BD1C0("Allocation failed", 1u);
      v32 = v48;
      v33 = srcb;
    }
    v35 = (_BYTE *)(v34 + 16);
    if ( v26 + 1 <= 1 )
    {
LABEL_37:
      v35[v26] = 0;
      *(_QWORD *)v34 = v26;
      *(_DWORD *)(v34 + 8) = 0;
      *v33 = v34;
      ++*(_DWORD *)(a1 + 12);
      v36 = (__int64 *)(*(_QWORD *)a1 + 8LL * (unsigned int)sub_16D1CD0(a1, v32));
      v29 = *v36;
      if ( !*v36 || v29 == -8 )
      {
        v37 = v36 + 1;
        do
        {
          do
            v29 = *v37++;
          while ( v29 == -8 );
        }
        while ( !v29 );
      }
      goto LABEL_29;
    }
LABEL_54:
    v47 = v33;
    v50 = v32;
    v42 = memcpy(v35, v25, v26);
    v33 = v47;
    v32 = v50;
    v35 = v42;
    goto LABEL_37;
  }
  v10 = (_BYTE *)*a2;
  v11 = a2[1];
  v56 = v58;
  if ( &v10[v11] && !v10 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v54[0] = v11;
  if ( v11 > 0xF )
  {
    srca = v10;
    v38 = sub_22409D0((__int64)&v56, (unsigned __int64 *)v54, 0);
    v10 = srca;
    v56 = (_QWORD *)v38;
    v39 = (_QWORD *)v38;
    v58[0] = v54[0];
  }
  else
  {
    if ( v11 == 1 )
    {
      LOBYTE(v58[0]) = *v10;
      v12 = v58;
      goto LABEL_8;
    }
    if ( !v11 )
    {
      v12 = v58;
      goto LABEL_8;
    }
    v39 = v58;
  }
  memcpy(v39, v10, v11);
  v11 = v54[0];
  v12 = v56;
LABEL_8:
  n = v11;
  *((_BYTE *)v12 + v11) = 0;
  sub_394A9D0(a1 + 32, &v56);
  if ( v56 != v58 )
    j_j___libc_free_0((unsigned __int64)v56);
  for ( i = 0; ; i = (unsigned __int64)(v16 + 2) )
  {
    v15 = sub_22417D0(a2, 42, i);
    v16 = v15;
    if ( v15 == (char *)-1LL )
      break;
    v14 = a2[1];
    if ( (unsigned __int64)v15 > v14 )
      sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", "basic_string::replace", (size_t)v15, v14);
    sub_2241130((unsigned __int64 *)a2, (size_t)v15, a2[1] != (_QWORD)v15, ".*", 2u);
  }
  v17 = *a2;
  v55 = 770;
  v51[0] = v17;
  v51[1] = a2[1];
  v52[0] = "^(";
  v52[1] = v51;
  v53 = 1283;
  v54[0] = (__int64)v52;
  v54[1] = (__int64)")$";
  sub_16E2FC0((__int64 *)&v56, (__int64)v54);
  v18 = (__int64 *)*a2;
  if ( v56 == v58 )
  {
    v40 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)v18 = v58[0];
      else
        memcpy(v18, v58, n);
      v40 = n;
      v18 = (__int64 *)*a2;
    }
    a2[1] = v40;
    *((_BYTE *)v18 + v40) = 0;
    v18 = v56;
  }
  else
  {
    v19 = v58[0];
    v20 = n;
    if ( v18 == a2 + 2 )
    {
      *a2 = (__int64)v56;
      a2[1] = v20;
      a2[2] = v19;
    }
    else
    {
      v21 = a2[2];
      *a2 = (__int64)v56;
      a2[1] = v20;
      a2[2] = v19;
      if ( v18 )
      {
        v56 = v18;
        v58[0] = v21;
        goto LABEL_18;
      }
    }
    v56 = v58;
    v18 = v58;
  }
LABEL_18:
  n = 0;
  *(_BYTE *)v18 = 0;
  if ( v56 != v58 )
    j_j___libc_free_0((unsigned __int64)v56);
  sub_16C9340((__int64)v54, *a2, a2[1], 0);
  v9 = sub_16C9430(v54, a4);
  if ( !(_BYTE)v9 )
    goto LABEL_26;
  v22 = sub_22077B0(0x10u);
  v23 = (_QWORD *)v22;
  if ( v22 )
    sub_16C93D0(v22, v54);
  v56 = v23;
  v24 = *(_QWORD *)(a1 + 128);
  LODWORD(n) = a3;
  if ( v24 == *(_QWORD *)(a1 + 136) )
  {
    sub_3946F50((unsigned __int64 *)(a1 + 120), (char *)v24, (__int64 *)&v56);
    v23 = v56;
LABEL_56:
    if ( v23 )
    {
      sub_16C93F0(v23);
      j_j___libc_free_0((unsigned __int64)v23);
    }
    goto LABEL_26;
  }
  if ( !v24 )
  {
    *(_QWORD *)(a1 + 128) = 16;
    goto LABEL_56;
  }
  *(_QWORD *)v24 = v23;
  *(_DWORD *)(v24 + 8) = n;
  *(_QWORD *)(a1 + 128) += 16LL;
LABEL_26:
  sub_16C93F0(v54);
  return v9;
}
