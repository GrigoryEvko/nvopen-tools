// Function: sub_2A85000
// Address: 0x2a85000
//
__int64 __fastcall sub_2A85000(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  unsigned __int64 *v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  unsigned __int64 *v14; // rax
  __int64 *v15; // rdi
  __int64 v16; // r12
  int v17; // eax
  _QWORD *v18; // r12
  __int64 *v19; // r14
  __int64 v20; // rdi
  unsigned int v21; // ecx
  __int64 v22; // rsi
  __int64 *v23; // r14
  __int64 v24; // rsi
  __int64 v25; // rdi
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rdi
  __int64 *v28; // rdi
  __int64 v29; // r12
  __int64 v30; // rax
  __int64 *v31; // rax
  __int64 *v32; // r14
  __int64 i; // rdx
  unsigned int v34; // ecx
  __int64 v35; // rsi
  __int64 *v36; // rax
  __int64 *v37; // r14
  unsigned int v38; // r14d
  unsigned __int64 *v39; // rbx
  unsigned __int64 *v40; // r12
  __int64 v41; // rbx
  unsigned __int64 v42; // r12
  __int64 *v43; // rdi
  __int64 v45; // [rsp+8h] [rbp-C8h]
  __int64 *v47; // [rsp+18h] [rbp-B8h]
  __int64 *v48; // [rsp+18h] [rbp-B8h]
  __int64 v49; // [rsp+18h] [rbp-B8h]
  __int64 *v50; // [rsp+18h] [rbp-B8h]
  __int64 *v51; // [rsp+18h] [rbp-B8h]
  __int64 **v52[2]; // [rsp+20h] [rbp-B0h] BYREF
  const char *v53; // [rsp+30h] [rbp-A0h] BYREF
  char v54; // [rsp+50h] [rbp-80h]
  char v55; // [rsp+51h] [rbp-7Fh]
  unsigned __int64 v56; // [rsp+60h] [rbp-70h] BYREF
  __int64 v57; // [rsp+68h] [rbp-68h]
  __int64 v58; // [rsp+70h] [rbp-60h]
  unsigned __int64 *v59; // [rsp+78h] [rbp-58h]
  unsigned __int64 *v60; // [rsp+80h] [rbp-50h]
  __int64 v61; // [rsp+88h] [rbp-48h]
  __int64 v62; // [rsp+90h] [rbp-40h]
  __int64 v63; // [rsp+98h] [rbp-38h]

  v4 = *a2;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v5 = *(_QWORD *)(v4 + 16);
  v57 = 0;
  v6 = *(_QWORD *)(v4 + 8);
  v56 = 0;
  sub_CA9F50((__int64 *)v52, v6, v5 - v6, (__int64)&v56, 1u, 0);
  v10 = sub_CAFE70((unsigned __int64 *)v52, v6, v7, v8, v9);
  v45 = sub_CA8A20(v52);
  while ( 1 )
  {
    v14 = (unsigned __int64 *)v45;
    if ( v10 )
      break;
LABEL_23:
    while ( 2 )
    {
      if ( !v14 || !*v14 )
        goto LABEL_67;
      v15 = (__int64 *)*v10;
      v16 = *(_QWORD *)(*v10 + 104);
      if ( !v16 )
        goto LABEL_26;
LABEL_8:
      v17 = *(_DWORD *)(v16 + 32);
      if ( v17 )
        goto LABEL_43;
      if ( (unsigned __int8)sub_CAF190((__int64 **)v15, v6, v11, v12, v13) )
      {
LABEL_29:
        v29 = *(_QWORD *)*v10;
        v30 = sub_22077B0(0xA0u);
        if ( v30 )
        {
          v6 = v29;
          v49 = v30;
          sub_CAFBE0(v30, v29);
          v30 = v49;
        }
        v18 = (_QWORD *)*v10;
        *v10 = v30;
        if ( v18 )
        {
          sub_2A80110(v18[16]);
          v31 = (__int64 *)v18[3];
          v32 = &v31[*((unsigned int *)v18 + 8)];
          if ( v31 != v32 )
          {
            for ( i = v18[3]; ; i = v18[3] )
            {
              v50 = v31;
              v34 = (unsigned int)(((__int64)v31 - i) >> 3) >> 7;
              v35 = 4096LL << v34;
              if ( v34 >= 0x1E )
                v35 = 0x40000000000LL;
              sub_C7D6A0(*v31, v35, 16);
              v31 = v50 + 1;
              if ( v32 == v50 + 1 )
                break;
            }
          }
          v36 = (__int64 *)v18[9];
          v37 = &v36[2 * *((unsigned int *)v18 + 20)];
          if ( v36 != v37 )
          {
            do
            {
              v51 = v36;
              sub_C7D6A0(*v36, v36[1], 16);
              v36 = v51 + 2;
            }
            while ( v37 != v51 + 2 );
            v37 = (__int64 *)v18[9];
          }
          v26 = (unsigned __int64)v37;
          if ( v37 != v18 + 11 )
            goto LABEL_19;
LABEL_20:
          v27 = v18[3];
          if ( (_QWORD *)v27 != v18 + 5 )
            _libc_free(v27);
          v6 = 160;
          j_j___libc_free_0((unsigned __int64)v18);
          v14 = (unsigned __int64 *)v45;
          if ( !v10 )
            continue;
          goto LABEL_3;
        }
      }
      else
      {
LABEL_10:
        v18 = (_QWORD *)*v10;
        *v10 = 0;
        if ( v18 )
        {
          sub_2A80110(v18[16]);
          v19 = (__int64 *)v18[3];
          v47 = &v19[*((unsigned int *)v18 + 8)];
          while ( v47 != v19 )
          {
            v20 = *v19;
            v21 = (unsigned int)(((__int64)v19 - v18[3]) >> 3) >> 7;
            v22 = 4096LL << v21;
            if ( v21 >= 0x1E )
              v22 = 0x40000000000LL;
            ++v19;
            sub_C7D6A0(v20, v22, 16);
          }
          v23 = (__int64 *)v18[9];
          v48 = &v23[2 * *((unsigned int *)v18 + 20)];
          if ( v23 != v48 )
          {
            do
            {
              v24 = v23[1];
              v25 = *v23;
              v23 += 2;
              sub_C7D6A0(v25, v24, 16);
            }
            while ( v48 != v23 );
            v48 = (__int64 *)v18[9];
          }
          v26 = (unsigned __int64)v48;
          if ( v48 == v18 + 11 )
            goto LABEL_20;
LABEL_19:
          _libc_free(v26);
          goto LABEL_20;
        }
      }
      break;
    }
  }
LABEL_3:
  v15 = (__int64 *)*v10;
  if ( !*v10 )
    goto LABEL_23;
  if ( v14 && *v14 && v10 == v14 )
  {
LABEL_67:
    v38 = 1;
    goto LABEL_54;
  }
  v16 = v15[13];
  if ( v16 )
    goto LABEL_8;
LABEL_26:
  v16 = sub_CAD820((__int64)v15, v6, v11, v12, v13);
  v15[13] = v16;
  v17 = *(_DWORD *)(v16 + 32);
  if ( !v17 )
  {
LABEL_27:
    v28 = (__int64 *)*v10;
LABEL_28:
    if ( !(unsigned __int8)sub_CAF190((__int64 **)v28, v6, v11, v12, v13) )
      goto LABEL_10;
    goto LABEL_29;
  }
LABEL_43:
  if ( v17 == 4 )
  {
    *(_BYTE *)(v16 + 76) = 0;
    sub_CAEB90(v16, v6);
    if ( !*(_QWORD *)(v16 + 80) )
      v16 = 0;
    while ( v16 )
    {
      v6 = (__int64)v52;
      if ( !(unsigned __int8)sub_2A84E00(a1, (unsigned __int64 *)v52, *(_QWORD *)(v16 + 80), a3, v13) )
        goto LABEL_53;
      sub_CAEB90(v16, (unsigned __int64)v52);
      if ( !*(_QWORD *)(v16 + 80) )
      {
        v28 = (__int64 *)*v10;
        goto LABEL_28;
      }
    }
    goto LABEL_27;
  }
  v6 = v15[13];
  v55 = 1;
  v53 = "DescriptorList node must be a map";
  v54 = 3;
  if ( !v6 )
  {
    v6 = sub_CAD820((__int64)v15, 0, v11, v12, v13);
    v15[13] = v6;
  }
  sub_CA89D0(v52, v6, (__int64)&v53, 0);
LABEL_53:
  v38 = 0;
LABEL_54:
  sub_CA8840((__int64 *)v52, v6);
  v39 = v60;
  v40 = v59;
  if ( v60 != v59 )
  {
    do
    {
      if ( (unsigned __int64 *)*v40 != v40 + 2 )
        j_j___libc_free_0(*v40);
      v40 += 4;
    }
    while ( v39 != v40 );
    v40 = v59;
  }
  if ( v40 )
    j_j___libc_free_0((unsigned __int64)v40);
  v41 = v57;
  v42 = v56;
  if ( v57 != v56 )
  {
    do
    {
      v43 = (__int64 *)v42;
      v42 += 24LL;
      sub_C8EE20(v43);
    }
    while ( v41 != v42 );
    v42 = v56;
  }
  if ( v42 )
    j_j___libc_free_0(v42);
  return v38;
}
