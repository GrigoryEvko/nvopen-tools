// Function: sub_2647CD0
// Address: 0x2647cd0
//
__int64 __fastcall sub_2647CD0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r13
  __int64 v4; // rbx
  volatile signed __int32 *v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rcx
  unsigned __int64 *v8; // r14
  __int64 (__fastcall *v9)(__int64); // rax
  unsigned __int64 *v10; // r15
  unsigned __int64 *v11; // rdi
  __int64 v12; // rax
  unsigned __int64 v13; // rdi
  unsigned __int64 *v14; // r14
  __int64 (__fastcall *v15)(__int64); // rax
  unsigned __int64 *v16; // r15
  unsigned __int64 *v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned int v19; // r13d
  __int64 v20; // rbx
  unsigned __int64 v21; // r12
  volatile signed __int32 *v22; // rdi
  __int64 v24; // [rsp+0h] [rbp-C0h]
  __int64 v25; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v26; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v27; // [rsp+18h] [rbp-A8h]
  __int64 *v28; // [rsp+28h] [rbp-98h]
  unsigned __int64 v29; // [rsp+30h] [rbp-90h] BYREF
  __int64 v30; // [rsp+38h] [rbp-88h]
  __int64 v31; // [rsp+40h] [rbp-80h]
  __int64 v32; // [rsp+48h] [rbp-78h]
  _BYTE v33[16]; // [rsp+50h] [rbp-70h] BYREF
  __int64 (__fastcall *v34)(__int64 *); // [rsp+60h] [rbp-60h]
  __int64 v35; // [rsp+68h] [rbp-58h]
  unsigned __int64 v36; // [rsp+70h] [rbp-50h] BYREF
  __int64 v37; // [rsp+78h] [rbp-48h]
  __int64 (__fastcall *v38)(_QWORD *); // [rsp+80h] [rbp-40h]
  __int64 v39; // [rsp+88h] [rbp-38h]
  char v40; // [rsp+90h] [rbp-30h] BYREF

  if ( *(_BYTE *)a1 || byte_4FF3628 )
  {
    sub_2640E50(&v36, (_QWORD *)(a1 + 72), a3);
  }
  else
  {
    v36 = 0;
    v37 = 0;
    v38 = 0;
  }
  sub_2640E50(&v26, &v36, a3);
  v28 = (__int64 *)(a1 + 48);
  v3 = v36;
  v4 = v37;
  if ( v37 != v36 )
  {
    do
    {
      v5 = *(volatile signed __int32 **)(v3 + 8);
      if ( v5 )
        sub_A191D0(v5);
      v3 += 16LL;
    }
    while ( v4 != v3 );
    v3 = v36;
  }
  if ( v3 )
    j_j___libc_free_0(v3);
  v6 = *v28;
  v29 = v26;
  v30 = v6;
  v24 = v27;
  v7 = v28[1];
  v31 = v27;
  v32 = v7;
  v25 = v28[1];
  if ( v25 == v6 )
    goto LABEL_26;
  while ( 1 )
  {
    v8 = (unsigned __int64 *)v33;
    v35 = 0;
    v34 = sub_263DA90;
    v9 = sub_263DA70;
    v10 = (unsigned __int64 *)v33;
    v11 = &v29;
    if ( ((unsigned __int8)sub_263DA70 & 1) != 0 )
LABEL_13:
      v9 = *(__int64 (__fastcall **)(__int64))((char *)v9 + *v11 - 1);
    v12 = v9((__int64)v11);
    if ( !v12 )
    {
      while ( 1 )
      {
        v8 += 2;
        if ( &v36 == v8 )
          break;
        v13 = v10[3];
        v9 = (__int64 (__fastcall *)(__int64))v10[2];
        v10 = v8;
        v11 = (unsigned __int64 *)((char *)&v29 + v13);
        if ( ((unsigned __int8)v9 & 1) != 0 )
          goto LABEL_13;
        v12 = v9((__int64)v11);
        if ( v12 )
          goto LABEL_18;
      }
LABEL_40:
      BUG();
    }
LABEL_18:
    if ( *(_DWORD *)(*(_QWORD *)v12 + 40LL) )
      break;
    v14 = &v36;
    v39 = 0;
    v38 = sub_263DA40;
    v15 = sub_263DA10;
    v16 = &v36;
    v17 = &v29;
    if ( ((unsigned __int8)sub_263DA10 & 1) != 0 )
LABEL_20:
      v15 = *(__int64 (__fastcall **)(__int64))((char *)v15 + *v17 - 1);
    while ( !(unsigned __int8)v15((__int64)v17) )
    {
      v14 += 2;
      if ( &v40 == (char *)v14 )
        goto LABEL_40;
      v18 = v16[3];
      v15 = (__int64 (__fastcall *)(__int64))v16[2];
      v16 = v14;
      v17 = (unsigned __int64 *)((char *)&v29 + v18);
      if ( ((unsigned __int8)v15 & 1) != 0 )
        goto LABEL_20;
    }
    if ( v25 == v30 )
    {
LABEL_26:
      if ( v24 == v29 && v25 == v32 && v24 == v31 )
      {
        v19 = 1;
        goto LABEL_32;
      }
    }
  }
  v19 = 0;
LABEL_32:
  v20 = v27;
  v21 = v26;
  if ( v27 != v26 )
  {
    do
    {
      v22 = *(volatile signed __int32 **)(v21 + 8);
      if ( v22 )
        sub_A191D0(v22);
      v21 += 16LL;
    }
    while ( v20 != v21 );
    v21 = v26;
  }
  if ( v21 )
    j_j___libc_free_0(v21);
  return v19;
}
