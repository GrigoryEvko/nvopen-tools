// Function: sub_2647F70
// Address: 0x2647f70
//
__int64 __fastcall sub_2647F70(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r13
  __int64 v4; // rbx
  volatile signed __int32 *v5; // rdi
  unsigned int v6; // r13d
  __int64 v7; // rax
  __int64 v8; // rcx
  unsigned __int64 *v9; // r15
  __int64 (__fastcall *v10)(__int64); // rax
  unsigned __int64 *v11; // r14
  unsigned __int64 *v12; // rdi
  __int64 v13; // rax
  unsigned __int64 v14; // rdi
  unsigned __int64 *v15; // r15
  __int64 (__fastcall *v16)(__int64); // rax
  unsigned __int64 *v17; // r14
  unsigned __int64 *v18; // rdi
  unsigned __int64 v19; // rdi
  __int64 v20; // rbx
  unsigned __int64 v21; // r12
  volatile signed __int32 *v22; // rdi
  __int64 v24; // [rsp+8h] [rbp-C8h]
  __int64 v25; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v26; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v27; // [rsp+28h] [rbp-A8h]
  __int64 *v28; // [rsp+38h] [rbp-98h]
  unsigned __int64 v29; // [rsp+40h] [rbp-90h] BYREF
  __int64 v30; // [rsp+48h] [rbp-88h]
  __int64 v31; // [rsp+50h] [rbp-80h]
  __int64 v32; // [rsp+58h] [rbp-78h]
  _BYTE v33[16]; // [rsp+60h] [rbp-70h] BYREF
  __int64 (__fastcall *v34)(__int64 *); // [rsp+70h] [rbp-60h]
  __int64 v35; // [rsp+78h] [rbp-58h]
  unsigned __int64 v36; // [rsp+80h] [rbp-50h] BYREF
  __int64 v37; // [rsp+88h] [rbp-48h]
  __int64 (__fastcall *v38)(_QWORD *); // [rsp+90h] [rbp-40h]
  __int64 v39; // [rsp+98h] [rbp-38h]
  char v40; // [rsp+A0h] [rbp-30h] BYREF

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
  v6 = 0;
  v7 = *v28;
  v29 = v26;
  v30 = v7;
  v24 = v27;
  v8 = v28[1];
  v31 = v27;
  v32 = v8;
  v25 = v28[1];
  if ( v25 == v7 )
    goto LABEL_26;
  do
  {
    do
    {
      v9 = (unsigned __int64 *)v33;
      v35 = 0;
      v34 = sub_263DA90;
      v10 = sub_263DA70;
      v11 = (unsigned __int64 *)v33;
      v12 = &v29;
      if ( ((unsigned __int8)sub_263DA70 & 1) != 0 )
LABEL_13:
        v10 = *(__int64 (__fastcall **)(__int64))((char *)v10 + *v12 - 1);
      v13 = v10((__int64)v12);
      if ( !v13 )
      {
        while ( 1 )
        {
          v9 += 2;
          if ( &v36 == v9 )
            break;
          v14 = v11[3];
          v10 = (__int64 (__fastcall *)(__int64))v11[2];
          v11 = v9;
          v12 = (unsigned __int64 *)((char *)&v29 + v14);
          if ( ((unsigned __int8)v10 & 1) != 0 )
            goto LABEL_13;
          v13 = v10((__int64)v12);
          if ( v13 )
            goto LABEL_18;
        }
LABEL_38:
        BUG();
      }
LABEL_18:
      LOBYTE(v6) = *(_BYTE *)(*(_QWORD *)v13 + 16LL) | v6;
      if ( (_BYTE)v6 == 3 )
        goto LABEL_29;
      v15 = &v36;
      v39 = 0;
      v38 = sub_263DA40;
      v16 = sub_263DA10;
      v17 = &v36;
      v18 = &v29;
      if ( ((unsigned __int8)sub_263DA10 & 1) != 0 )
LABEL_20:
        v16 = *(__int64 (__fastcall **)(__int64))((char *)v16 + *v18 - 1);
      while ( !(unsigned __int8)v16((__int64)v18) )
      {
        v15 += 2;
        if ( &v40 == (char *)v15 )
          goto LABEL_38;
        v19 = v17[3];
        v16 = (__int64 (__fastcall *)(__int64))v17[2];
        v17 = v15;
        v18 = (unsigned __int64 *)((char *)&v29 + v19);
        if ( ((unsigned __int8)v16 & 1) != 0 )
          goto LABEL_20;
      }
    }
    while ( v25 != v30 );
LABEL_26:
    ;
  }
  while ( v24 != v29 || v25 != v32 || v24 != v31 );
LABEL_29:
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
  return v6;
}
