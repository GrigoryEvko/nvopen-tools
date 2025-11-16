// Function: sub_28FDC50
// Address: 0x28fdc50
//
__int64 __fastcall sub_28FDC50(_QWORD *a1, __int64 a2)
{
  unsigned int v2; // r14d
  unsigned int v4; // eax
  _QWORD **v5; // r13
  _QWORD **i; // r12
  __int64 v7; // rax
  _QWORD *v8; // rbx
  unsigned __int64 v9; // r15
  __int64 v10; // rdi
  unsigned int v11; // eax
  _QWORD *v12; // rbx
  _QWORD *v13; // r12
  __int64 v14; // rdi
  __int64 **v15; // rax
  __int64 **v16; // rdx
  _BYTE v17[8]; // [rsp+0h] [rbp-F0h] BYREF
  _QWORD *v18; // [rsp+8h] [rbp-E8h]
  unsigned int v19; // [rsp+18h] [rbp-D8h]
  __int64 v20; // [rsp+28h] [rbp-C8h]
  unsigned int v21; // [rsp+38h] [rbp-B8h]
  __int64 v22; // [rsp+48h] [rbp-A8h]
  unsigned int v23; // [rsp+58h] [rbp-98h]
  _BYTE v24[8]; // [rsp+60h] [rbp-90h] BYREF
  __int64 **v25; // [rsp+68h] [rbp-88h]
  int v26; // [rsp+74h] [rbp-7Ch]
  unsigned __int8 v27; // [rsp+7Ch] [rbp-74h]
  unsigned __int64 v28; // [rsp+98h] [rbp-58h]
  int v29; // [rsp+A4h] [rbp-4Ch]
  int v30; // [rsp+A8h] [rbp-48h]
  char v31; // [rsp+ACh] [rbp-44h]

  v2 = 0;
  if ( (unsigned __int8)sub_BB98D0(a1, a2) )
    return v2;
  v2 = 1;
  sub_BBB200((__int64)v17);
  sub_28FC5C0((__int64)v24, (__int64)(a1 + 22), a2);
  if ( v29 != v30 )
  {
LABEL_4:
    if ( v31 )
      goto LABEL_5;
    goto LABEL_34;
  }
  v2 = v27;
  if ( !v27 )
  {
    LOBYTE(v2) = sub_C8CA60((__int64)v24, (__int64)&qword_4F82400) == 0;
    goto LABEL_4;
  }
  v15 = v25;
  v16 = &v25[v26];
  if ( v25 != v16 )
  {
    while ( *v15 != &qword_4F82400 )
    {
      if ( v16 == ++v15 )
        goto LABEL_33;
    }
    v2 = 0;
  }
LABEL_33:
  if ( v31 )
    goto LABEL_7;
LABEL_34:
  _libc_free(v28);
LABEL_5:
  if ( !v27 )
    _libc_free((unsigned __int64)v25);
LABEL_7:
  sub_C7D6A0(v22, 24LL * v23, 8);
  v4 = v21;
  if ( v21 )
  {
    v5 = (_QWORD **)(v20 + 32LL * v21);
    for ( i = (_QWORD **)(v20 + 8); ; i += 4 )
    {
      v7 = (__int64)*(i - 1);
      if ( v7 != -4096 && v7 != -8192 )
      {
        v8 = *i;
        while ( v8 != i )
        {
          v9 = (unsigned __int64)v8;
          v8 = (_QWORD *)*v8;
          v10 = *(_QWORD *)(v9 + 24);
          if ( v10 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 8LL))(v10);
          j_j___libc_free_0(v9);
        }
      }
      if ( v5 == i + 3 )
        break;
    }
    v4 = v21;
  }
  sub_C7D6A0(v20, 32LL * v4, 8);
  v11 = v19;
  if ( v19 )
  {
    v12 = v18;
    v13 = &v18[2 * v19];
    do
    {
      if ( *v12 != -4096 && *v12 != -8192 )
      {
        v14 = v12[1];
        if ( v14 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
      }
      v12 += 2;
    }
    while ( v13 != v12 );
    v11 = v19;
  }
  sub_C7D6A0((__int64)v18, 16LL * v11, 8);
  return v2;
}
