// Function: sub_2891270
// Address: 0x2891270
//
__int64 __fastcall sub_2891270(__int64 a1, __int64 a2)
{
  unsigned int v2; // r14d
  unsigned int v3; // eax
  _QWORD **v4; // r13
  _QWORD **i; // r12
  __int64 v6; // rax
  _QWORD *v7; // rbx
  unsigned __int64 v8; // r15
  __int64 v9; // rdi
  unsigned int v10; // eax
  _QWORD *v11; // rbx
  _QWORD *v12; // r12
  __int64 v13; // rdi
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

  v2 = 1;
  sub_BBB200((__int64)v17);
  sub_28910D0((__int64)v24, a1 + 169, a2);
  if ( v29 != v30 )
  {
LABEL_2:
    if ( v31 )
      goto LABEL_3;
    goto LABEL_32;
  }
  v2 = v27;
  if ( !v27 )
  {
    LOBYTE(v2) = sub_C8CA60((__int64)v24, (__int64)&qword_4F82400) == 0;
    goto LABEL_2;
  }
  v15 = v25;
  v16 = &v25[v26];
  if ( v25 != v16 )
  {
    while ( *v15 != &qword_4F82400 )
    {
      if ( v16 == ++v15 )
        goto LABEL_31;
    }
    v2 = 0;
  }
LABEL_31:
  if ( !v31 )
  {
LABEL_32:
    _libc_free(v28);
LABEL_3:
    if ( !v27 )
      _libc_free((unsigned __int64)v25);
  }
  sub_C7D6A0(v22, 24LL * v23, 8);
  v3 = v21;
  if ( v21 )
  {
    v4 = (_QWORD **)(v20 + 32LL * v21);
    for ( i = (_QWORD **)(v20 + 8); ; i += 4 )
    {
      v6 = (__int64)*(i - 1);
      if ( v6 != -4096 && v6 != -8192 )
      {
        v7 = *i;
        while ( v7 != i )
        {
          v8 = (unsigned __int64)v7;
          v7 = (_QWORD *)*v7;
          v9 = *(_QWORD *)(v8 + 24);
          if ( v9 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
          j_j___libc_free_0(v8);
        }
      }
      if ( v4 == i + 3 )
        break;
    }
    v3 = v21;
  }
  sub_C7D6A0(v20, 32LL * v3, 8);
  v10 = v19;
  if ( v19 )
  {
    v11 = v18;
    v12 = &v18[2 * v19];
    do
    {
      if ( *v11 != -8192 && *v11 != -4096 )
      {
        v13 = v11[1];
        if ( v13 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
      }
      v11 += 2;
    }
    while ( v12 != v11 );
    v10 = v19;
  }
  sub_C7D6A0((__int64)v18, 16LL * v10, 8);
  return v2;
}
