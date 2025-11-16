// Function: sub_DFED00
// Address: 0xdfed00
//
__int64 __fastcall sub_DFED00(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 *v4; // rdi
  unsigned int v5; // eax
  _QWORD **v6; // r13
  _QWORD **i; // r12
  __int64 v8; // rax
  _QWORD *v9; // rbx
  _QWORD *v10; // r15
  __int64 v11; // rdi
  unsigned int v12; // eax
  _QWORD *v13; // rbx
  _QWORD *v14; // r12
  __int64 v15; // rdi
  __int64 v17; // [rsp+8h] [rbp-98h] BYREF
  _BYTE v18[8]; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v19; // [rsp+18h] [rbp-88h]
  unsigned int v20; // [rsp+28h] [rbp-78h]
  __int64 v21; // [rsp+38h] [rbp-68h]
  unsigned int v22; // [rsp+48h] [rbp-58h]
  __int64 v23; // [rsp+58h] [rbp-48h]
  unsigned int v24; // [rsp+68h] [rbp-38h]

  sub_BBB200((__int64)v18);
  v3 = a1 + 208;
  sub_DFE9F0((__int64)&v17, a1 + 176, a2);
  v4 = (__int64 *)(a1 + 208);
  if ( *(_BYTE *)(a1 + 216) )
  {
    sub_DFE910(v4, &v17);
  }
  else
  {
    sub_DF93A0(v4, &v17);
    *(_BYTE *)(a1 + 216) = 1;
  }
  sub_DFE7B0(&v17);
  sub_C7D6A0(v23, 24LL * v24, 8);
  v5 = v22;
  if ( v22 )
  {
    v6 = (_QWORD **)(v21 + 32LL * v22);
    for ( i = (_QWORD **)(v21 + 8); ; i += 4 )
    {
      v8 = (__int64)*(i - 1);
      if ( v8 != -4096 && v8 != -8192 )
      {
        v9 = *i;
        while ( v9 != i )
        {
          v10 = v9;
          v9 = (_QWORD *)*v9;
          v11 = v10[3];
          if ( v11 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
          j_j___libc_free_0(v10, 32);
        }
      }
      if ( v6 == i + 3 )
        break;
    }
    v5 = v22;
  }
  sub_C7D6A0(v21, 32LL * v5, 8);
  v12 = v20;
  if ( v20 )
  {
    v13 = v19;
    v14 = &v19[2 * v20];
    do
    {
      if ( *v13 != -4096 && *v13 != -8192 )
      {
        v15 = v13[1];
        if ( v15 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v15 + 8LL))(v15);
      }
      v13 += 2;
    }
    while ( v14 != v13 );
    v12 = v20;
  }
  sub_C7D6A0((__int64)v19, 16LL * v12, 8);
  return v3;
}
