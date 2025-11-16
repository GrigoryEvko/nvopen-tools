// Function: sub_B8C960
// Address: 0xb8c960
//
__int64 __fastcall sub_B8C960(_QWORD *a1, unsigned int a2, int *a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v8; // rdi
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  int *v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r12
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  __int64 v26; // rdi
  _QWORD *v27; // rsi
  __int64 v28; // r12
  __int64 v31; // [rsp+10h] [rbp-70h]
  _QWORD *v32; // [rsp+20h] [rbp-60h] BYREF
  __int64 v33; // [rsp+28h] [rbp-58h]
  _QWORD v34[10]; // [rsp+30h] [rbp-50h] BYREF

  v8 = *a1;
  v32 = v34;
  v33 = 0x400000001LL;
  v9 = sub_BCB2E0(v8);
  v10 = sub_AD64C0(v9, a2, 0);
  v13 = &a3[a4];
  for ( v34[0] = sub_B8C140((__int64)a1, v10, v11, v12); v13 != a3; LODWORD(v33) = v33 + 1 )
  {
    v14 = sub_AD64C0(v9, *a3, 1u);
    v17 = sub_B8C140((__int64)a1, v14, v15, v16);
    v18 = (unsigned int)v33;
    if ( (unsigned __int64)(unsigned int)v33 + 1 > HIDWORD(v33) )
    {
      v31 = v17;
      sub_C8D5F0(&v32, v34, (unsigned int)v33 + 1LL, 8);
      v18 = (unsigned int)v33;
      v17 = v31;
    }
    ++a3;
    v32[v18] = v17;
  }
  v19 = sub_BCB2A0(*a1);
  v20 = sub_AD64C0(v19, a5, 0);
  v23 = sub_B8C140((__int64)a1, v20, v21, v22);
  v24 = (unsigned int)v33;
  v25 = (unsigned int)v33 + 1LL;
  if ( v25 > HIDWORD(v33) )
  {
    sub_C8D5F0(&v32, v34, v25, 8);
    v24 = (unsigned int)v33;
  }
  v32[v24] = v23;
  v26 = *a1;
  v27 = v32;
  LODWORD(v33) = v33 + 1;
  v28 = sub_B9C770(v26, v32, (unsigned int)v33, 0, 1);
  if ( v32 != v34 )
    _libc_free(v32, v27);
  return v28;
}
