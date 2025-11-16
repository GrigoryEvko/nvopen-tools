// Function: sub_A7A090
// Address: 0xa7a090
//
unsigned __int64 __fastcall sub_A7A090(__int64 *a1, __int64 *a2, int a3, int a4)
{
  unsigned __int64 result; // rax
  __int64 v8; // rax
  const void *v9; // r9
  char *v10; // rdi
  const void *v11; // r8
  signed __int64 v12; // r11
  int v13; // eax
  __int64 v14; // r10
  __int64 v15; // r14
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rax
  signed __int64 v19; // [rsp+8h] [rbp-B8h]
  const void *v20; // [rsp+10h] [rbp-B0h]
  __int64 v21; // [rsp+18h] [rbp-A8h]
  __int64 v22; // [rsp+28h] [rbp-98h]
  int v23; // [rsp+28h] [rbp-98h]
  unsigned __int64 v24; // [rsp+28h] [rbp-98h]
  __int64 v25; // [rsp+28h] [rbp-98h]
  __int64 v26; // [rsp+38h] [rbp-88h] BYREF
  char *v27; // [rsp+40h] [rbp-80h] BYREF
  __int64 v28; // [rsp+48h] [rbp-78h]
  _BYTE dest[112]; // [rsp+50h] [rbp-70h] BYREF

  v26 = sub_A74490(a1, a3);
  if ( (unsigned __int8)sub_A73170(&v26, a4) )
    return *a1;
  v22 = sub_A73290(&v26);
  v8 = sub_A73280(&v26);
  v9 = (const void *)v22;
  v27 = dest;
  v10 = dest;
  v11 = (const void *)v8;
  v12 = v22 - v8;
  v28 = 0x800000000LL;
  v13 = 0;
  v14 = v12 >> 3;
  if ( (unsigned __int64)v12 > 0x40 )
  {
    v19 = v12;
    v20 = v11;
    v21 = v22;
    v25 = v12 >> 3;
    sub_C8D5F0(&v27, dest, v12 >> 3, 8);
    v13 = v28;
    v12 = v19;
    v11 = v20;
    v9 = (const void *)v21;
    LODWORD(v14) = v25;
    v10 = &v27[8 * (unsigned int)v28];
  }
  if ( v9 != v11 )
  {
    v23 = v14;
    memcpy(v10, v11, v12);
    v13 = v28;
    LODWORD(v14) = v23;
  }
  LODWORD(v28) = v14 + v13;
  v15 = sub_A778C0(a2, a4, 0);
  v16 = (unsigned int)v28;
  v17 = (unsigned int)v28 + 1LL;
  if ( v17 > HIDWORD(v28) )
  {
    sub_C8D5F0(&v27, dest, v17, 8);
    v16 = (unsigned int)v28;
  }
  *(_QWORD *)&v27[8 * v16] = v15;
  LODWORD(v28) = v28 + 1;
  v18 = sub_A79C90(a2, v27, (unsigned int)v28);
  result = sub_A78500(a1, (unsigned __int64 *)a2, a3, v18);
  if ( v27 != dest )
  {
    v24 = result;
    _libc_free(v27, a2);
    return v24;
  }
  return result;
}
