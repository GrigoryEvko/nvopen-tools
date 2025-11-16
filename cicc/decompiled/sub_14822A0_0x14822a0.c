// Function: sub_14822A0
// Address: 0x14822a0
//
__int64 __fastcall sub_14822A0(__int64 a1, __int64 a2, __int64 a3, char a4, char a5)
{
  unsigned int v5; // r15d
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 *v14; // rax
  unsigned int v15; // eax
  unsigned int v16; // ebx
  __int64 v17; // r12
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rax
  unsigned int v21; // eax
  unsigned int v22; // ebx
  __int64 v24; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v25; // [rsp+18h] [rbp-78h]
  __int64 v26; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v27; // [rsp+28h] [rbp-68h]
  __int64 v28; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v29; // [rsp+38h] [rbp-58h]
  __int64 v30; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v31; // [rsp+48h] [rbp-48h]
  __int64 v32; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v33; // [rsp+58h] [rbp-38h]

  v5 = 0;
  if ( a5 )
    return v5;
  v8 = sub_1456040(a2);
  v5 = sub_1456C90(a1, v8);
  v9 = sub_1456040(a3);
  v10 = sub_145CF80(a1, v9, 1, 0);
  if ( a4 )
  {
    v11 = sub_1477920(a1, a2, 1u);
    sub_158ACE0(&v24, v11);
    v27 = v5;
    v12 = 1LL << ((unsigned __int8)v5 - 1);
    if ( v5 <= 0x40 )
    {
      v26 = 0;
    }
    else
    {
      sub_16A4EF0(&v26, 0, 0);
      v12 = 1LL << ((unsigned __int8)v5 - 1);
      if ( v27 > 0x40 )
      {
        *(_QWORD *)(v26 + 8LL * ((v5 - 1) >> 6)) |= 1LL << ((unsigned __int8)v5 - 1);
        goto LABEL_7;
      }
    }
    v26 |= v12;
LABEL_7:
    v13 = sub_14806B0(a1, a3, v10, 0, 0);
    v14 = sub_1477920(a1, v13, 1u);
    sub_158ABC0(&v28, v14);
    v15 = v27;
    v27 = 0;
    v33 = v15;
    v32 = v26;
    sub_16A7200(&v32, &v28);
    v16 = v33;
    v17 = v32;
    v33 = 0;
    v31 = v16;
    v30 = v32;
    LOBYTE(v5) = (int)sub_16AEA10(&v30, &v24) > 0;
    if ( v16 > 0x40 )
      goto LABEL_12;
    goto LABEL_16;
  }
  v18 = sub_1477920(a1, a2, 0);
  sub_158AAD0(&v24, v18);
  v27 = v5;
  if ( v5 <= 0x40 )
    v26 = 0;
  else
    sub_16A4EF0(&v26, 0, 0);
  v19 = sub_14806B0(a1, a3, v10, 0, 0);
  v20 = sub_1477920(a1, v19, 0);
  sub_158A9F0(&v28, v20);
  v21 = v27;
  v27 = 0;
  v33 = v21;
  v32 = v26;
  sub_16A7200(&v32, &v28);
  v22 = v33;
  v17 = v32;
  v33 = 0;
  v31 = v22;
  v30 = v32;
  LOBYTE(v5) = (int)sub_16A9900(&v30, &v24) > 0;
  if ( v22 > 0x40 )
  {
LABEL_12:
    if ( v17 )
    {
      j_j___libc_free_0_0(v17);
      if ( v33 > 0x40 )
      {
        if ( v32 )
          j_j___libc_free_0_0(v32);
      }
    }
  }
LABEL_16:
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  return v5;
}
