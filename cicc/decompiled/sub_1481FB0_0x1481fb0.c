// Function: sub_1481FB0
// Address: 0x1481fb0
//
__int64 __fastcall sub_1481FB0(__int64 a1, __int64 a2, __int64 a3, char a4, char a5)
{
  unsigned int v5; // r15d
  __int64 v8; // rax
  unsigned int v9; // r15d
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 *v15; // rax
  unsigned int v16; // eax
  unsigned int v17; // ebx
  unsigned __int64 v18; // r12
  __int64 *v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rax
  unsigned int v22; // eax
  unsigned int v23; // ebx
  __int64 v25; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v26; // [rsp+18h] [rbp-78h]
  unsigned __int64 v27; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v28; // [rsp+28h] [rbp-68h]
  __int64 v29; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v30; // [rsp+38h] [rbp-58h]
  unsigned __int64 v31; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v32; // [rsp+48h] [rbp-48h]
  unsigned __int64 v33; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v34; // [rsp+58h] [rbp-38h]

  v5 = 0;
  if ( a5 )
    return v5;
  v8 = sub_1456040(a2);
  v9 = sub_1456C90(a1, v8);
  v10 = sub_1456040(a3);
  v11 = sub_145CF80(a1, v10, 1, 0);
  if ( a4 )
  {
    v12 = sub_1477920(a1, a2, 1u);
    sub_158ABC0(&v25, v12);
    v28 = v9;
    v13 = ~(1LL << ((unsigned __int8)v9 - 1));
    if ( v9 <= 0x40 )
    {
      v27 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v9;
    }
    else
    {
      sub_16A4EF0(&v27, -1, 1);
      v13 = ~(1LL << ((unsigned __int8)v9 - 1));
      if ( v28 > 0x40 )
      {
        *(_QWORD *)(v27 + 8LL * ((v9 - 1) >> 6)) &= ~(1LL << ((unsigned __int8)v9 - 1));
        goto LABEL_7;
      }
    }
    v27 &= v13;
LABEL_7:
    v14 = sub_14806B0(a1, a3, v11, 0, 0);
    v15 = sub_1477920(a1, v14, 1u);
    sub_158ABC0(&v29, v15);
    v16 = v28;
    v28 = 0;
    v34 = v16;
    v33 = v27;
    sub_16A7590(&v33, &v29);
    v17 = v34;
    v18 = v33;
    v34 = 0;
    v32 = v17;
    v31 = v33;
    v5 = (unsigned int)sub_16AEA10(&v31, &v25) >> 31;
    if ( v17 > 0x40 )
      goto LABEL_12;
    goto LABEL_16;
  }
  v19 = sub_1477920(a1, a2, 0);
  sub_158A9F0(&v25, v19);
  v28 = v9;
  if ( v9 <= 0x40 )
    v27 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v9;
  else
    sub_16A4EF0(&v27, -1, 1);
  v20 = sub_14806B0(a1, a3, v11, 0, 0);
  v21 = sub_1477920(a1, v20, 0);
  sub_158A9F0(&v29, v21);
  v22 = v28;
  v28 = 0;
  v34 = v22;
  v33 = v27;
  sub_16A7590(&v33, &v29);
  v23 = v34;
  v18 = v33;
  v34 = 0;
  v32 = v23;
  v31 = v33;
  v5 = (unsigned int)sub_16A9900(&v31, &v25) >> 31;
  if ( v23 > 0x40 )
  {
LABEL_12:
    if ( v18 )
    {
      j_j___libc_free_0_0(v18);
      if ( v34 > 0x40 )
      {
        if ( v33 )
          j_j___libc_free_0_0(v33);
      }
    }
  }
LABEL_16:
  if ( v30 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  return v5;
}
