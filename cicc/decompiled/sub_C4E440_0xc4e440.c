// Function: sub_C4E440
// Address: 0xc4e440
//
__int64 __fastcall sub_C4E440(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned int v4; // eax
  unsigned __int64 v5; // rdx
  const void *v6; // rdx
  unsigned int v7; // eax
  unsigned __int64 v8; // rdx
  __int64 v9; // rdx
  unsigned int v10; // eax
  bool v11; // cc
  __int64 v13; // [rsp+0h] [rbp-80h] BYREF
  unsigned int v14; // [rsp+8h] [rbp-78h]
  __int64 v15; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-68h]
  const void *v17; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v18; // [rsp+28h] [rbp-58h]
  const void *v19; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v20; // [rsp+38h] [rbp-48h]
  unsigned __int64 v21; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v22; // [rsp+48h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 8);
  v18 = v4;
  if ( v4 <= 0x40 )
  {
    v5 = *(_QWORD *)a2;
LABEL_3:
    v6 = (const void *)(*a3 ^ v5);
    v20 = v4;
    v17 = v6;
    v19 = v6;
    v18 = 0;
    v22 = v4;
LABEL_4:
    v21 = (unsigned __int64)v6;
    goto LABEL_5;
  }
  sub_C43780((__int64)&v17, (const void **)a2);
  v4 = v18;
  if ( v18 <= 0x40 )
  {
    v5 = (unsigned __int64)v17;
    goto LABEL_3;
  }
  sub_C43C10(&v17, a3);
  v4 = v18;
  v6 = v17;
  v18 = 0;
  v20 = v4;
  v19 = v17;
  v22 = v4;
  if ( v4 <= 0x40 )
    goto LABEL_4;
  sub_C43780((__int64)&v21, &v19);
  v4 = v22;
  if ( v22 > 0x40 )
  {
    sub_C482E0((__int64)&v21, 1u);
    goto LABEL_7;
  }
LABEL_5:
  if ( v4 == 1 )
    v21 = 0;
  else
    v21 >>= 1;
LABEL_7:
  v7 = *(_DWORD *)(a2 + 8);
  v14 = v7;
  if ( v7 > 0x40 )
  {
    sub_C43780((__int64)&v13, (const void **)a2);
    v7 = v14;
    if ( v14 > 0x40 )
    {
      sub_C43B90(&v13, a3);
      v7 = v14;
      v9 = v13;
      goto LABEL_10;
    }
    v8 = v13;
  }
  else
  {
    v8 = *(_QWORD *)a2;
  }
  v9 = *a3 & v8;
  v13 = v9;
LABEL_10:
  v16 = v7;
  v15 = v9;
  v14 = 0;
  sub_C45EE0((__int64)&v21, &v15);
  v10 = v22;
  v11 = v16 <= 0x40;
  v22 = 0;
  *(_DWORD *)(a1 + 8) = v10;
  *(_QWORD *)a1 = v21;
  if ( !v11 && v15 )
    j_j___libc_free_0_0(v15);
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  return a1;
}
