// Function: sub_C4E630
// Address: 0xc4e630
//
__int64 __fastcall sub_C4E630(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned int v5; // eax
  unsigned __int64 v6; // rdx
  unsigned int v7; // esi
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned int v15; // eax
  bool v16; // cc
  __int64 v18; // [rsp+0h] [rbp-80h] BYREF
  unsigned int v19; // [rsp+8h] [rbp-78h]
  __int64 v20; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v21; // [rsp+18h] [rbp-68h]
  unsigned __int64 v22; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v23; // [rsp+28h] [rbp-58h]
  unsigned __int64 v24; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v25; // [rsp+38h] [rbp-48h]
  unsigned __int64 v26; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v27; // [rsp+48h] [rbp-38h]

  v5 = *(_DWORD *)(a2 + 8);
  v23 = v5;
  if ( v5 <= 0x40 )
  {
    v6 = *(_QWORD *)a2;
    v7 = v5;
LABEL_3:
    v8 = *a3 ^ v6;
    v25 = v5;
    v22 = v8;
    v24 = v8;
    v23 = 0;
    v27 = v5;
    goto LABEL_4;
  }
  sub_C43780((__int64)&v22, (const void **)a2);
  v5 = v23;
  if ( v23 <= 0x40 )
  {
    v6 = v22;
    v7 = *(_DWORD *)(a2 + 8);
    goto LABEL_3;
  }
  sub_C43C10(&v22, a3);
  v5 = v23;
  v8 = v22;
  v23 = 0;
  v25 = v5;
  v24 = v22;
  v27 = v5;
  if ( v5 <= 0x40 )
  {
    v7 = *(_DWORD *)(a2 + 8);
  }
  else
  {
    sub_C43780((__int64)&v26, (const void **)&v24);
    v5 = v27;
    if ( v27 > 0x40 )
    {
      sub_C44B70((__int64)&v26, 1u);
      v7 = *(_DWORD *)(a2 + 8);
      goto LABEL_9;
    }
    v8 = v26;
    v7 = *(_DWORD *)(a2 + 8);
  }
LABEL_4:
  v9 = 0;
  if ( v5 )
  {
    v10 = (__int64)(v8 << (64 - (unsigned __int8)v5)) >> (64 - (unsigned __int8)v5);
    if ( v5 == 1 )
      v11 = v10 >> 63;
    else
      v11 = v10 >> 1;
    v9 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v5) & v11;
  }
  v26 = v9;
LABEL_9:
  v19 = v7;
  if ( v7 <= 0x40 )
  {
    v12 = *(_QWORD *)a2;
LABEL_11:
    v13 = *a3 | v12;
    v18 = v13;
    goto LABEL_12;
  }
  sub_C43780((__int64)&v18, (const void **)a2);
  v7 = v19;
  if ( v19 <= 0x40 )
  {
    v12 = v18;
    goto LABEL_11;
  }
  sub_C43BD0(&v18, a3);
  v7 = v19;
  v13 = v18;
LABEL_12:
  v21 = v7;
  v20 = v13;
  v19 = 0;
  if ( v27 > 0x40 )
  {
    sub_C43D10((__int64)&v26);
  }
  else
  {
    v14 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v27) & ~v26;
    if ( !v27 )
      v14 = 0;
    v26 = v14;
  }
  sub_C46250((__int64)&v26);
  sub_C45EE0((__int64)&v26, &v20);
  v15 = v27;
  v16 = v21 <= 0x40;
  v27 = 0;
  *(_DWORD *)(a1 + 8) = v15;
  *(_QWORD *)a1 = v26;
  if ( !v16 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  return a1;
}
