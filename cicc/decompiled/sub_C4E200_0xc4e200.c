// Function: sub_C4E200
// Address: 0xc4e200
//
__int64 __fastcall sub_C4E200(__int64 a1, __int64 a2, __int64 *a3)
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
  unsigned int v14; // eax
  bool v15; // cc
  __int64 v17; // [rsp+0h] [rbp-80h] BYREF
  unsigned int v18; // [rsp+8h] [rbp-78h]
  __int64 v19; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-68h]
  unsigned __int64 v21; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v22; // [rsp+28h] [rbp-58h]
  unsigned __int64 v23; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v24; // [rsp+38h] [rbp-48h]
  unsigned __int64 v25; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v26; // [rsp+48h] [rbp-38h]

  v5 = *(_DWORD *)(a2 + 8);
  v22 = v5;
  if ( v5 <= 0x40 )
  {
    v6 = *(_QWORD *)a2;
    v7 = v5;
LABEL_3:
    v8 = *a3 ^ v6;
    v24 = v5;
    v21 = v8;
    v23 = v8;
    v22 = 0;
    v26 = v5;
    goto LABEL_4;
  }
  sub_C43780((__int64)&v21, (const void **)a2);
  v5 = v22;
  if ( v22 <= 0x40 )
  {
    v6 = v21;
    v7 = *(_DWORD *)(a2 + 8);
    goto LABEL_3;
  }
  sub_C43C10(&v21, a3);
  v5 = v22;
  v8 = v21;
  v22 = 0;
  v24 = v5;
  v23 = v21;
  v26 = v5;
  if ( v5 <= 0x40 )
  {
    v7 = *(_DWORD *)(a2 + 8);
  }
  else
  {
    sub_C43780((__int64)&v25, (const void **)&v23);
    v5 = v26;
    if ( v26 > 0x40 )
    {
      sub_C44B70((__int64)&v25, 1u);
      v7 = *(_DWORD *)(a2 + 8);
      goto LABEL_9;
    }
    v8 = v25;
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
  v25 = v9;
LABEL_9:
  v18 = v7;
  if ( v7 > 0x40 )
  {
    sub_C43780((__int64)&v17, (const void **)a2);
    v7 = v18;
    if ( v18 > 0x40 )
    {
      sub_C43B90(&v17, a3);
      v7 = v18;
      v13 = v17;
      goto LABEL_12;
    }
    v12 = v17;
  }
  else
  {
    v12 = *(_QWORD *)a2;
  }
  v13 = *a3 & v12;
  v17 = v13;
LABEL_12:
  v20 = v7;
  v19 = v13;
  v18 = 0;
  sub_C45EE0((__int64)&v25, &v19);
  v14 = v26;
  v15 = v20 <= 0x40;
  v26 = 0;
  *(_DWORD *)(a1 + 8) = v14;
  *(_QWORD *)a1 = v25;
  if ( !v15 && v19 )
    j_j___libc_free_0_0(v19);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  return a1;
}
