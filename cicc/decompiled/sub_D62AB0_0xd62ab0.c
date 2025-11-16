// Function: sub_D62AB0
// Address: 0xd62ab0
//
__int64 __fastcall sub_D62AB0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // eax
  unsigned int v4; // edx
  unsigned int v5; // r14d
  const void *v6; // rbx
  const void *v7; // rax
  unsigned int v9; // eax
  bool v10; // cc
  const void *v11; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-58h]
  const void *v13; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-48h]
  const void *v15; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-38h]
  __int64 v17; // [rsp+30h] [rbp-30h] BYREF
  unsigned int v18; // [rsp+38h] [rbp-28h]

  *(_DWORD *)(a2 + 392) = 0;
  sub_D62600((__int64)&v15, a2, a3);
  v3 = v18;
  if ( v18 <= 1 )
    goto LABEL_13;
  v4 = v16;
  if ( v16 > 1 )
    goto LABEL_3;
  if ( *(_BYTE *)(a2 + 16) )
  {
LABEL_13:
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 8) = 1;
    *(_DWORD *)(a1 + 24) = 1;
    if ( v3 <= 0x40 )
      goto LABEL_8;
    goto LABEL_14;
  }
  v14 = v18;
  if ( v18 > 0x40 )
  {
    sub_C43690((__int64)&v13, 0, 0);
    if ( v16 > 0x40 && v15 )
      j_j___libc_free_0_0(v15);
    v4 = v14;
    v3 = v18;
    v15 = v13;
    v16 = v14;
    if ( v14 <= 1 || v18 <= 1 )
      goto LABEL_13;
  }
  else
  {
    v15 = 0;
    v4 = v18;
    v16 = v18;
  }
LABEL_3:
  v12 = v4;
  if ( v4 > 0x40 )
    sub_C43780((__int64)&v11, &v15);
  else
    v11 = v15;
  sub_C45EE0((__int64)&v11, &v17);
  v5 = v12;
  v12 = 0;
  v6 = v11;
  v14 = v16;
  if ( v16 > 0x40 )
  {
    sub_C43780((__int64)&v13, &v15);
    v9 = v14;
    v10 = v12 <= 0x40;
    *(_QWORD *)a1 = v6;
    *(_DWORD *)(a1 + 8) = v5;
    *(_DWORD *)(a1 + 24) = v9;
    *(_QWORD *)(a1 + 16) = v13;
    if ( !v10 && v11 )
      j_j___libc_free_0_0(v11);
  }
  else
  {
    *(_DWORD *)(a1 + 24) = v16;
    v7 = v15;
    *(_DWORD *)(a1 + 8) = v5;
    *(_QWORD *)a1 = v6;
    *(_QWORD *)(a1 + 16) = v7;
  }
  if ( v18 > 0x40 )
  {
LABEL_14:
    if ( v17 )
      j_j___libc_free_0_0(v17);
  }
LABEL_8:
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  return a1;
}
