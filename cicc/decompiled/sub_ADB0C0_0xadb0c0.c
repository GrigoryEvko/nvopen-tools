// Function: sub_ADB0C0
// Address: 0xadb0c0
//
__int64 __fastcall sub_ADB0C0(_QWORD *a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // r12
  __int64 v6; // rax
  unsigned __int8 *v7; // rax
  unsigned __int64 v8; // r12
  __int64 **v9; // rax
  __int64 v11; // [rsp+18h] [rbp-88h] BYREF
  __int64 v12; // [rsp+20h] [rbp-80h]
  unsigned int v13; // [rsp+28h] [rbp-78h]
  __int64 v14; // [rsp+30h] [rbp-70h]
  unsigned int v15; // [rsp+38h] [rbp-68h]
  char v16; // [rsp+40h] [rbp-60h]
  __int64 v17; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v18; // [rsp+58h] [rbp-48h]
  __int64 v19; // [rsp+60h] [rbp-40h]
  unsigned int v20; // [rsp+68h] [rbp-38h]
  char v21; // [rsp+70h] [rbp-30h]

  v2 = sub_BCB2D0(*a1);
  v3 = sub_ACD640(v2, 1, 0);
  v4 = *a1;
  v16 = 0;
  v5 = v3;
  v6 = sub_BCE3C0(v4, 0);
  v7 = (unsigned __int8 *)sub_AD6530(v6, 0);
  v21 = 0;
  v11 = v5;
  v8 = sub_AD9FD0((__int64)a1, v7, &v11, 1, 0, (__int64)&v17, 0);
  if ( v21 )
  {
    v21 = 0;
    if ( v20 > 0x40 && v19 )
      j_j___libc_free_0_0(v19);
    if ( v18 > 0x40 && v17 )
      j_j___libc_free_0_0(v17);
  }
  if ( v16 )
  {
    v16 = 0;
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
    if ( v13 > 0x40 && v12 )
      j_j___libc_free_0_0(v12);
  }
  v9 = (__int64 **)sub_BCB2E0(*a1);
  return sub_AD4C50(v8, v9, 0);
}
