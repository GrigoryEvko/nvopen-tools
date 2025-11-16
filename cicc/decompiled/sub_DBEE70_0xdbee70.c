// Function: sub_DBEE70
// Address: 0xdbee70
//
_QWORD *__fastcall sub_DBEE70(__int64 a1, _DWORD *a2, __int64 *a3)
{
  __int64 v4; // rax
  unsigned int v5; // eax
  unsigned int v6; // ebx
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned int v9; // eax
  _QWORD *v10; // r12
  __int64 v12; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v13; // [rsp+8h] [rbp-48h]
  unsigned __int64 v14; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-38h]
  unsigned __int64 v16; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-28h]

  v4 = sub_D95540(a1);
  v5 = sub_D97050((__int64)a3, v4);
  *a2 = 36;
  v6 = v5;
  v7 = sub_DBB9F0((__int64)a3, a1, 0, 0);
  sub_AB0910((__int64)&v14, v7);
  v13 = v6;
  if ( v6 > 0x40 )
    sub_C43690((__int64)&v12, 0, 0);
  else
    v12 = 0;
  if ( v15 > 0x40 )
  {
    sub_C43D10((__int64)&v14);
  }
  else
  {
    v8 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v15) & ~v14;
    if ( !v15 )
      v8 = 0;
    v14 = v8;
  }
  sub_C46250((__int64)&v14);
  sub_C45EE0((__int64)&v14, &v12);
  v9 = v15;
  v15 = 0;
  v17 = v9;
  v16 = v14;
  v10 = sub_DA26C0(a3, (__int64)&v16);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  return v10;
}
