// Function: sub_1FC14D0
// Address: 0x1fc14d0
//
__int64 __fastcall sub_1FC14D0(__int64 a1, __int64 a2, int a3, int a4, unsigned __int8 a5)
{
  char v7; // si
  char v8; // al
  __int64 v9; // rdi
  unsigned int v10; // r13d
  __int64 v12; // [rsp+8h] [rbp-70h] BYREF
  unsigned int v13; // [rsp+10h] [rbp-68h]
  __int64 v14; // [rsp+18h] [rbp-60h] BYREF
  unsigned int v15; // [rsp+20h] [rbp-58h]
  __int64 v16; // [rsp+28h] [rbp-50h] BYREF
  char v17; // [rsp+30h] [rbp-48h]
  char v18; // [rsp+31h] [rbp-47h]
  __int64 v19; // [rsp+38h] [rbp-40h]
  int v20; // [rsp+40h] [rbp-38h]
  __int64 v21; // [rsp+48h] [rbp-30h]
  int v22; // [rsp+50h] [rbp-28h]

  v7 = *(_BYTE *)(a1 + 25);
  v8 = *(_BYTE *)(a1 + 24);
  v19 = 0;
  v16 = *(_QWORD *)a1;
  v9 = *(_QWORD *)(a1 + 8);
  v17 = v7;
  v18 = v8;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v13 = 1;
  v12 = 0;
  v15 = 1;
  v14 = 0;
  v10 = sub_20A8CB0(v9, a2, a3, a4, (unsigned int)&v12, (unsigned int)&v14, (__int64)&v16, 0, a5);
  if ( (_BYTE)v10 )
  {
    sub_1F81BC0(a1, a2);
    sub_1FB18E0(a1, &v16);
  }
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  return v10;
}
