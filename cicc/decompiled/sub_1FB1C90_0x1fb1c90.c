// Function: sub_1FB1C90
// Address: 0x1fb1c90
//
__int64 __fastcall sub_1FB1C90(__int64 a1, __int64 a2, int a3, int a4)
{
  char v6; // si
  char v7; // al
  __int64 v8; // rdi
  unsigned int v9; // r13d
  __int64 v11; // [rsp+0h] [rbp-70h] BYREF
  __int64 v12; // [rsp+8h] [rbp-68h]
  __int64 v13; // [rsp+10h] [rbp-60h]
  __int64 v14; // [rsp+18h] [rbp-58h]
  __int64 v15; // [rsp+20h] [rbp-50h] BYREF
  char v16; // [rsp+28h] [rbp-48h]
  char v17; // [rsp+29h] [rbp-47h]
  __int64 v18; // [rsp+30h] [rbp-40h]
  int v19; // [rsp+38h] [rbp-38h]
  __int64 v20; // [rsp+40h] [rbp-30h]
  int v21; // [rsp+48h] [rbp-28h]

  v6 = *(_BYTE *)(a1 + 25);
  v7 = *(_BYTE *)(a1 + 24);
  v18 = 0;
  v15 = *(_QWORD *)a1;
  v8 = *(_QWORD *)(a1 + 8);
  v16 = v6;
  v17 = v7;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v11 = 0;
  v12 = 1;
  v13 = 0;
  v14 = 1;
  v9 = sub_20A2AF0(v8, a2, a3, a4, (unsigned int)&v11, (unsigned int)&v15, 0, 0);
  if ( (_BYTE)v9 )
  {
    sub_1F81BC0(a1, a2);
    sub_1FB18E0(a1, &v15);
  }
  if ( (unsigned int)v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  if ( (unsigned int)v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  return v9;
}
