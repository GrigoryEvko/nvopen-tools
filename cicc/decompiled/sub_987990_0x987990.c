// Function: sub_987990
// Address: 0x987990
//
__int64 __fastcall sub_987990(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r13d
  unsigned int v4; // r15d
  bool v6; // cc
  __int64 v7; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v8; // [rsp+8h] [rbp-58h]
  __int64 v9; // [rsp+10h] [rbp-50h] BYREF
  int v10; // [rsp+18h] [rbp-48h]
  __int64 v11; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v12; // [rsp+28h] [rbp-38h]

  v3 = a3;
  v4 = *(_DWORD *)(a2 + 8);
  sub_C449B0(&v7, a2, a3);
  if ( v8 != v4 )
  {
    if ( v4 > 0x3F || v8 > 0x40 )
      sub_C43C90(&v7, v4, v8);
    else
      v7 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v4 - (unsigned __int8)v8 + 64) << v4;
  }
  sub_C449B0(&v9, a2 + 16, v3);
  v12 = v8;
  if ( v8 > 0x40 )
  {
    sub_C43780(&v11, &v7);
    v6 = v8 <= 0x40;
    *(_DWORD *)(a1 + 8) = v12;
    *(_QWORD *)a1 = v11;
    *(_DWORD *)(a1 + 24) = v10;
    *(_QWORD *)(a1 + 16) = v9;
    if ( !v6 && v7 )
      j_j___libc_free_0_0(v7);
  }
  else
  {
    *(_DWORD *)(a1 + 8) = v8;
    *(_QWORD *)a1 = v7;
    *(_DWORD *)(a1 + 24) = v10;
    *(_QWORD *)(a1 + 16) = v9;
  }
  return a1;
}
