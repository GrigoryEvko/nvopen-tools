// Function: sub_1561C30
// Address: 0x1561c30
//
__int64 __fastcall sub_1561C30(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  _QWORD *v5; // r13
  __int64 v7[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v8[8]; // [rsp+10h] [rbp-40h] BYREF

  v3 = a1 + 8;
  v4 = a1 + 16;
  if ( a2 )
  {
    v7[0] = (__int64)v8;
    sub_155C9E0(v7, a2, (__int64)&a2[a3]);
    v5 = (_QWORD *)v7[0];
    LOBYTE(v3) = v4 != sub_1561A70(v3, (__int64)v7);
    if ( v5 != v8 )
      j_j___libc_free_0(v5, v8[0] + 1LL);
  }
  else
  {
    LOBYTE(v8[0]) = 0;
    v7[0] = (__int64)v8;
    v7[1] = 0;
    LOBYTE(v3) = v4 != sub_1561A70(a1 + 8, (__int64)v7);
  }
  return (unsigned int)v3;
}
