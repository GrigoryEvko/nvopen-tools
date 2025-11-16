// Function: sub_120B460
// Address: 0x120b460
//
__int64 __fastcall sub_120B460(__int64 **a1, __int64 *a2)
{
  unsigned int v2; // r12d
  _QWORD *v4; // [rsp+0h] [rbp-40h] BYREF
  size_t v5; // [rsp+8h] [rbp-38h]
  _QWORD v6[6]; // [rsp+10h] [rbp-30h] BYREF

  v4 = v6;
  v5 = 0;
  LOBYTE(v6[0]) = 0;
  v2 = sub_120B3D0((__int64)a1, (__int64)&v4);
  if ( !(_BYTE)v2 )
    *a2 = sub_B9B140(*a1, v4, v5);
  if ( v4 != v6 )
    j_j___libc_free_0(v4, v6[0] + 1LL);
  return v2;
}
