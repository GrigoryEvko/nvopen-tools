// Function: sub_CE87C0
// Address: 0xce87c0
//
__int64 __fastcall sub_CE87C0(_BYTE *a1)
{
  unsigned int v1; // r12d
  _QWORD v3[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v4[4]; // [rsp+10h] [rbp-20h] BYREF

  v3[0] = v4;
  v4[0] = 0x65636166727573LL;
  v3[1] = 7;
  v1 = sub_CE7F70(a1, (__int64)v3);
  if ( (_QWORD *)v3[0] != v4 )
    j_j___libc_free_0(v3[0], v4[0] + 1LL);
  return v1;
}
