// Function: sub_137E510
// Address: 0x137e510
//
__int64 __fastcall sub_137E510(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  unsigned __int64 v6[2]; // [rsp+0h] [rbp-120h] BYREF
  _QWORD v7[34]; // [rsp+10h] [rbp-110h] BYREF

  v6[0] = (unsigned __int64)v7;
  v7[0] = a1;
  v6[1] = 0x2000000001LL;
  v4 = sub_137E120((__int64)v6, a2, a3, a4);
  if ( (_QWORD *)v6[0] != v7 )
    _libc_free(v6[0]);
  return v4;
}
