// Function: sub_A72E40
// Address: 0xa72e40
//
__int64 __fastcall sub_A72E40(__int64 *a1, _QWORD *a2)
{
  __int64 v3; // rdi
  unsigned int v4; // r12d
  _BYTE v6[8]; // [rsp+8h] [rbp-B8h] BYREF
  _QWORD v7[2]; // [rsp+10h] [rbp-B0h] BYREF
  _BYTE v8[160]; // [rsp+20h] [rbp-A0h] BYREF

  v4 = (unsigned int)a2;
  v3 = *a1;
  v7[1] = 0x2000000000LL;
  v7[0] = v8;
  sub_A72AD0(v3, (__int64)v7);
  LOBYTE(v4) = *a1 == sub_C65B40(*a2 + 400LL, v7, v6, off_49D9AB0);
  if ( (_BYTE *)v7[0] != v8 )
    _libc_free(v7[0], v7);
  return v4;
}
