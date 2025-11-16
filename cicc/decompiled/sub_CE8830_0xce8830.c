// Function: sub_CE8830
// Address: 0xce8830
//
__int64 __fastcall sub_CE8830(_BYTE *a1)
{
  unsigned int v1; // r12d
  bool v3; // zf
  _QWORD v4[2]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v5[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v6[2]; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v7[6]; // [rsp+30h] [rbp-30h] BYREF

  v4[0] = v5;
  v5[0] = 0x72656C706D6173LL;
  v4[1] = 7;
  v1 = sub_CE7F70(a1, (__int64)v4);
  if ( !(_BYTE)v1 )
  {
    v3 = *a1 == 22;
    v7[0] = 0x72656C706D6173LL;
    v6[0] = v7;
    v6[1] = 7;
    if ( v3 )
    {
      v1 = sub_CE7A30((__int64)a1, (__int64)v6);
      if ( (_QWORD *)v6[0] != v7 )
        j_j___libc_free_0(v6[0], v7[0] + 1LL);
    }
  }
  if ( (_QWORD *)v4[0] != v5 )
    j_j___libc_free_0(v4[0], v5[0] + 1LL);
  return v1;
}
