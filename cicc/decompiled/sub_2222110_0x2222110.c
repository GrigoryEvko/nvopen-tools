// Function: sub_2222110
// Address: 0x2222110
//
__int64 __fastcall sub_2222110(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // r12d
  __int64 v7[2]; // [rsp+0h] [rbp-48h] BYREF
  _BYTE v8[56]; // [rsp+10h] [rbp-38h] BYREF

  v7[0] = (__int64)v8;
  sub_221FC40(v7, a2, (__int64)&a2[a3]);
  v5 = (*(__int64 (__fastcall **)(__int64, __int64 *, __int64))(*(_QWORD *)a1 + 16LL))(a1, v7, a4);
  if ( (_BYTE *)v7[0] != v8 )
    j___libc_free_0(v7[0]);
  return v5;
}
