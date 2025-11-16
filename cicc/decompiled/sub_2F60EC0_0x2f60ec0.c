// Function: sub_2F60EC0
// Address: 0x2f60ec0
//
void __fastcall sub_2F60EC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdi
  unsigned __int64 v7[2]; // [rsp+0h] [rbp-70h] BYREF
  _BYTE v8[96]; // [rsp+10h] [rbp-60h] BYREF

  if ( (unsigned __int8)sub_2E168A0(*(_QWORD **)(a1 + 40), a2, a3, a4, a5, a6) )
  {
    v6 = *(_QWORD *)(a1 + 40);
    v7[0] = (unsigned __int64)v8;
    v7[1] = 0x800000000LL;
    sub_2E15100(v6, a2, (__int64)v7);
    if ( (_BYTE *)v7[0] != v8 )
      _libc_free(v7[0]);
  }
}
