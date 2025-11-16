// Function: sub_39CEB00
// Address: 0x39ceb00
//
__int64 __fastcall sub_39CEB00(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 *a5,
        unsigned __int64 a6)
{
  __int64 v7; // rax
  _BYTE *v8; // rdi
  __int64 v9; // r13
  _BYTE *v10; // rsi
  unsigned __int64 v11; // rax
  _QWORD *v12; // rdx
  _BYTE *v14; // [rsp+0h] [rbp-70h] BYREF
  __int64 v15; // [rsp+8h] [rbp-68h]
  _BYTE v16[96]; // [rsp+10h] [rbp-60h] BYREF

  v14 = v16;
  v15 = 0x800000000LL;
  v7 = sub_39CD420(a1, a2, (__int64)&v14, 0, a5, a6);
  v8 = v14;
  v9 = v7;
  v10 = &v14[8 * (unsigned int)v15];
  if ( v10 != v14 )
  {
    do
    {
      v11 = *(_QWORD *)v8;
      *(_QWORD *)(*(_QWORD *)v8 + 40LL) = a3;
      v12 = *(_QWORD **)(a3 + 32);
      if ( v12 )
      {
        *(_QWORD *)v11 = *v12;
        **(_QWORD **)(a3 + 32) = v11 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v8 += 8;
      *(_QWORD *)(a3 + 32) = v11;
    }
    while ( v8 != v10 );
    v8 = v14;
  }
  if ( v8 != v16 )
    _libc_free((unsigned __int64)v8);
  return v9;
}
