// Function: sub_2BD7120
// Address: 0x2bd7120
//
__int64 __fastcall sub_2BD7120(
        __int64 a1,
        unsigned __int8 *a2,
        unsigned __int8 *a3,
        __int64 a4,
        __int64 a5,
        __m128i a6)
{
  int v7; // r13d
  int v8; // eax
  __int64 v9; // r14
  unsigned int v10; // r13d
  _QWORD *v11; // r12
  __int64 v12; // rax
  _BYTE *v14; // [rsp+0h] [rbp-60h] BYREF
  __int64 v15; // [rsp+8h] [rbp-58h]
  _BYTE v16[80]; // [rsp+10h] [rbp-50h] BYREF

  v14 = v16;
  v15 = 0x200000000LL;
  v7 = sub_2BD62F0(a1, a2, a3, a4, a5, (__int64)&v14, a6);
  v8 = sub_2BCF820(a1, (__int64)v14, (unsigned int)v15, a5, a6);
  v9 = (__int64)v14;
  v10 = v8 | v7;
  v11 = &v14[24 * (unsigned int)v15];
  if ( v14 != (_BYTE *)v11 )
  {
    do
    {
      v12 = *(v11 - 1);
      v11 -= 3;
      if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
        sub_BD60C0(v11);
    }
    while ( (_QWORD *)v9 != v11 );
    v11 = v14;
  }
  if ( v11 != (_QWORD *)v16 )
    _libc_free((unsigned __int64)v11);
  return v10;
}
