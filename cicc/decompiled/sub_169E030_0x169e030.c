// Function: sub_169E030
// Address: 0x169e030
//
__int64 __fastcall sub_169E030(
        __int64 a1,
        void *a2,
        __int64 a3,
        unsigned int a4,
        unsigned __int8 a5,
        unsigned int a6,
        _BYTE *a7)
{
  __int16 *v9; // rax
  __int16 *v10; // rbx
  unsigned int v11; // r13d
  __int64 v13; // r14
  __int64 v14; // rsi
  __int64 v15; // r12
  __int64 v18; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v19; // [rsp+28h] [rbp-58h]
  __int16 *v20; // [rsp+38h] [rbp-48h] BYREF
  __int64 v21; // [rsp+40h] [rbp-40h]

  sub_169D930((__int64)&v18, a1);
  v9 = (__int16 *)sub_16982C0();
  v10 = v9;
  if ( v9 == word_42AE980 )
    sub_169D060(&v20, (__int64)v9, &v18);
  else
    sub_169D050((__int64)&v20, word_42AE980, &v18);
  if ( v20 == v10 )
    v11 = sub_169E030((unsigned int)&v20, (_DWORD)a2, a3, a4, a5, a6, (__int64)a7);
  else
    v11 = sub_169A0A0((__int64)&v20, a2, a3, a4, a5, a6, a7);
  if ( v20 == v10 )
  {
    v13 = v21;
    if ( v21 )
    {
      v14 = 32LL * *(_QWORD *)(v21 - 8);
      v15 = v21 + v14;
      if ( v21 != v21 + v14 )
      {
        do
        {
          v15 -= 32;
          if ( v10 == *(__int16 **)(v15 + 8) )
            sub_169DEB0((__int64 *)(v15 + 16));
          else
            sub_1698460(v15 + 8);
        }
        while ( v13 != v15 );
      }
      j_j_j___libc_free_0_0(v13 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v20);
  }
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  return v11;
}
