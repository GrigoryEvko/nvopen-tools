// Function: sub_16A4A90
// Address: 0x16a4a90
//
void __fastcall sub_16A4A90(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4, unsigned __int8 a5)
{
  __int16 *v7; // rax
  __int16 *v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rsi
  __int64 v11; // r12
  __int64 v13; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-58h]
  __int16 *v15; // [rsp+28h] [rbp-48h] BYREF
  __int64 v16; // [rsp+30h] [rbp-40h]

  sub_169D930((__int64)&v13, a1);
  v7 = (__int16 *)sub_16982C0();
  v8 = v7;
  if ( v7 == word_42AE980 )
    sub_169D060(&v15, (__int64)v7, &v13);
  else
    sub_169D050((__int64)&v15, word_42AE980, &v13);
  if ( v15 == v8 )
    sub_16A4A90(&v15, a2, a3, a4, a5);
  else
    sub_16A3760((__int64)&v15, a2, a3, a4, a5);
  if ( v15 == v8 )
  {
    v9 = v16;
    if ( v16 )
    {
      v10 = 32LL * *(_QWORD *)(v16 - 8);
      v11 = v16 + v10;
      if ( v16 != v16 + v10 )
      {
        do
        {
          v11 -= 32;
          if ( v8 == *(__int16 **)(v11 + 8) )
            sub_169DEB0((__int64 *)(v11 + 16));
          else
            sub_1698460(v11 + 8);
        }
        while ( v9 != v11 );
      }
      j_j_j___libc_free_0_0(v9 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v15);
  }
  if ( v14 > 0x40 )
  {
    if ( v13 )
      j_j___libc_free_0_0(v13);
  }
}
