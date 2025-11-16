// Function: sub_169EBA0
// Address: 0x169eba0
//
__int64 __fastcall sub_169EBA0(_QWORD *a1, unsigned int a2)
{
  __int16 *v2; // rax
  __int16 *v3; // r12
  __int64 v4; // r15
  __int64 v5; // rsi
  __int64 i; // r13
  __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v9; // rbx
  unsigned int v11; // [rsp+4h] [rbp-7Ch]
  __int64 v12; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-68h]
  __int64 v14; // [rsp+20h] [rbp-60h] BYREF
  __int64 v15; // [rsp+28h] [rbp-58h]
  __int64 v16[9]; // [rsp+38h] [rbp-48h] BYREF

  sub_169D930((__int64)&v14, (__int64)a1);
  v2 = (__int16 *)sub_16982C0();
  v3 = v2;
  if ( v2 == word_42AE980 )
    sub_169D060(v16, (__int64)v2, &v14);
  else
    sub_169D050((__int64)v16, word_42AE980, &v14);
  if ( (unsigned int)v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  if ( (__int16 *)v16[0] == v3 )
    v11 = sub_169EBA0(v16, a2);
  else
    v11 = sub_169D440((__int64)v16, a2);
  if ( (__int16 *)v16[0] == v3 )
    sub_169D930((__int64)&v12, (__int64)v16);
  else
    sub_169D7E0((__int64)&v12, v16);
  sub_169D060(&v14, (__int64)&unk_42AE990, &v12);
  v4 = a1[1];
  if ( v4 )
  {
    v5 = 32LL * *(_QWORD *)(v4 - 8);
    for ( i = v4 + v5; v4 != i; sub_169DEB0((__int64 *)(i + 16)) )
    {
      while ( 1 )
      {
        i -= 32;
        if ( v3 == *(__int16 **)(i + 8) )
          break;
        sub_1698460(i + 8);
        if ( v4 == i )
          goto LABEL_12;
      }
    }
LABEL_12:
    j_j_j___libc_free_0_0(v4 - 8);
  }
  sub_169C7E0(a1, &v14);
  v7 = v15;
  if ( v15 )
  {
    v8 = 32LL * *(_QWORD *)(v15 - 8);
    v9 = v15 + v8;
    if ( v15 != v15 + v8 )
    {
      do
      {
        while ( 1 )
        {
          v9 -= 32;
          if ( v3 == *(__int16 **)(v9 + 8) )
            break;
          sub_1698460(v9 + 8);
          if ( v7 == v9 )
            goto LABEL_15;
        }
        sub_169DEB0((__int64 *)(v9 + 16));
      }
      while ( v7 != v9 );
    }
LABEL_15:
    j_j_j___libc_free_0_0(v7 - 8);
  }
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  sub_127D120(v16);
  return v11;
}
