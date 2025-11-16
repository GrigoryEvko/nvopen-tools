// Function: sub_169E440
// Address: 0x169e440
//
__int64 __fastcall sub_169E440(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int16 *v6; // rax
  __int16 *v7; // rbx
  __int64 v8; // r14
  __int64 v9; // rsi
  __int64 i; // r12
  __int64 v11; // r15
  __int64 v12; // rsi
  __int64 v13; // r12
  unsigned int v15; // [rsp+4h] [rbp-7Ch]
  __int64 v16; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-68h]
  __int64 v18; // [rsp+20h] [rbp-60h] BYREF
  __int64 v19; // [rsp+28h] [rbp-58h]
  _BYTE v20[8]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v21[9]; // [rsp+38h] [rbp-48h] BYREF

  v6 = (__int16 *)sub_16982C0();
  v7 = v6;
  if ( v6 == word_42AE980 )
    sub_169C4E0(v21, (__int64)v6);
  else
    sub_1698360((__int64)v21, (__int64)word_42AE980);
  v15 = sub_169E610(v20, a2, a3, a4);
  if ( (__int16 *)v21[0] == v7 )
    sub_169D930((__int64)&v16, (__int64)v21);
  else
    sub_169D7E0((__int64)&v16, v21);
  sub_169D060(&v18, (__int64)&unk_42AE990, &v16);
  v8 = a1[1];
  if ( v8 )
  {
    v9 = 32LL * *(_QWORD *)(v8 - 8);
    for ( i = v8 + v9; v8 != i; sub_169DEB0((__int64 *)(i + 16)) )
    {
      while ( 1 )
      {
        i -= 32;
        if ( v7 == *(__int16 **)(i + 8) )
          break;
        sub_1698460(i + 8);
        if ( v8 == i )
          goto LABEL_7;
      }
    }
LABEL_7:
    j_j_j___libc_free_0_0(v8 - 8);
  }
  sub_169C7E0(a1, &v18);
  v11 = v19;
  if ( v19 )
  {
    v12 = 32LL * *(_QWORD *)(v19 - 8);
    v13 = v19 + v12;
    if ( v19 != v19 + v12 )
    {
      do
      {
        while ( 1 )
        {
          v13 -= 32;
          if ( v7 == *(__int16 **)(v13 + 8) )
            break;
          sub_1698460(v13 + 8);
          if ( v11 == v13 )
            goto LABEL_10;
        }
        sub_169DEB0((__int64 *)(v13 + 16));
      }
      while ( v11 != v13 );
    }
LABEL_10:
    j_j_j___libc_free_0_0(v11 - 8);
  }
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  sub_127D120(v21);
  return v15;
}
