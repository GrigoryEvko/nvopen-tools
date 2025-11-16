// Function: sub_169F930
// Address: 0x169f930
//
__int64 __fastcall sub_169F930(_QWORD *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int16 *v5; // rax
  __int16 *v6; // rbx
  __int64 v7; // r15
  __int64 v8; // rsi
  __int64 i; // r14
  __int64 v10; // r14
  __int64 v11; // rsi
  __int64 v12; // r12
  char *v14; // rax
  char v15; // al
  unsigned int v17; // [rsp+Ch] [rbp-C4h]
  __int16 *v18; // [rsp+10h] [rbp-C0h]
  __int64 v19; // [rsp+20h] [rbp-B0h] BYREF
  unsigned int v20; // [rsp+28h] [rbp-A8h]
  __int64 v21; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v22; // [rsp+38h] [rbp-98h]
  _BYTE v23[8]; // [rsp+40h] [rbp-90h] BYREF
  __int16 *v24[3]; // [rsp+48h] [rbp-88h] BYREF
  __int64 v25; // [rsp+60h] [rbp-70h] BYREF
  _QWORD v26[3]; // [rsp+68h] [rbp-68h] BYREF
  __int64 v27; // [rsp+80h] [rbp-50h] BYREF
  _QWORD v28[9]; // [rsp+88h] [rbp-48h] BYREF

  sub_169D930((__int64)&v27, (__int64)a1);
  v5 = (__int16 *)sub_16982C0();
  v6 = v5;
  if ( v5 == word_42AE980 )
  {
    sub_169D060(v24, (__int64)v5, &v27);
    if ( LODWORD(v28[0]) > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
    sub_169D930((__int64)&v21, a3);
    sub_169D060(v28, (__int64)word_42AE980, &v21);
    sub_169D930((__int64)&v19, a2);
    sub_169D060(v26, (__int64)word_42AE980, &v19);
  }
  else
  {
    sub_169D050((__int64)v24, word_42AE980, &v27);
    if ( LODWORD(v28[0]) > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
    sub_169D930((__int64)&v21, a3);
    sub_169D050((__int64)v28, word_42AE980, &v21);
    sub_169D930((__int64)&v19, a2);
    sub_169D050((__int64)v26, word_42AE980, &v19);
  }
  v18 = v24[0];
  if ( v24[0] == v6 )
  {
    v17 = sub_169F930(v24, v26, v28, a4);
  }
  else if ( v18 == sub_1698270()
         && ((v14 = (char *)sub_16D40F0(qword_4FBB490)) == 0 ? (v15 = qword_4FBB490[2]) : (v15 = *v14), v15) )
  {
    v17 = sub_169F510((__int64)v23, (__int64)&v25, (__int64)&v27, a4);
  }
  else
  {
    v17 = sub_169DD30(v24, v26, v28, a4);
  }
  sub_127D120(v26);
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  sub_127D120(v28);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  if ( v24[0] == v6 )
    sub_169D930((__int64)&v25, (__int64)v24);
  else
    sub_169D7E0((__int64)&v25, (__int64 *)v24);
  sub_169D060(&v27, (__int64)&unk_42AE990, &v25);
  v7 = a1[1];
  if ( v7 )
  {
    v8 = 32LL * *(_QWORD *)(v7 - 8);
    for ( i = v7 + v8; v7 != i; sub_169DEB0((__int64 *)(i + 16)) )
    {
      while ( 1 )
      {
        i -= 32;
        if ( v6 == *(__int16 **)(i + 8) )
          break;
        sub_1698460(i + 8);
        if ( v7 == i )
          goto LABEL_17;
      }
    }
LABEL_17:
    j_j_j___libc_free_0_0(v7 - 8);
  }
  sub_169C7E0(a1, &v27);
  v10 = v28[0];
  if ( v28[0] )
  {
    v11 = 32LL * *(_QWORD *)(v28[0] - 8LL);
    v12 = v28[0] + v11;
    if ( v28[0] != v28[0] + v11 )
    {
      do
      {
        while ( 1 )
        {
          v12 -= 32;
          if ( v6 == *(__int16 **)(v12 + 8) )
            break;
          sub_1698460(v12 + 8);
          if ( v10 == v12 )
            goto LABEL_20;
        }
        sub_169DEB0((__int64 *)(v12 + 16));
      }
      while ( v10 != v12 );
    }
LABEL_20:
    j_j_j___libc_free_0_0(v10 - 8);
  }
  if ( LODWORD(v26[0]) > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  sub_127D120(v24);
  return v17;
}
