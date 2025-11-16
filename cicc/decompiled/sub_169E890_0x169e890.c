// Function: sub_169E890
// Address: 0x169e890
//
__int64 __fastcall sub_169E890(_QWORD *a1, __int64 a2, float a3)
{
  __int16 *v3; // rax
  __int16 *v4; // rbx
  _BYTE *v5; // rax
  __int64 v6; // r12
  __int64 v7; // rsi
  __int64 i; // r15
  __int64 v9; // r13
  __int64 v10; // rsi
  __int64 v11; // r12
  unsigned int v13; // [rsp+4h] [rbp-8Ch]
  __int64 v14; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v15; // [rsp+18h] [rbp-78h]
  __int16 *v16[3]; // [rsp+28h] [rbp-68h] BYREF
  __int64 v17; // [rsp+40h] [rbp-50h] BYREF
  _QWORD v18[9]; // [rsp+48h] [rbp-48h] BYREF

  sub_169D930((__int64)&v17, (__int64)a1);
  v3 = (__int16 *)sub_16982C0();
  v4 = v3;
  if ( v3 == word_42AE980 )
  {
    sub_169D060(v16, (__int64)v3, &v17);
    if ( LODWORD(v18[0]) > 0x40 && v17 )
      j_j___libc_free_0_0(v17);
    sub_169D930((__int64)&v14, a2);
    sub_169D060(v18, (__int64)word_42AE980, &v14);
  }
  else
  {
    sub_169D050((__int64)v16, word_42AE980, &v17);
    if ( LODWORD(v18[0]) > 0x40 && v17 )
      j_j___libc_free_0_0(v17);
    sub_169D930((__int64)&v14, a2);
    sub_169D050((__int64)v18, word_42AE980, &v14);
  }
  if ( v16[0] == v4 )
  {
    v13 = sub_169E890(v16, v18);
    goto LABEL_8;
  }
  v5 = (_BYTE *)sub_16D40F0(qword_4FBB490);
  if ( v5 )
  {
    if ( !*v5 )
    {
LABEL_7:
      v13 = sub_169DA10(v16, (__int64)v18);
      goto LABEL_8;
    }
  }
  else if ( !LOBYTE(qword_4FBB490[2]) )
  {
    goto LABEL_7;
  }
  if ( v4 == v16[0] )
    sub_169CAA0((__int64)v16, 0, 0, 0, a3);
  else
    sub_16986F0(v16, 0, 0, 0);
  v13 = 1;
LABEL_8:
  sub_127D120(v18);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  if ( v16[0] == v4 )
    sub_169D930((__int64)&v14, (__int64)v16);
  else
    sub_169D7E0((__int64)&v14, (__int64 *)v16);
  sub_169D060(&v17, (__int64)&unk_42AE990, &v14);
  v6 = a1[1];
  if ( v6 )
  {
    v7 = 32LL * *(_QWORD *)(v6 - 8);
    for ( i = v6 + v7; v6 != i; sub_169DEB0((__int64 *)(i + 16)) )
    {
      while ( 1 )
      {
        i -= 32;
        if ( v4 == *(__int16 **)(i + 8) )
          break;
        sub_1698460(i + 8);
        if ( v6 == i )
          goto LABEL_19;
      }
    }
LABEL_19:
    j_j_j___libc_free_0_0(v6 - 8);
  }
  sub_169C7E0(a1, &v17);
  v9 = v18[0];
  if ( v18[0] )
  {
    v10 = 32LL * *(_QWORD *)(v18[0] - 8LL);
    v11 = v18[0] + v10;
    if ( v18[0] != v18[0] + v10 )
    {
      do
      {
        while ( 1 )
        {
          v11 -= 32;
          if ( v4 == *(__int16 **)(v11 + 8) )
            break;
          sub_1698460(v11 + 8);
          if ( v9 == v11 )
            goto LABEL_26;
        }
        sub_169DEB0((__int64 *)(v11 + 16));
      }
      while ( v9 != v11 );
    }
LABEL_26:
    j_j_j___libc_free_0_0(v9 - 8);
  }
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  sub_127D120(v16);
  return v13;
}
