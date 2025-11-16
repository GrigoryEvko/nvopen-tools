// Function: sub_169EEB0
// Address: 0x169eeb0
//
__int64 __fastcall sub_169EEB0(_QWORD *a1, __int64 a2, unsigned int a3, float a4)
{
  __int16 *v4; // rax
  __int16 *v5; // rbx
  __int64 v6; // r12
  __int64 v7; // rsi
  __int64 i; // r14
  __int64 v9; // r13
  __int64 v10; // rsi
  __int64 v11; // r12
  char *v13; // rax
  char v14; // al
  unsigned int v16; // [rsp+14h] [rbp-8Ch]
  __int64 v17; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v18; // [rsp+28h] [rbp-78h]
  _BYTE v19[8]; // [rsp+30h] [rbp-70h] BYREF
  __int16 *v20[3]; // [rsp+38h] [rbp-68h] BYREF
  __int64 v21; // [rsp+50h] [rbp-50h] BYREF
  _QWORD v22[9]; // [rsp+58h] [rbp-48h] BYREF

  sub_169D930((__int64)&v21, (__int64)a1);
  v4 = (__int16 *)sub_16982C0();
  v5 = v4;
  if ( v4 == word_42AE980 )
  {
    sub_169D060(v20, (__int64)v4, &v21);
    if ( LODWORD(v22[0]) > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
    sub_169D930((__int64)&v17, a2);
    sub_169D060(v22, (__int64)word_42AE980, &v17);
    if ( v20[0] != v5 )
      goto LABEL_4;
LABEL_41:
    v16 = sub_169EEB0(v20, v22, a3);
    goto LABEL_8;
  }
  sub_169D050((__int64)v20, word_42AE980, &v21);
  if ( LODWORD(v22[0]) > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  sub_169D930((__int64)&v17, a2);
  sub_169D050((__int64)v22, word_42AE980, &v17);
  if ( v20[0] == v5 )
    goto LABEL_41;
LABEL_4:
  if ( (unsigned __int8)sub_169DE70((__int64)v19) || (unsigned __int8)sub_169DE70((__int64)&v21) )
  {
    if ( v5 == v20[0] )
      sub_169CAA0((__int64)v20, 0, 0, 0, a4);
    else
      sub_16986F0(v20, 0, 0, 0);
    v16 = 1;
  }
  else if ( v20[0] == sub_1698270()
         && ((v13 = (char *)sub_16D40F0(qword_4FBB490)) == 0 ? (v14 = qword_4FBB490[2]) : (v14 = *v13), v14) )
  {
    v16 = sub_1581A10((__int64)v19, (__int64)&v21, a3, a4);
  }
  else
  {
    v16 = sub_16994B0(v20, (__int64)v22, a3);
  }
LABEL_8:
  sub_127D120(v22);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  if ( v20[0] == v5 )
    sub_169D930((__int64)&v17, (__int64)v20);
  else
    sub_169D7E0((__int64)&v17, (__int64 *)v20);
  sub_169D060(&v21, (__int64)&unk_42AE990, &v17);
  v6 = a1[1];
  if ( v6 )
  {
    v7 = 32LL * *(_QWORD *)(v6 - 8);
    for ( i = v6 + v7; v6 != i; sub_169DEB0((__int64 *)(i + 16)) )
    {
      while ( 1 )
      {
        i -= 32;
        if ( v5 == *(__int16 **)(i + 8) )
          break;
        sub_1698460(i + 8);
        if ( v6 == i )
          goto LABEL_19;
      }
    }
LABEL_19:
    j_j_j___libc_free_0_0(v6 - 8);
  }
  sub_169C7E0(a1, &v21);
  v9 = v22[0];
  if ( v22[0] )
  {
    v10 = 32LL * *(_QWORD *)(v22[0] - 8LL);
    v11 = v22[0] + v10;
    if ( v22[0] != v22[0] + v10 )
    {
      do
      {
        while ( 1 )
        {
          v11 -= 32;
          if ( v5 == *(__int16 **)(v11 + 8) )
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
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  sub_127D120(v20);
  return v16;
}
