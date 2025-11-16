// Function: sub_6DE780
// Address: 0x6de780
//
__m128i **__fastcall sub_6DE780(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // r14d
  __int64 v5; // rdi
  int i; // eax
  unsigned int v7; // eax
  unsigned int *v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rsi
  __m128i **v11; // r12
  __int64 v13; // rax
  char v14; // r13
  bool v15; // r13
  unsigned __int16 v16; // di
  __m128i *v17; // rax
  __m128i *v18; // rdi
  __int64 v19; // [rsp+8h] [rbp-248h] BYREF
  __m128i v20; // [rsp+10h] [rbp-240h] BYREF
  _BYTE v21[19]; // [rsp+20h] [rbp-230h] BYREF
  char v22; // [rsp+33h] [rbp-21Dh]
  _BYTE v23[400]; // [rsp+C0h] [rbp-190h] BYREF

  v4 = dword_4F06650[0];
  sub_7B8B50(a1, a2, a3, a4);
  if ( (_DWORD)a1 )
  {
    v5 = *((unsigned int *)qword_4D03C00 + 2);
    for ( i = v4; ; i = v7 + 1 )
    {
      v7 = v5 & i;
      v8 = (unsigned int *)(*qword_4D03C00 + 24LL * v7);
      v9 = *v8;
      if ( v4 == (_DWORD)v9 )
        break;
      if ( !(_DWORD)v9 )
        return 0;
    }
    v10 = v8[2];
    v11 = (__m128i **)*((_QWORD *)v8 + 2);
    v20.m128i_i64[1] = (__int64)v11;
    v20.m128i_i64[0] = v10;
    if ( (unsigned int)v10 > dword_4F06650[0] )
    {
      do
      {
        if ( word_4F06418[0] == 9 )
          break;
        sub_7B8B50(v5, v10, v9, v8);
      }
      while ( v20.m128i_i32[0] > dword_4F06650[0] );
    }
  }
  else
  {
    v13 = qword_4F04C68[0] + 776LL * dword_4F04C64;
    v14 = *(_BYTE *)(v13 + 12);
    *(_BYTE *)(v13 + 12) = v14 & 0xFD;
    v15 = (v14 & 2) != 0;
    v11 = (__m128i **)sub_7272D0();
    v11[1] = *(__m128i **)&dword_4F063F8;
    sub_6E1DD0(&v19);
    sub_6E1E00(5, v21, 0, 1);
    v22 |= 2u;
    v16 = word_4F06418[0];
    if ( !sub_6878E0(word_4F06418[0]) && (!sub_687960(v16) || dword_4D04964) )
      sub_6851C0(0xBDDu, &dword_4F063F8);
    sub_69ED20((__int64)v23, 0, 3, 12288);
    v17 = (__m128i *)sub_6F6F40(v23, 0);
    *v11 = v17;
    v18 = v17;
    sub_68A310(v17);
    sub_6E2B30(v18, 0);
    sub_6E1DF0(v19);
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12)
                                                              & 0xFD
                                                              | (2 * v15);
    v20.m128i_i64[1] = (__int64)v11;
    v20.m128i_i32[0] = dword_4F06650[0];
    sub_69DF50(qword_4D03C00, v4, &v20, v4);
  }
  return v11;
}
