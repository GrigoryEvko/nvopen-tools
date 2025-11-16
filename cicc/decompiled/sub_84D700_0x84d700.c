// Function: sub_84D700
// Address: 0x84d700
//
__int64 __fastcall sub_84D700(__int64 a1, __m128i *a2)
{
  unsigned int v4; // r12d
  char v6; // al
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 v9; // rsi
  int v10; // ebx
  _QWORD *v11; // rcx
  _BOOL4 v12; // ebx
  __m128i *v13; // rax
  __int64 v14; // rdi
  _QWORD *m128i_i64; // [rsp+8h] [rbp-B8h] BYREF
  __m128i v16; // [rsp+10h] [rbp-B0h] BYREF
  __m128i v17[3]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v18; // [rsp+50h] [rbp-70h]
  _BYTE v19[88]; // [rsp+68h] [rbp-58h] BYREF

  if ( !(unsigned int)sub_8D3A70(a2) )
  {
    v4 = 0;
    sub_838020(a1, 0, a2, 0, word_4D04898, 0, v17);
    if ( v17[0].m128i_i32[2] == 7 )
      return v4;
    v6 = *(_BYTE *)(a1 + 16);
    m128i_i64 = 0;
    v7 = *(_QWORD *)a1;
    v8 = v18;
    if ( word_4D04898 && v18 )
    {
      m128i_i64 = sub_724DC0();
      v4 = sub_6E47F0(v8, a1, (__int64)a2, (__int64)m128i_i64);
      if ( !v4 )
      {
LABEL_17:
        sub_724E30((__int64)&m128i_i64);
        return v4;
      }
      v11 = m128i_i64;
      v12 = 1;
      v7 = m128i_i64[16];
    }
    else
    {
      if ( v6 == 2 )
      {
        v9 = *(_QWORD *)a1;
        m128i_i64 = (_QWORD *)(a1 + 144);
        return (unsigned int)sub_8DD690(v19, v9, 1, a1 + 144, a2, 0) != 0;
      }
      if ( !(unsigned int)sub_8D2D50(a2) || *(_BYTE *)(a1 + 17) != 1 || sub_6ED0A0(a1) || *(_BYTE *)(a1 + 16) != 1 )
      {
        v10 = sub_8DD690(v19, v7, 0, m128i_i64, a2, 0);
        if ( v10 )
          return 1;
        goto LABEL_23;
      }
      v16 = 0u;
      v13 = (__m128i *)sub_724DC0();
      v14 = *(_QWORD *)(a1 + 144);
      m128i_i64 = v13->m128i_i64;
      v12 = sub_7A30C0(v14, 1, 1, v13, &v16) != 0;
      sub_67E3D0(&v16);
      v11 = m128i_i64;
    }
    v4 = sub_8DD690(v19, v7, v12, v11, a2, 0);
    if ( v4 )
    {
      v4 = 1;
      goto LABEL_17;
    }
    if ( v12 )
      goto LABEL_17;
    v10 = 1;
LABEL_23:
    v4 = sub_696840(a1) != 0;
    if ( !v10 )
      return v4;
    goto LABEL_17;
  }
  v17[0].m128i_i64[0] = (__int64)sub_724DC0();
  v4 = sub_84D340(a1, a2, (__m128i *)v17[0].m128i_i64[0]);
  sub_724E30((__int64)v17);
  return v4;
}
