// Function: sub_2C2C3F0
// Address: 0x2c2c3f0
//
__int64 __fastcall sub_2C2C3F0(const __m128i *a1, __int64 a2)
{
  char v2; // al
  __int64 result; // rax
  unsigned __int8 v4; // al
  unsigned __int8 v5; // al
  bool v6; // zf
  unsigned __int8 v7; // [rsp+8h] [rbp-98h]
  unsigned __int8 v8; // [rsp+8h] [rbp-98h]
  unsigned __int8 v9; // [rsp+8h] [rbp-98h]
  __m128i v10; // [rsp+10h] [rbp-90h] BYREF
  __int64 v11[2]; // [rsp+20h] [rbp-80h] BYREF
  __int64 v12; // [rsp+30h] [rbp-70h]
  __int64 v13; // [rsp+38h] [rbp-68h]
  __m128i v14; // [rsp+40h] [rbp-60h] BYREF
  __int64 v15[2]; // [rsp+50h] [rbp-50h] BYREF
  __int64 v16; // [rsp+60h] [rbp-40h]
  __int64 v17; // [rsp+68h] [rbp-38h]

  v2 = *(_BYTE *)(a2 + 8);
  if ( v2 == 23 )
  {
LABEL_2:
    if ( *(_DWORD *)(a2 + 160) != 29 )
      return 0;
    goto LABEL_7;
  }
  if ( v2 != 9 )
  {
    if ( v2 != 16 )
    {
      v6 = v2 == 4;
      result = 0;
      if ( !v6 || *(_BYTE *)(a2 + 160) != 29 )
        return result;
      goto LABEL_7;
    }
    goto LABEL_2;
  }
  if ( **(_BYTE **)(a2 + 136) != 58 )
    return 0;
LABEL_7:
  v10 = _mm_loadu_si128(a1 + 3);
  sub_9865C0((__int64)v11, (__int64)a1[4].m128i_i64);
  v12 = a1[5].m128i_i64[0];
  v13 = a1[5].m128i_i64[1];
  if ( (unsigned __int8)sub_2C2C0F0(&v10, **(_QWORD **)(a2 + 48)) )
  {
    v14 = _mm_loadu_si128(a1);
    sub_9865C0((__int64)v15, (__int64)a1[1].m128i_i64);
    v16 = a1[2].m128i_i64[0];
    v17 = a1[2].m128i_i64[1];
    v5 = sub_2C2C240((__int64)&v14, *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
    if ( v5 )
    {
      v9 = v5;
      sub_969240(v15);
      sub_969240(v11);
      return v9;
    }
    sub_969240(v15);
    sub_969240(v11);
  }
  else
  {
    sub_969240(v11);
  }
  v10 = _mm_loadu_si128(a1 + 3);
  sub_9865C0((__int64)v11, (__int64)a1[4].m128i_i64);
  v12 = a1[5].m128i_i64[0];
  v13 = a1[5].m128i_i64[1];
  v4 = sub_2C2C0F0(&v10, *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL * (unsigned int)(*(_DWORD *)(a2 + 56) - 1)));
  if ( v4 )
  {
    v14 = _mm_loadu_si128(a1);
    sub_9865C0((__int64)v15, (__int64)a1[1].m128i_i64);
    v16 = a1[2].m128i_i64[0];
    v17 = a1[2].m128i_i64[1];
    v7 = sub_2C2C240((__int64)&v14, *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL * (unsigned int)(*(_DWORD *)(a2 + 56) - 2)));
    sub_969240(v15);
    v4 = v7;
  }
  v8 = v4;
  sub_969240(v11);
  return v8;
}
