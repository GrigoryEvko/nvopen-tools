// Function: sub_14EE150
// Address: 0x14ee150
//
__m128i *__fastcall sub_14EE150(__m128i *a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  const char *v4; // rax
  __int64 v5; // rax
  __m128i *v6; // rdi
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // [rsp+8h] [rbp-368h] BYREF
  __int64 v12; // [rsp+10h] [rbp-360h]
  __m128i v13; // [rsp+30h] [rbp-340h] BYREF
  __int16 v14; // [rsp+40h] [rbp-330h]
  __m128i v15; // [rsp+50h] [rbp-320h] BYREF
  char v16; // [rsp+60h] [rbp-310h]
  char v17; // [rsp+61h] [rbp-30Fh]
  __m128i v18[2]; // [rsp+70h] [rbp-300h] BYREF
  __m128i v19; // [rsp+90h] [rbp-2E0h] BYREF
  __int16 v20; // [rsp+A0h] [rbp-2D0h]
  __m128i v21[2]; // [rsp+B0h] [rbp-2C0h] BYREF
  __m128i v22; // [rsp+D0h] [rbp-2A0h] BYREF
  char v23; // [rsp+E0h] [rbp-290h]
  char v24; // [rsp+E1h] [rbp-28Fh]
  __m128i v25; // [rsp+F0h] [rbp-280h] BYREF
  char v26; // [rsp+100h] [rbp-270h]
  char v27; // [rsp+101h] [rbp-26Fh]
  __m128i *v28; // [rsp+110h] [rbp-260h] BYREF
  __int64 v29; // [rsp+118h] [rbp-258h]
  __m128i v30; // [rsp+120h] [rbp-250h] BYREF
  __int64 *v31; // [rsp+130h] [rbp-240h] BYREF
  __int64 v32; // [rsp+138h] [rbp-238h]
  _BYTE v33[560]; // [rsp+140h] [rbp-230h] BYREF

  if ( (unsigned __int8)sub_15127D0(a2, 13, 0) )
  {
    v33[1] = 1;
    v31 = (__int64 *)"Invalid record";
    v33[0] = 3;
    sub_14EE0F0((__int64 *)&v28, (__int64)&v31);
    v9 = (__int64)v28;
    a1[2].m128i_i8[0] |= 3u;
    a1->m128i_i64[0] = v9 & 0xFFFFFFFFFFFFFFFELL;
    return a1;
  }
  v29 = 0;
  v31 = (__int64 *)v33;
  v32 = 0x4000000000LL;
  v28 = &v30;
  v30.m128i_i8[0] = 0;
  do
  {
    while ( 1 )
    {
      v3 = sub_14ECC00(a2, 0);
      if ( (_DWORD)v3 == 1 )
      {
        a1[2].m128i_i8[0] = a1[2].m128i_i8[0] & 0xFC | 2;
        a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
        if ( v28 == &v30 )
        {
          a1[1] = _mm_load_si128(&v30);
        }
        else
        {
          a1->m128i_i64[0] = (__int64)v28;
          a1[1].m128i_i64[0] = v30.m128i_i64[0];
        }
        a1->m128i_i64[1] = v29;
        goto LABEL_9;
      }
      if ( (_DWORD)v3 != 3 )
      {
        v27 = 1;
        v4 = "Malformed block";
LABEL_6:
        v25.m128i_i64[0] = (__int64)v4;
        v26 = 3;
        sub_14EE0F0(v22.m128i_i64, (__int64)&v25);
        v5 = v22.m128i_i64[0];
        a1[2].m128i_i8[0] |= 3u;
        v6 = v28;
        a1->m128i_i64[0] = v5 & 0xFFFFFFFFFFFFFFFELL;
        goto LABEL_7;
      }
      LODWORD(v32) = 0;
      v8 = sub_1510D70(a2, HIDWORD(v3), &v31, 0);
      if ( v8 != 1 )
        break;
      sub_14EA4D0(v31, v32, &v28);
    }
    if ( v8 != 2 )
    {
      v27 = 1;
      v4 = "Invalid value";
      goto LABEL_6;
    }
  }
  while ( !(unsigned int)*v31 );
  LODWORD(v12) = *v31;
  v22.m128i_i64[0] = (__int64)"'";
  v13.m128i_i64[0] = (__int64)"Incompatible epoch: Bitcode '";
  v20 = 266;
  v15.m128i_i64[0] = (__int64)"' vs current: '";
  v13.m128i_i64[1] = v12;
  v14 = 2307;
  v24 = 1;
  v23 = 3;
  v19.m128i_i64[0] = 0;
  v17 = 1;
  v16 = 3;
  sub_14EC200(v18, &v13, &v15);
  sub_14EC200(v21, v18, &v19);
  sub_14EC200(&v25, v21, &v22);
  sub_14EE0F0(&v11, (__int64)&v25);
  v10 = v11;
  a1[2].m128i_i8[0] |= 3u;
  v6 = v28;
  a1->m128i_i64[0] = v10 & 0xFFFFFFFFFFFFFFFELL;
LABEL_7:
  if ( v6 != &v30 )
    j_j___libc_free_0(v6, v30.m128i_i64[0] + 1);
LABEL_9:
  if ( v31 != (__int64 *)v33 )
    _libc_free((unsigned __int64)v31);
  return a1;
}
