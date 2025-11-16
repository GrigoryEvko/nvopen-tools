// Function: sub_1212490
// Address: 0x1212490
//
__int64 __fastcall sub_1212490(__int64 *a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  __int64 v3; // rdi
  __int32 v4; // edx
  __int64 v5; // r12
  __int64 v6; // rcx
  __m128i *v7; // rax
  __int64 v8; // rax
  _QWORD v9[2]; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v10; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v11[2]; // [rsp+20h] [rbp-80h] BYREF
  __m128i v12; // [rsp+30h] [rbp-70h] BYREF
  __int16 v13; // [rsp+40h] [rbp-60h]
  _QWORD v14[4]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v15; // [rsp+70h] [rbp-30h]

  if ( a1[7] )
  {
    v5 = *a1;
    sub_8FD6D0((__int64)v9, "use of undefined value '%", (_QWORD *)(a1[5] + 32));
    if ( v9[1] == 0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    v7 = (__m128i *)sub_2241490(v9, "'", 1, v6);
    v11[0] = &v12;
    if ( (__m128i *)v7->m128i_i64[0] == &v7[1] )
    {
      v12 = _mm_loadu_si128(v7 + 1);
    }
    else
    {
      v11[0] = v7->m128i_i64[0];
      v12.m128i_i64[0] = v7[1].m128i_i64[0];
    }
    v11[1] = v7->m128i_i64[1];
    v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
    v7->m128i_i64[1] = 0;
    v7[1].m128i_i8[0] = 0;
    v14[0] = v11;
    v8 = a1[5];
    v15 = 260;
    sub_11FD800(v5 + 176, *(_QWORD *)(v8 + 72), (__int64)v14, 1);
    if ( (__m128i *)v11[0] != &v12 )
      j_j___libc_free_0(v11[0], v12.m128i_i64[0] + 1);
    if ( (__int64 *)v9[0] != &v10 )
      j_j___libc_free_0(v9[0], v10 + 1);
  }
  else
  {
    result = 0;
    if ( !a1[13] )
      return result;
    v2 = a1[11];
    v3 = *a1;
    v4 = *(_DWORD *)(v2 + 32);
    v11[0] = "use of undefined value '%";
    v14[2] = "'";
    v12.m128i_i32[0] = v4;
    v13 = 2307;
    v14[0] = v11;
    v15 = 770;
    sub_11FD800(v3 + 176, *(_QWORD *)(v2 + 48), (__int64)v14, 1);
  }
  return 1;
}
