// Function: sub_388F860
// Address: 0x388f860
//
__int64 __fastcall sub_388F860(__int64 *a1)
{
  unsigned int v1; // r12d
  __int64 v3; // rax
  __int64 v4; // rdi
  int v5; // edx
  __int64 v6; // r12
  __m128i *v7; // rax
  __int64 v8; // rax
  const char ***v9; // [rsp+0h] [rbp-80h] BYREF
  __int16 v10; // [rsp+10h] [rbp-70h]
  const char *v11; // [rsp+20h] [rbp-60h] BYREF
  const char ***v12; // [rsp+28h] [rbp-58h]
  __int64 v13; // [rsp+30h] [rbp-50h] BYREF
  const char **v14; // [rsp+40h] [rbp-40h] BYREF
  char *v15; // [rsp+48h] [rbp-38h]
  _OWORD v16[3]; // [rsp+50h] [rbp-30h] BYREF

  if ( a1[7] )
  {
    v6 = *a1;
    sub_8FD6D0((__int64)&v11, "use of undefined value '%", (_QWORD *)(a1[5] + 32));
    if ( v12 == (const char ***)0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    v7 = (__m128i *)sub_2241490((unsigned __int64 *)&v11, "'", 1u);
    v14 = (const char **)v16;
    if ( (__m128i *)v7->m128i_i64[0] == &v7[1] )
    {
      v16[0] = _mm_loadu_si128(v7 + 1);
    }
    else
    {
      v14 = (const char **)v7->m128i_i64[0];
      *(_QWORD *)&v16[0] = v7[1].m128i_i64[0];
    }
    v15 = (char *)v7->m128i_i64[1];
    v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
    v7->m128i_i64[1] = 0;
    v7[1].m128i_i8[0] = 0;
    v9 = &v14;
    v8 = a1[5];
    v10 = 260;
    v1 = sub_38814C0(v6 + 8, *(_QWORD *)(v8 + 72), (__int64)&v9);
    if ( v14 != (const char **)v16 )
      j_j___libc_free_0((unsigned __int64)v14);
    if ( v11 == (const char *)&v13 )
      return v1;
    j_j___libc_free_0((unsigned __int64)v11);
    return v1;
  }
  else
  {
    v1 = 0;
    if ( !a1[13] )
      return v1;
    v3 = a1[11];
    v4 = *a1;
    v5 = *(_DWORD *)(v3 + 32);
    v11 = "use of undefined value '%";
    v15 = "'";
    LODWORD(v9) = v5;
    LOWORD(v16[0]) = 770;
    v12 = v9;
    LOWORD(v13) = 2307;
    v14 = &v11;
    return (unsigned int)sub_38814C0(v4 + 8, *(_QWORD *)(v3 + 48), (__int64)&v14);
  }
}
