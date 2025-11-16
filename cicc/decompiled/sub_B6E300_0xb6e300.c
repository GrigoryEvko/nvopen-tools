// Function: sub_B6E300
// Address: 0xb6e300
//
__int64 *__fastcall sub_B6E300(__int64 a1)
{
  __int64 *v1; // rsi
  unsigned int v3; // r14d
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  const void *v7; // r15
  __int64 *v8; // r8
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rcx
  int v12; // eax
  __int64 v13; // [rsp+8h] [rbp-E8h]
  __m128i s2; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v15; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v16[2]; // [rsp+40h] [rbp-B0h] BYREF
  _QWORD v17[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int64 *v18; // [rsp+60h] [rbp-90h] BYREF
  __int64 v19; // [rsp+68h] [rbp-88h]
  __int16 v20; // [rsp+80h] [rbp-70h]
  _BYTE *v21; // [rsp+90h] [rbp-60h] BYREF
  __int64 v22; // [rsp+98h] [rbp-58h]
  _BYTE v23[80]; // [rsp+A0h] [rbp-50h] BYREF

  v1 = (__int64 *)&v21;
  v21 = v23;
  v22 = 0x400000000LL;
  if ( !(unsigned __int8)sub_B6E2F0(a1, (__int64)&v21) )
  {
    LOBYTE(v19) = 0;
    goto LABEL_3;
  }
  v3 = *(_DWORD *)(a1 + 36);
  v4 = sub_BD5D20(a1);
  v6 = v5;
  v1 = (__int64 *)v3;
  v7 = (const void *)v4;
  sub_B6E0E0(&s2, v3, (__int64)v21, (unsigned int)v22, *(__int64 **)(a1 + 40), *(_QWORD *)(a1 + 24));
  v8 = (__int64 *)s2.m128i_i64[0];
  if ( v6 != s2.m128i_i64[1]
    || s2.m128i_i64[1]
    && (v1 = (__int64 *)s2.m128i_i64[0],
        v13 = s2.m128i_i64[0],
        v12 = memcmp(v7, (const void *)s2.m128i_i64[0], s2.m128i_u64[1]),
        v8 = (__int64 *)v13,
        v12) )
  {
    v1 = v8;
    v9 = sub_BA8B30(*(_QWORD *)(a1 + 40), v8);
    v10 = v9;
    if ( v9 )
    {
      if ( !*(_BYTE *)v9 && *(_QWORD *)(v9 + 24) == *(_QWORD *)(a1 + 24) )
        goto LABEL_14;
      v16[0] = (__int64)v17;
      sub_B5E3D0(v16, s2.m128i_i64[0], s2.m128i_i64[0] + s2.m128i_i64[1]);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v16[1]) <= 7 )
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490(v16, ".renamed", 8, v11);
      v20 = 260;
      v18 = v16;
      sub_BD6B50(v10, &v18);
      if ( (_QWORD *)v16[0] != v17 )
        j_j___libc_free_0(v16[0], v17[0] + 1LL);
    }
    v1 = (__int64 *)v3;
    v10 = sub_B6E160(*(__int64 **)(a1 + 40), v3, (__int64)v21, (unsigned int)v22);
LABEL_14:
    *(_WORD *)(v10 + 2) = *(_WORD *)(v10 + 2) & 0xC00F | *(_WORD *)(a1 + 2) & 0x3FF0;
    v8 = (__int64 *)s2.m128i_i64[0];
    v18 = (__int64 *)v10;
    LOBYTE(v19) = 1;
    goto LABEL_18;
  }
  LOBYTE(v19) = 0;
LABEL_18:
  if ( v8 != &v15 )
  {
    v1 = (__int64 *)(v15 + 1);
    j_j___libc_free_0(v8, v15 + 1);
  }
LABEL_3:
  if ( v21 != v23 )
    _libc_free(v21, v1);
  return v18;
}
