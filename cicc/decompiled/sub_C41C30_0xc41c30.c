// Function: sub_C41C30
// Address: 0xc41c30
//
float __fastcall sub_C41C30(__int64 *a1, __m128i a2)
{
  _DWORD *v2; // rbx
  _DWORD *v3; // r12
  __int64 *v4; // rdi
  float v5; // r13d
  void **v7; // rbx
  bool v8; // [rsp+Fh] [rbp-41h] BYREF
  _DWORD *v9; // [rsp+10h] [rbp-40h] BYREF
  __int64 *v10; // [rsp+18h] [rbp-38h]

  v2 = (_DWORD *)*a1;
  v3 = sub_C33340();
  if ( v2 == dword_3F657C0 )
  {
    if ( v3 == v2 )
      a1 = (__int64 *)a1[1];
    return sub_C3AAA0(a1);
  }
  else
  {
    if ( v2 == v3 )
      sub_C3C790(&v9, (_QWORD **)a1);
    else
      sub_C33EB0(&v9, a1);
    sub_C41640((__int64 *)&v9, dword_3F657C0, 1, &v8);
    v4 = (__int64 *)&v9;
    if ( v9 == v3 )
      v4 = v10;
    *(float *)a2.m128i_i32 = sub_C3AAA0(v4);
    v5 = COERCE_FLOAT(_mm_cvtsi128_si32(a2));
    if ( v9 == v3 )
    {
      if ( v10 )
      {
        v7 = (void **)&v10[3 * *(v10 - 1)];
        while ( v10 != (__int64 *)v7 )
        {
          v7 -= 3;
          if ( v3 == *v7 )
            sub_969EE0((__int64)v7);
          else
            sub_C338F0((__int64)v7);
        }
        j_j_j___libc_free_0_0(v7 - 1);
      }
    }
    else
    {
      sub_C338F0((__int64)&v9);
    }
    return v5;
  }
}
