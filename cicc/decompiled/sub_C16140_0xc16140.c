// Function: sub_C16140
// Address: 0xc16140
//
__int64 __fastcall sub_C16140(__int128 a1, __int64 a2, __int64 a3)
{
  char *v4; // r14
  const char *v5; // r12
  size_t v6; // rdx
  size_t v7; // r15
  unsigned __int64 v8; // rcx
  __int64 v9; // rax
  __int128 v10; // [rsp+0h] [rbp-70h] BYREF
  __m128i si128; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v12[3]; // [rsp+20h] [rbp-50h] BYREF
  char v13; // [rsp+38h] [rbp-38h] BYREF

  v12[1] = ".part.";
  v10 = a1;
  v12[0] = ".llvm.";
  v12[2] = ".__uniq.";
  if ( !a3 )
    goto LABEL_7;
  if ( a3 == 3 )
  {
    if ( *(_WORD *)a2 != 27745 || *(_BYTE *)(a2 + 2) != 108 )
      return v10;
LABEL_7:
    si128.m128i_i8[0] = 46;
    sub_C931B0(&v10, &si128, 1, 0);
    return v10;
  }
  if ( a3 != 8 || *(_QWORD *)a2 != 0x64657463656C6573LL )
    return v10;
  v4 = (char *)v12;
  v5 = ".llvm.";
  si128 = _mm_load_si128((const __m128i *)&v10);
  while ( 1 )
  {
    v6 = 0;
    v7 = 0;
    if ( !v5 || (v7 = strlen(v5), v6 = v7, v7 != 8) || *(_QWORD *)v5 != 0x2E71696E755F5F2ELL || !unk_4C5C708 )
    {
      v8 = sub_C93460(&si128, v5, v6);
      if ( v8 != -1 )
      {
        v9 = si128.m128i_i64[1];
        while ( v9 )
        {
          --v9;
          if ( *(_BYTE *)(si128.m128i_i64[0] + v9) == 46 )
            goto LABEL_19;
        }
        v9 = -1;
LABEL_19:
        if ( v8 + v7 - 1 == v9 )
        {
          if ( v8 > si128.m128i_i64[1] )
            v8 = si128.m128i_u64[1];
          si128.m128i_i64[1] = v8;
        }
      }
    }
    v4 += 8;
    if ( &v13 == v4 )
      return si128.m128i_i64[0];
    v5 = *(const char **)v4;
  }
}
