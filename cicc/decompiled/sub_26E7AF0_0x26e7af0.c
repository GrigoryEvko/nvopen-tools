// Function: sub_26E7AF0
// Address: 0x26e7af0
//
__int64 __fastcall sub_26E7AF0(_QWORD *a1, const void **a2, const __m128i *a3, char a4)
{
  size_t v8; // rdx
  __int64 result; // rax
  const void *v10; // rdi
  const void *v11; // rsi
  int v12; // r8d
  __int64 v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v8 = (size_t)a2[1];
  if ( v8 != a3->m128i_i64[1] )
    goto LABEL_2;
  v10 = *a2;
  v11 = (const void *)a3->m128i_i64[0];
  if ( v10 == (const void *)a3->m128i_i64[0] )
    return 1;
  if ( !v10 || !v11 || (v12 = memcmp(v10, v11, v8), result = 1, v12) )
  {
LABEL_2:
    result = LOBYTE(qword_4FF8120[17]);
    if ( (_BYTE)result )
    {
      v13[0] = 0;
      if ( sub_26E20C0((__int64)a1, (__int64)a2, v13) || !sub_26E2030((__int64)a1, (__int64)a3) )
        return 0;
      else
        return sub_26E7A00(a1, v13[0], a3, a4);
    }
  }
  return result;
}
