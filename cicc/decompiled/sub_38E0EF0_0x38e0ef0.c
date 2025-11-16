// Function: sub_38E0EF0
// Address: 0x38e0ef0
//
unsigned __int64 __fastcall sub_38E0EF0(_QWORD *a1, unsigned __int8 a2, unsigned __int64 a3)
{
  unsigned __int64 result; // rax
  __m128i *v6; // rsi
  unsigned __int64 *v7; // rbx
  __int64 (*v8)(); // rdx
  __int64 v9; // rax
  __int64 v10; // rdi
  const char *v11; // [rsp+0h] [rbp-40h] BYREF
  int v12; // [rsp+8h] [rbp-38h]
  _BYTE v13[12]; // [rsp+Ch] [rbp-34h]

  result = sub_38DD280((__int64)a1, a3);
  if ( result )
  {
    v6 = *(__m128i **)(result + 80);
    v7 = (unsigned __int64 *)result;
    if ( *(__m128i **)(result + 72) == v6 )
    {
      v8 = *(__int64 (**)())(*a1 + 16LL);
      v9 = 1;
      if ( v8 != sub_38DBC10 )
      {
        v9 = ((__int64 (__fastcall *)(_QWORD *))v8)(a1);
        v6 = (__m128i *)v7[10];
      }
      v11 = (const char *)v9;
      result = 0xAFFFFFFFFLL;
      v12 = a2;
      *(_QWORD *)v13 = 0xAFFFFFFFFLL;
      if ( v6 == (__m128i *)v7[11] )
      {
        return sub_38E08D0(v7 + 9, v6, (const __m128i *)&v11);
      }
      else
      {
        if ( v6 )
        {
          *v6 = _mm_loadu_si128((const __m128i *)&v11);
          result = *(_QWORD *)&v13[4];
          v6[1].m128i_i64[0] = *(_QWORD *)&v13[4];
          v6 = (__m128i *)v7[10];
        }
        v7[10] = (unsigned __int64)&v6[1].m128i_u64[1];
      }
    }
    else
    {
      v10 = a1[1];
      *(_WORD *)&v13[4] = 259;
      v11 = "If present, PushMachFrame must be the first UOP";
      return (unsigned __int64)sub_38BE3D0(v10, a3, (__int64)&v11);
    }
  }
  return result;
}
