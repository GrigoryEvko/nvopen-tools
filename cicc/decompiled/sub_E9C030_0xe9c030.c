// Function: sub_E9C030
// Address: 0xe9c030
//
__int64 __fastcall sub_E9C030(_QWORD *a1, unsigned __int8 a2, _QWORD *a3)
{
  __int64 result; // rax
  __m128i *v6; // rsi
  __int64 v7; // rbx
  __int64 (*v8)(); // rdx
  __int64 v9; // rax
  __int64 v10; // rdi
  const char *v11; // [rsp+0h] [rbp-50h] BYREF
  int v12; // [rsp+8h] [rbp-48h]
  _BYTE v13[12]; // [rsp+Ch] [rbp-44h]
  char v14; // [rsp+20h] [rbp-30h]
  char v15; // [rsp+21h] [rbp-2Fh]

  result = sub_E99590((__int64)a1, a3);
  if ( result )
  {
    v6 = *(__m128i **)(result + 96);
    v7 = result;
    if ( *(__m128i **)(result + 88) == v6 )
    {
      v8 = *(__int64 (**)())(*a1 + 88LL);
      v9 = 1;
      if ( v8 != sub_E97650 )
      {
        v9 = ((__int64 (__fastcall *)(_QWORD *))v8)(a1);
        v6 = *(__m128i **)(v7 + 96);
      }
      v11 = (const char *)v9;
      result = 0xAFFFFFFFFLL;
      v12 = a2;
      *(_QWORD *)v13 = 0xAFFFFFFFFLL;
      if ( v6 == *(__m128i **)(v7 + 104) )
      {
        return sub_E9B9B0((const __m128i **)(v7 + 88), v6, (const __m128i *)&v11);
      }
      else
      {
        if ( v6 )
        {
          *v6 = _mm_loadu_si128((const __m128i *)&v11);
          result = *(_QWORD *)&v13[4];
          v6[1].m128i_i64[0] = *(_QWORD *)&v13[4];
          v6 = *(__m128i **)(v7 + 96);
        }
        *(_QWORD *)(v7 + 96) = (char *)v6 + 24;
      }
    }
    else
    {
      v10 = a1[1];
      v15 = 1;
      v14 = 3;
      v11 = "If present, PushMachFrame must be the first UOP";
      return sub_E66880(v10, a3, (__int64)&v11);
    }
  }
  return result;
}
