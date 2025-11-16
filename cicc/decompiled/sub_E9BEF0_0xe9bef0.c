// Function: sub_E9BEF0
// Address: 0xe9bef0
//
__int64 __fastcall sub_E9BEF0(_QWORD *a1, unsigned int a2, unsigned int a3, _QWORD *a4)
{
  __int64 result; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  __int64 (*v9)(); // rax
  unsigned int v10; // eax
  __m128i *v11; // rsi
  __int64 v12; // rdi
  __m128i v13; // [rsp+0h] [rbp-60h] BYREF
  __int64 v14; // [rsp+10h] [rbp-50h]
  char v15; // [rsp+20h] [rbp-40h]
  char v16; // [rsp+21h] [rbp-3Fh]

  result = sub_E99590((__int64)a1, a4);
  if ( result )
  {
    if ( (a3 & 7) != 0 )
    {
      v12 = a1[1];
      v16 = 1;
      v15 = 3;
      v13.m128i_i64[0] = (__int64)"register save offset is not 8 byte aligned";
      return sub_E66880(v12, a4, (__int64)&v13);
    }
    else
    {
      v7 = result;
      v8 = 1;
      v9 = *(__int64 (**)())(*a1 + 88LL);
      if ( v9 != sub_E97650 )
        v8 = ((__int64 (__fastcall *)(_QWORD *))v9)(a1);
      v10 = sub_E91EA0(*(_QWORD *)(a1[1] + 160LL), a2);
      v13.m128i_i64[0] = v8;
      v13.m128i_i64[1] = __PAIR64__(v10, a3);
      result = (unsigned int)(a3 > 0x7FFF8) + 4;
      LODWORD(v14) = (a3 > 0x7FFF8) + 4;
      v11 = *(__m128i **)(v7 + 96);
      if ( v11 == *(__m128i **)(v7 + 104) )
      {
        return sub_E9B9B0((const __m128i **)(v7 + 88), v11, &v13);
      }
      else
      {
        if ( v11 )
        {
          *v11 = _mm_loadu_si128(&v13);
          result = v14;
          v11[1].m128i_i64[0] = v14;
          v11 = *(__m128i **)(v7 + 96);
        }
        *(_QWORD *)(v7 + 96) = (char *)v11 + 24;
      }
    }
  }
  return result;
}
