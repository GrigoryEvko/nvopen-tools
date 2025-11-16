// Function: sub_19497C0
// Address: 0x19497c0
//
__int64 __fastcall sub_19497C0(__int64 a1, _QWORD *a2, __int64 a3, const __m128i *a4, __m128i a5, __m128i a6)
{
  __int64 v8; // rdx
  __int64 v9; // rcx
  __m128i v10; // xmm0
  __int64 v11; // r15
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rbx

  v8 = a4->m128i_i64[0];
  v9 = a4->m128i_i64[1];
  if ( v8 == v9 || (unsigned __int8)sub_147A340((__int64)a2, 0x27u, v8, v9) )
  {
LABEL_6:
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  if ( *(_BYTE *)(a3 + 16) )
  {
    v11 = sub_1456040(*(_QWORD *)a3);
    if ( v11 == sub_1456040(a4->m128i_i64[0]) )
    {
      v13 = sub_147A9C0(a2, *(_QWORD *)a3, a4->m128i_i64[0], a5, a6);
      v14 = sub_1480950(a2, *(_QWORD *)(a3 + 8), a4->m128i_i64[1], a5, a6);
      v15 = v14;
      if ( v13 != v14 && !(unsigned __int8)sub_147A340((__int64)a2, 0x27u, v13, v14) )
      {
        *(_BYTE *)(a1 + 16) = 1;
        *(_QWORD *)a1 = v13;
        *(_QWORD *)(a1 + 8) = v15;
        return a1;
      }
    }
    goto LABEL_6;
  }
  v10 = _mm_loadu_si128(a4);
  *(_BYTE *)(a1 + 16) = 1;
  *(__m128i *)a1 = v10;
  return a1;
}
