// Function: sub_27CACD0
// Address: 0x27cacd0
//
__int64 __fastcall sub_27CACD0(__int64 a1, __int64 *a2, __int64 a3, const __m128i *a4)
{
  _BYTE *v6; // rdx
  _BYTE *v7; // rcx
  __m128i v8; // xmm0
  __int64 v9; // rcx
  __int64 v11; // rcx
  _BYTE *v12; // rax
  _BYTE *v13; // rbx
  __int64 v14; // [rsp+8h] [rbp-38h]
  _BYTE *v15; // [rsp+8h] [rbp-38h]

  v6 = (_BYTE *)a4->m128i_i64[0];
  v7 = (_BYTE *)a4->m128i_i64[1];
  if ( v6 == v7 || (unsigned __int8)sub_DC3A60((__int64)a2, 39, v6, v7) )
  {
LABEL_6:
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  if ( *(_BYTE *)(a3 + 16) )
  {
    v14 = sub_D95540(*(_QWORD *)a3);
    if ( v14 == sub_D95540(a4->m128i_i64[0]) )
    {
      v15 = (_BYTE *)sub_DCDFA0(a2, *(_QWORD *)a3, a4->m128i_i64[0], v9);
      v12 = (_BYTE *)sub_DCE160(a2, *(_QWORD *)(a3 + 8), a4->m128i_i64[1], v11);
      v13 = v12;
      if ( v15 != v12 && !(unsigned __int8)sub_DC3A60((__int64)a2, 39, v15, v12) )
      {
        *(_QWORD *)(a1 + 8) = v13;
        *(_BYTE *)(a1 + 16) = 1;
        *(_QWORD *)a1 = v15;
        return a1;
      }
    }
    goto LABEL_6;
  }
  v8 = _mm_loadu_si128(a4);
  *(_BYTE *)(a1 + 16) = 1;
  *(__m128i *)a1 = v8;
  return a1;
}
