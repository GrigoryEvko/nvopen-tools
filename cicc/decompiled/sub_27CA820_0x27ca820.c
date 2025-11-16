// Function: sub_27CA820
// Address: 0x27ca820
//
__int64 __fastcall sub_27CA820(__int64 a1, __int64 *a2, __int64 a3, const __m128i *a4)
{
  _BYTE *v6; // rdx
  _BYTE *v7; // rcx
  __m128i v8; // xmm0
  __int64 v9; // rcx
  _BYTE *v11; // rax
  _BYTE *v12; // rbx
  __int64 v13; // [rsp+8h] [rbp-38h]
  _BYTE *v14; // [rsp+8h] [rbp-38h]

  v6 = (_BYTE *)a4->m128i_i64[0];
  v7 = (_BYTE *)a4->m128i_i64[1];
  if ( v6 == v7 || (unsigned __int8)sub_DC3A60((__int64)a2, 35, v6, v7) )
  {
LABEL_6:
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  if ( *(_BYTE *)(a3 + 16) )
  {
    v13 = sub_D95540(*(_QWORD *)a3);
    if ( v13 == sub_D95540(a4->m128i_i64[0]) )
    {
      v14 = (_BYTE *)sub_DCE050(a2, *(_QWORD *)a3, a4->m128i_i64[0], v9);
      v11 = sub_DCEE80(a2, *(_QWORD *)(a3 + 8), a4->m128i_i64[1], 0);
      v12 = v11;
      if ( v14 != v11 && !(unsigned __int8)sub_DC3A60((__int64)a2, 35, v14, v11) )
      {
        *(_QWORD *)(a1 + 8) = v12;
        *(_BYTE *)(a1 + 16) = 1;
        *(_QWORD *)a1 = v14;
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
