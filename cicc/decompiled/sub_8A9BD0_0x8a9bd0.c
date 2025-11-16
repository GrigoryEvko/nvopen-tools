// Function: sub_8A9BD0
// Address: 0x8a9bd0
//
__int64 __fastcall sub_8A9BD0(const __m128i *a1, __int64 a2)
{
  __m128i *v2; // rax
  __m128i *v3; // r13
  __m128i *v4; // rax
  __int64 v5; // r14
  __m128i *v6; // rax
  __m128i *v7; // r13
  __m128i *v8; // rax
  _QWORD *i; // rbx
  __int64 v10; // r15
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 result; // rax
  __int64 v14; // [rsp+8h] [rbp-38h]

  v5 = sub_878920(**(_QWORD **)(a2 + 64));
  switch ( *(_BYTE *)(v5 + 80) )
  {
    case 4:
    case 5:
      v14 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 80LL);
      break;
    case 6:
      v14 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v14 = *(_QWORD *)(*(_QWORD *)(v5 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v14 = *(_QWORD *)(v5 + 88);
      break;
    default:
      v2 = (__m128i *)sub_879040();
      v2->m128i_i64[1] = a2;
      v3 = v2;
      sub_879080(v2 + 1, a1 + 18, a1[12].m128i_i64[0]);
      a1[20].m128i_i32[0] = 1;
      v4 = (__m128i *)sub_823970(504);
      v3[3].m128i_i64[1] = (__int64)v4;
      qmemcpy(v4, a1, 0x1F8u);
      BUG();
  }
  v6 = (__m128i *)sub_879040();
  v6->m128i_i64[1] = a2;
  v7 = v6;
  sub_879080(v6 + 1, a1 + 18, a1[12].m128i_i64[0]);
  a1[20].m128i_i32[0] = 1;
  v8 = (__m128i *)sub_823970(504);
  v7[3].m128i_i64[1] = (__int64)v8;
  qmemcpy(v8, a1, 0x1F8u);
  for ( i = *(_QWORD **)(v14 + 168); i; i = (_QWORD *)*i )
  {
    v10 = i[1];
    v11 = *(_QWORD *)(v10 + 88);
    if ( (*(_BYTE *)(v11 + 177) & 0x20) == 0
      && (*(_BYTE *)(v11 + 178) & 1) == 0
      && !(unsigned int)sub_8D23B0(*(_QWORD *)(v10 + 88)) )
    {
      v12 = sub_892330(v11);
      sub_8A9820(v7, v10, v11, v5, v12);
    }
  }
  result = *(_QWORD *)(v14 + 184);
  v7->m128i_i64[0] = result;
  *(_QWORD *)(v14 + 184) = v7;
  return result;
}
