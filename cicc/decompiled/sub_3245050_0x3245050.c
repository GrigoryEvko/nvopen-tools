// Function: sub_3245050
// Address: 0x3245050
//
__m128i *__fastcall sub_3245050(_QWORD *a1, const __m128i *a2)
{
  _QWORD *v2; // r15
  __int64 v3; // r12
  unsigned __int32 v4; // ebx
  unsigned __int32 v5; // edx
  __int64 v6; // rax
  bool v7; // r8
  __m128i *v8; // rbx
  __int64 v10; // rax
  char v11; // [rsp+Ch] [rbp-34h]

  v2 = a1 + 1;
  v3 = a1[2];
  if ( !v3 )
  {
    v3 = (__int64)(a1 + 1);
    if ( v2 == (_QWORD *)a1[3] )
    {
      v7 = 1;
LABEL_11:
      v11 = v7;
      v8 = (__m128i *)sub_22077B0(0x30u);
      v8[2] = _mm_loadu_si128(a2);
      sub_220F040(v11, (__int64)v8, (_QWORD *)v3, v2);
      ++a1[5];
      return v8;
    }
    v4 = a2->m128i_i32[0];
LABEL_13:
    v10 = sub_220EF80(v3);
    if ( v4 <= *(_DWORD *)(v10 + 32) )
      return (__m128i *)v10;
LABEL_9:
    v7 = 1;
    if ( v2 != (_QWORD *)v3 )
      v7 = v4 < *(_DWORD *)(v3 + 32);
    goto LABEL_11;
  }
  v4 = a2->m128i_i32[0];
  while ( 1 )
  {
    v5 = *(_DWORD *)(v3 + 32);
    v6 = *(_QWORD *)(v3 + 24);
    if ( v4 < v5 )
      v6 = *(_QWORD *)(v3 + 16);
    if ( !v6 )
      break;
    v3 = v6;
  }
  if ( v4 < v5 )
  {
    if ( v3 == a1[3] )
      goto LABEL_9;
    goto LABEL_13;
  }
  if ( v4 > v5 )
    goto LABEL_9;
  return (__m128i *)v3;
}
