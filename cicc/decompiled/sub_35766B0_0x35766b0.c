// Function: sub_35766B0
// Address: 0x35766b0
//
void __fastcall sub_35766B0(__int64 a1, __int64 a2)
{
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  __int64 v5; // r15
  __int64 *v6; // r12
  __int64 *v7; // rax
  const char *v8; // rax
  size_t v9; // rdx
  _BYTE *v10; // rdi
  unsigned __int8 *v11; // rsi
  _BYTE *v12; // rax
  size_t v13; // r13
  _BYTE *v14; // rax

  v3 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0x23u )
  {
    v5 = sub_CB6200(a2, "MachineUniformityInfo for function: ", 0x24u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_44E5AE0);
    v3[2].m128i_i32[0] = 540700271;
    v5 = a2;
    *v3 = si128;
    v3[1] = _mm_load_si128((const __m128i *)&xmmword_44E5AF0);
    *(_QWORD *)(a2 + 32) += 36LL;
  }
  v6 = (__int64 *)(a1 + 200);
  v7 = (__int64 *)sub_3574EF0(a1 + 200);
  v8 = sub_2E791E0(v7);
  v10 = *(_BYTE **)(v5 + 32);
  v11 = (unsigned __int8 *)v8;
  v12 = *(_BYTE **)(v5 + 24);
  v13 = v9;
  if ( v12 - v10 < v9 )
  {
    v5 = sub_CB6200(v5, v11, v9);
    v12 = *(_BYTE **)(v5 + 24);
    v10 = *(_BYTE **)(v5 + 32);
  }
  else if ( v9 )
  {
    memcpy(v10, v11, v9);
    v14 = *(_BYTE **)(v5 + 24);
    v10 = (_BYTE *)(v13 + *(_QWORD *)(v5 + 32));
    *(_QWORD *)(v5 + 32) = v10;
    if ( v14 != v10 )
      goto LABEL_6;
LABEL_9:
    sub_CB6200(v5, (unsigned __int8 *)"\n", 1u);
    goto LABEL_7;
  }
  if ( v12 == v10 )
    goto LABEL_9;
LABEL_6:
  *v10 = 10;
  ++*(_QWORD *)(v5 + 32);
LABEL_7:
  sub_35766A0(v6, a2);
}
