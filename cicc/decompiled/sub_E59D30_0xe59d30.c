// Function: sub_E59D30
// Address: 0xe59d30
//
__int64 __fastcall sub_E59D30(__int64 a1, unsigned int a2, unsigned int a3, unsigned int a4, __int64 a5, __int64 a6)
{
  __int64 v11; // rdi
  __m128i *v12; // rdx
  __m128i si128; // xmm0
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 v16; // rdi
  _BYTE *v17; // rax
  __int64 v18; // rdi
  _BYTE *v19; // rax
  __int64 v20; // rdi
  _BYTE *v21; // rax

  v11 = *(_QWORD *)(a1 + 304);
  v12 = *(__m128i **)(v11 + 32);
  if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 0x15u )
  {
    v11 = sub_CB6200(v11, "\t.cv_inline_linetable\t", 0x16u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7FA30);
    v12[1].m128i_i32[0] = 1818386804;
    v12[1].m128i_i16[2] = 2405;
    *v12 = si128;
    *(_QWORD *)(v11 + 32) += 22LL;
  }
  v14 = sub_CB59D0(v11, a2);
  v15 = *(_BYTE **)(v14 + 32);
  if ( (unsigned __int64)v15 >= *(_QWORD *)(v14 + 24) )
  {
    v14 = sub_CB5D20(v14, 32);
  }
  else
  {
    *(_QWORD *)(v14 + 32) = v15 + 1;
    *v15 = 32;
  }
  v16 = sub_CB59D0(v14, a3);
  v17 = *(_BYTE **)(v16 + 32);
  if ( (unsigned __int64)v17 >= *(_QWORD *)(v16 + 24) )
  {
    v16 = sub_CB5D20(v16, 32);
  }
  else
  {
    *(_QWORD *)(v16 + 32) = v17 + 1;
    *v17 = 32;
  }
  v18 = sub_CB59D0(v16, a4);
  v19 = *(_BYTE **)(v18 + 32);
  if ( (unsigned __int64)v19 >= *(_QWORD *)(v18 + 24) )
  {
    sub_CB5D20(v18, 32);
  }
  else
  {
    *(_QWORD *)(v18 + 32) = v19 + 1;
    *v19 = 32;
  }
  sub_EA12C0(a5, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  v20 = *(_QWORD *)(a1 + 304);
  v21 = *(_BYTE **)(v20 + 32);
  if ( (unsigned __int64)v21 >= *(_QWORD *)(v20 + 24) )
  {
    sub_CB5D20(v20, 32);
  }
  else
  {
    *(_QWORD *)(v20 + 32) = v21 + 1;
    *v21 = 32;
  }
  sub_EA12C0(a6, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  sub_E4D880(a1);
  return nullsub_347(a1, a2, a3, a4, a5, a6);
}
