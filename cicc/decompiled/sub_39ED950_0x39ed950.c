// Function: sub_39ED950
// Address: 0x39ed950
//
__int64 __fastcall sub_39ED950(
        __int64 a1,
        unsigned int a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        unsigned int a6,
        unsigned __int64 a7)
{
  __int64 v12; // rdi
  __m128i *v13; // rdx
  __m128i si128; // xmm0
  __int64 v15; // rax
  _QWORD *v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rax
  void *v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rdi
  _BYTE *v22; // rax
  __int64 v23; // rdi
  _BYTE *v24; // rax
  __int64 v25; // rdi
  _BYTE *v26; // rax

  v12 = *(_QWORD *)(a1 + 272);
  v13 = *(__m128i **)(v12 + 24);
  if ( *(_QWORD *)(v12 + 16) - (_QWORD)v13 <= 0x13u )
  {
    v12 = sub_16E7EE0(v12, "\t.cv_inline_site_id ", 0x14u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7FA40);
    v13[1].m128i_i32[0] = 543451487;
    *v13 = si128;
    *(_QWORD *)(v12 + 24) += 20LL;
  }
  v15 = sub_16E7A90(v12, a2);
  v16 = *(_QWORD **)(v15 + 24);
  v17 = v15;
  if ( *(_QWORD *)(v15 + 16) - (_QWORD)v16 <= 7u )
  {
    v17 = sub_16E7EE0(v15, " within ", 8u);
  }
  else
  {
    *v16 = 0x206E696874697720LL;
    *(_QWORD *)(v15 + 24) += 8LL;
  }
  v18 = sub_16E7A90(v17, a3);
  v19 = *(void **)(v18 + 24);
  v20 = v18;
  if ( *(_QWORD *)(v18 + 16) - (_QWORD)v19 <= 0xBu )
  {
    v20 = sub_16E7EE0(v18, " inlined_at ", 0xCu);
  }
  else
  {
    qmemcpy(v19, " inlined_at ", 12);
    *(_QWORD *)(v18 + 24) += 12LL;
  }
  v21 = sub_16E7A90(v20, a4);
  v22 = *(_BYTE **)(v21 + 24);
  if ( (unsigned __int64)v22 >= *(_QWORD *)(v21 + 16) )
  {
    v21 = sub_16E7DE0(v21, 32);
  }
  else
  {
    *(_QWORD *)(v21 + 24) = v22 + 1;
    *v22 = 32;
  }
  v23 = sub_16E7A90(v21, a5);
  v24 = *(_BYTE **)(v23 + 24);
  if ( (unsigned __int64)v24 >= *(_QWORD *)(v23 + 16) )
  {
    v23 = sub_16E7DE0(v23, 32);
  }
  else
  {
    *(_QWORD *)(v23 + 24) = v24 + 1;
    *v24 = 32;
  }
  v25 = sub_16E7A90(v23, a6);
  v26 = *(_BYTE **)(v25 + 24);
  if ( (unsigned __int64)v26 >= *(_QWORD *)(v25 + 16) )
  {
    sub_16E7DE0(v25, 10);
  }
  else
  {
    *(_QWORD *)(v25 + 24) = v26 + 1;
    *v26 = 10;
  }
  return sub_38DBE80(a1, a2, a3, a4, a5, a6, a7);
}
