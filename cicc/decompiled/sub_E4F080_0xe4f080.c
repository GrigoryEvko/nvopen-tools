// Function: sub_E4F080
// Address: 0xe4f080
//
_BYTE *__fastcall sub_E4F080(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  __m128i *v6; // rdx
  __m128i si128; // xmm0
  __int64 v8; // rdi
  _BYTE *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r14
  _BYTE *v12; // rax
  __int64 v13; // rdi
  _BYTE *v14; // rax
  __int64 v15; // rdi
  _WORD *v16; // rdx

  v4 = *(_QWORD *)(a1 + 304);
  v6 = *(__m128i **)(v4 + 32);
  if ( *(_QWORD *)(v4 + 24) - (_QWORD)v6 <= 0x10u )
  {
    v4 = sub_CB6200(v4, "\t.linker_option \"", 0x11u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F7F860);
    v6[1].m128i_i8[0] = 34;
    *v6 = si128;
    *(_QWORD *)(v4 + 32) += 17LL;
  }
  v8 = sub_CB6200(v4, *(unsigned __int8 **)a2, *(_QWORD *)(a2 + 8));
  v9 = *(_BYTE **)(v8 + 32);
  if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 24) )
  {
    sub_CB5D20(v8, 34);
  }
  else
  {
    *(_QWORD *)(v8 + 32) = v9 + 1;
    *v9 = 34;
  }
  v10 = a2 + 32 * a3;
  v11 = a2 + 32;
  if ( v10 != a2 + 32 )
  {
    while ( 1 )
    {
      v15 = *(_QWORD *)(a1 + 304);
      v16 = *(_WORD **)(v15 + 32);
      if ( *(_QWORD *)(v15 + 24) - (_QWORD)v16 > 1u )
      {
        *v16 = 8236;
        v12 = (_BYTE *)(*(_QWORD *)(v15 + 32) + 2LL);
        *(_QWORD *)(v15 + 32) = v12;
        if ( *(_QWORD *)(v15 + 24) <= (unsigned __int64)v12 )
          goto LABEL_13;
      }
      else
      {
        v15 = sub_CB6200(v15, (unsigned __int8 *)", ", 2u);
        v12 = *(_BYTE **)(v15 + 32);
        if ( *(_QWORD *)(v15 + 24) <= (unsigned __int64)v12 )
        {
LABEL_13:
          v15 = sub_CB5D20(v15, 34);
          goto LABEL_9;
        }
      }
      *(_QWORD *)(v15 + 32) = v12 + 1;
      *v12 = 34;
LABEL_9:
      v13 = sub_CB6200(v15, *(unsigned __int8 **)v11, *(_QWORD *)(v11 + 8));
      v14 = *(_BYTE **)(v13 + 32);
      if ( (unsigned __int64)v14 >= *(_QWORD *)(v13 + 24) )
      {
        v11 += 32;
        sub_CB5D20(v13, 34);
        if ( v10 == v11 )
          return sub_E4D880(a1);
      }
      else
      {
        v11 += 32;
        *(_QWORD *)(v13 + 32) = v14 + 1;
        *v14 = 34;
        if ( v10 == v11 )
          return sub_E4D880(a1);
      }
    }
  }
  return sub_E4D880(a1);
}
