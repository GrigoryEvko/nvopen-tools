// Function: sub_22ACDD0
// Address: 0x22acdd0
//
__int64 **__fastcall sub_22ACDD0(__int64 **a1, __int64 a2)
{
  __m128i *v2; // rax
  __m128i si128; // xmm0
  _WORD *v4; // rdx
  __int64 **result; // rax
  _WORD *v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 *v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // r13
  __int64 v14; // rdx
  __int64 v15; // r12
  _QWORD *v16; // r14
  __int64 v17; // rdi
  _BYTE *v18; // rax
  __m128i v19; // xmm0
  _BYTE *v20; // rax
  _QWORD *v21; // rax
  __m128i *v22; // rdx
  __m128i v23; // xmm0
  __m128i *v24; // rax
  __m128i v25; // xmm0
  __int64 v26; // r12
  __int64 v27; // rax
  __int64 **v28; // [rsp+8h] [rbp-48h]
  __int64 *v30; // [rsp+18h] [rbp-38h]

  v2 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v2 <= 0x11u )
  {
    sub_CB6200(a2, "IV Users for loop ", 0x12u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4289EE0);
    v2[1].m128i_i16[0] = 8304;
    *v2 = si128;
    *(_QWORD *)(a2 + 32) += 18LL;
  }
  sub_A5BF40(*(unsigned __int8 **)(*a1)[4], a2, 0, 0);
  if ( (unsigned __int8)sub_DCFA10(a1[4], (char *)*a1) )
  {
    v24 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v24 <= 0x1Au )
    {
      v26 = sub_CB6200(a2, " with backedge-taken count ", 0x1Bu);
    }
    else
    {
      v25 = _mm_load_si128((const __m128i *)&xmmword_4289EF0);
      v26 = a2;
      qmemcpy(&v24[1], "aken count ", 11);
      *v24 = v25;
      *(_QWORD *)(a2 + 32) += 27LL;
    }
    v27 = sub_DCF3A0(a1[4], (char *)*a1, 0);
    sub_D955C0(v27, v26);
  }
  v4 = *(_WORD **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 1u )
  {
    sub_CB6200(a2, (unsigned __int8 *)":\n", 2u);
  }
  else
  {
    *v4 = 2618;
    *(_QWORD *)(a2 + 32) += 2LL;
  }
  result = a1 + 25;
  v28 = a1 + 25;
  v30 = a1[26];
  if ( v30 != (__int64 *)(a1 + 25) )
  {
    do
    {
      v6 = *(_WORD **)(a2 + 32);
      v7 = (__int64)(v30 - 4);
      if ( !v30 )
        v7 = 0;
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v6 <= 1u )
      {
        sub_CB6200(a2, (unsigned __int8 *)"  ", 2u);
      }
      else
      {
        *v6 = 8224;
        *(_QWORD *)(a2 + 32) += 2LL;
      }
      sub_A5BF40(*(unsigned __int8 **)(v7 + 72), a2, 0, 0);
      v8 = *(_QWORD *)(a2 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v8) <= 2 )
      {
        v9 = sub_CB6200(a2, (unsigned __int8 *)" = ", 3u);
      }
      else
      {
        *(_BYTE *)(v8 + 2) = 32;
        v9 = a2;
        *(_WORD *)v8 = 15648;
        *(_QWORD *)(a2 + 32) += 3LL;
      }
      v10 = sub_22ACDC0((__int64)a1, v7);
      sub_D955C0((__int64)v10, v9);
      v11 = *(_QWORD **)(v7 + 88);
      if ( *(_BYTE *)(v7 + 108) )
        v12 = *(unsigned int *)(v7 + 100);
      else
        v12 = *(unsigned int *)(v7 + 96);
      v13 = &v11[v12];
      v14 = *(_QWORD *)(a2 + 32);
      if ( v11 != v13 )
      {
        while ( 1 )
        {
          v15 = *v11;
          v16 = v11;
          if ( *v11 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v13 == ++v11 )
            goto LABEL_18;
        }
        if ( v13 != v11 )
        {
          if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v14) > 0x14 )
          {
LABEL_27:
            v19 = _mm_load_si128((const __m128i *)&xmmword_4289F00);
            *(_DWORD *)(v14 + 16) = 1886351212;
            *(_BYTE *)(v14 + 20) = 32;
            *(__m128i *)v14 = v19;
            *(_QWORD *)(a2 + 32) += 21LL;
            goto LABEL_28;
          }
          while ( 1 )
          {
            sub_CB6200(a2, " (post-inc with loop ", 0x15u);
LABEL_28:
            sub_A5BF40(**(unsigned __int8 ***)(v15 + 32), a2, 0, 0);
            v20 = *(_BYTE **)(a2 + 32);
            if ( *(_BYTE **)(a2 + 24) == v20 )
            {
              sub_CB6200(a2, (unsigned __int8 *)")", 1u);
              v14 = *(_QWORD *)(a2 + 32);
            }
            else
            {
              *v20 = 41;
              v14 = *(_QWORD *)(a2 + 32) + 1LL;
              *(_QWORD *)(a2 + 32) = v14;
            }
            v21 = v16 + 1;
            if ( v16 + 1 == v13 )
              break;
            v15 = *v21;
            for ( ++v16; *v21 >= 0xFFFFFFFFFFFFFFFELL; v16 = v21 )
            {
              if ( v13 == ++v21 )
                goto LABEL_18;
              v15 = *v21;
            }
            if ( v16 == v13 )
              break;
            if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v14) > 0x14 )
              goto LABEL_27;
          }
        }
      }
LABEL_18:
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v14) <= 4 )
      {
        sub_CB6200(a2, " in  ", 5u);
        v17 = *(_QWORD *)(v7 + 24);
        if ( v17 )
        {
LABEL_20:
          sub_A69870(v17, (_BYTE *)a2, 0);
          v18 = *(_BYTE **)(a2 + 32);
          goto LABEL_21;
        }
      }
      else
      {
        *(_DWORD *)v14 = 544106784;
        *(_BYTE *)(v14 + 4) = 32;
        *(_QWORD *)(a2 + 32) += 5LL;
        v17 = *(_QWORD *)(v7 + 24);
        if ( v17 )
          goto LABEL_20;
      }
      v22 = *(__m128i **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v22 <= 0x13u )
      {
        sub_CB6200(a2, "Printing <null> User", 0x14u);
        v18 = *(_BYTE **)(a2 + 32);
LABEL_21:
        if ( (unsigned __int64)v18 >= *(_QWORD *)(a2 + 24) )
          goto LABEL_42;
        goto LABEL_22;
      }
      v23 = _mm_load_si128((const __m128i *)&xmmword_4289F10);
      v22[1].m128i_i32[0] = 1919251285;
      *v22 = v23;
      v18 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 20LL);
      *(_QWORD *)(a2 + 32) = v18;
      if ( (unsigned __int64)v18 >= *(_QWORD *)(a2 + 24) )
      {
LABEL_42:
        sub_CB5D20(a2, 10);
        goto LABEL_23;
      }
LABEL_22:
      *(_QWORD *)(a2 + 32) = v18 + 1;
      *v18 = 10;
LABEL_23:
      result = (__int64 **)v30[1];
      v30 = (__int64 *)result;
    }
    while ( v28 != result );
  }
  return result;
}
