// Function: sub_13CA0A0
// Address: 0x13ca0a0
//
_QWORD *__fastcall sub_13CA0A0(_QWORD *a1, __int64 a2)
{
  __m128i *v2; // rax
  __m128i si128; // xmm0
  _WORD *v4; // rdx
  _QWORD *result; // rax
  _WORD *v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // rax
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
  _QWORD *v28; // [rsp+8h] [rbp-48h]
  _QWORD *v30; // [rsp+18h] [rbp-38h]

  v2 = *(__m128i **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v2 <= 0x11u )
  {
    sub_16E7EE0(a2, "IV Users for loop ", 18);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4289EE0);
    v2[1].m128i_i16[0] = 8304;
    *v2 = si128;
    *(_QWORD *)(a2 + 24) += 18LL;
  }
  sub_15537D0(**(_QWORD **)(*a1 + 32LL), a2, 0);
  if ( (unsigned __int8)sub_1481F90(a1[4]) )
  {
    v24 = *(__m128i **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v24 <= 0x1Au )
    {
      v26 = sub_16E7EE0(a2, " with backedge-taken count ", 27);
    }
    else
    {
      v25 = _mm_load_si128((const __m128i *)&xmmword_4289EF0);
      v26 = a2;
      qmemcpy(&v24[1], "aken count ", 11);
      *v24 = v25;
      *(_QWORD *)(a2 + 24) += 27LL;
    }
    v27 = sub_1481F60(a1[4], *a1);
    sub_1456620(v27, v26);
  }
  v4 = *(_WORD **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v4 <= 1u )
  {
    sub_16E7EE0(a2, ":\n", 2);
  }
  else
  {
    *v4 = 2618;
    *(_QWORD *)(a2 + 24) += 2LL;
  }
  result = a1 + 26;
  v28 = a1 + 26;
  v30 = (_QWORD *)a1[27];
  if ( v30 != a1 + 26 )
  {
    do
    {
      v6 = *(_WORD **)(a2 + 24);
      v7 = (__int64)(v30 - 4);
      if ( !v30 )
        v7 = 0;
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v6 <= 1u )
      {
        sub_16E7EE0(a2, "  ", 2);
      }
      else
      {
        *v6 = 8224;
        *(_QWORD *)(a2 + 24) += 2LL;
      }
      sub_15537D0(*(_QWORD *)(v7 + 72), a2, 0);
      v8 = *(_QWORD *)(a2 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v8) <= 2 )
      {
        v9 = sub_16E7EE0(a2, " = ", 3);
      }
      else
      {
        *(_BYTE *)(v8 + 2) = 32;
        v9 = a2;
        *(_WORD *)v8 = 15648;
        *(_QWORD *)(a2 + 24) += 3LL;
      }
      v10 = sub_13CA090((__int64)a1, v7);
      sub_1456620(v10, v9);
      v11 = *(_QWORD **)(v7 + 96);
      if ( v11 == *(_QWORD **)(v7 + 88) )
        v12 = *(unsigned int *)(v7 + 108);
      else
        v12 = *(unsigned int *)(v7 + 104);
      v13 = &v11[v12];
      v14 = *(_QWORD *)(a2 + 24);
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
        while ( v13 != v16 )
        {
          if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v14) <= 0x14 )
          {
            sub_16E7EE0(a2, " (post-inc with loop ", 21);
          }
          else
          {
            v19 = _mm_load_si128((const __m128i *)&xmmword_4289F00);
            *(_DWORD *)(v14 + 16) = 1886351212;
            *(_BYTE *)(v14 + 20) = 32;
            *(__m128i *)v14 = v19;
            *(_QWORD *)(a2 + 24) += 21LL;
          }
          sub_15537D0(**(_QWORD **)(v15 + 32), a2, 0);
          v20 = *(_BYTE **)(a2 + 24);
          if ( *(_BYTE **)(a2 + 16) == v20 )
          {
            sub_16E7EE0(a2, ")", 1);
            v14 = *(_QWORD *)(a2 + 24);
          }
          else
          {
            *v20 = 41;
            v14 = *(_QWORD *)(a2 + 24) + 1LL;
            *(_QWORD *)(a2 + 24) = v14;
          }
          v21 = v16 + 1;
          if ( v16 + 1 == v13 )
            break;
          while ( 1 )
          {
            v15 = *v21;
            v16 = v21;
            if ( *v21 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v13 == ++v21 )
            {
              if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v14) > 4 )
                goto LABEL_19;
              goto LABEL_34;
            }
          }
        }
      }
LABEL_18:
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v14) <= 4 )
      {
LABEL_34:
        sub_16E7EE0(a2, " in  ", 5);
        v17 = *(_QWORD *)(v7 + 24);
        if ( v17 )
        {
LABEL_20:
          sub_155C2B0(v17, a2, 0);
          v18 = *(_BYTE **)(a2 + 24);
          goto LABEL_21;
        }
      }
      else
      {
LABEL_19:
        *(_DWORD *)v14 = 544106784;
        *(_BYTE *)(v14 + 4) = 32;
        *(_QWORD *)(a2 + 24) += 5LL;
        v17 = *(_QWORD *)(v7 + 24);
        if ( v17 )
          goto LABEL_20;
      }
      v22 = *(__m128i **)(a2 + 24);
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v22 <= 0x13u )
      {
        sub_16E7EE0(a2, "Printing <null> User", 20);
        v18 = *(_BYTE **)(a2 + 24);
LABEL_21:
        if ( (unsigned __int64)v18 >= *(_QWORD *)(a2 + 16) )
          goto LABEL_37;
        goto LABEL_22;
      }
      v23 = _mm_load_si128((const __m128i *)&xmmword_4289F10);
      v22[1].m128i_i32[0] = 1919251285;
      *v22 = v23;
      v18 = (_BYTE *)(*(_QWORD *)(a2 + 24) + 20LL);
      *(_QWORD *)(a2 + 24) = v18;
      if ( (unsigned __int64)v18 >= *(_QWORD *)(a2 + 16) )
      {
LABEL_37:
        sub_16E7DE0(a2, 10);
        goto LABEL_23;
      }
LABEL_22:
      *(_QWORD *)(a2 + 24) = v18 + 1;
      *v18 = 10;
LABEL_23:
      result = (_QWORD *)v30[1];
      v30 = result;
    }
    while ( v28 != result );
  }
  return result;
}
