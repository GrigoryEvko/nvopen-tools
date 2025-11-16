// Function: sub_1E0B0B0
// Address: 0x1e0b0b0
//
__int64 __fastcall sub_1E0B0B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __m128i *v5; // rdx
  __m128i si128; // xmm0
  __int64 v7; // r14
  const char *v8; // rax
  size_t v9; // rdx
  _WORD *v10; // rdi
  char *v11; // rsi
  unsigned __int64 v12; // rax
  _BYTE *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r15
  __int64 (*v16)(void); // rax
  __int64 v17; // rax
  __m128i *v18; // rdx
  __m128i v19; // xmm0
  __int64 v20; // rax
  unsigned int *v21; // rbx
  __int64 v22; // rsi
  __int64 *v23; // rdi
  __int64 v24; // rdx
  _WORD *v25; // rdx
  _BYTE *v26; // rax
  __int64 i; // r15
  _BYTE *v28; // rax
  __m128i *v29; // rdx
  __m128i v30; // xmm0
  const char *v31; // rax
  size_t v32; // rdx
  _BYTE *v33; // rdi
  char *v34; // rsi
  unsigned __int64 v35; // rax
  unsigned __int64 v37; // rax
  unsigned __int64 v38; // rax
  _DWORD *v39; // rdx
  __int64 v40; // r10
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // [rsp+0h] [rbp-A0h]
  unsigned int *v44; // [rsp+10h] [rbp-90h]
  size_t v45; // [rsp+10h] [rbp-90h]
  size_t v47; // [rsp+18h] [rbp-88h]
  _BYTE v48[16]; // [rsp+20h] [rbp-80h] BYREF
  void (__fastcall *v49)(_BYTE *, _BYTE *, __int64); // [rsp+30h] [rbp-70h]
  void (__fastcall *v50)(_BYTE *, __int64); // [rsp+38h] [rbp-68h]
  __int64 v51[2]; // [rsp+40h] [rbp-60h] BYREF
  void (__fastcall *v52)(__int64 *, __int64 *, __int64); // [rsp+50h] [rbp-50h]
  void (__fastcall *v53)(__int64 *, __int64); // [rsp+58h] [rbp-48h]

  v3 = a2;
  v5 = *(__m128i **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v5 <= 0x1Bu )
  {
    v7 = sub_16E7EE0(a2, "# Machine code for function ", 0x1Cu);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_42EB020);
    v7 = a2;
    qmemcpy(&v5[1], "or function ", 12);
    *v5 = si128;
    *(_QWORD *)(a2 + 24) += 28LL;
  }
  v8 = sub_1E0A440((__int64 *)a1);
  v10 = *(_WORD **)(v7 + 24);
  v11 = (char *)v8;
  v12 = *(_QWORD *)(v7 + 16) - (_QWORD)v10;
  if ( v9 > v12 )
  {
    v41 = sub_16E7EE0(v7, v11, v9);
    v10 = *(_WORD **)(v41 + 24);
    v7 = v41;
    v12 = *(_QWORD *)(v41 + 16) - (_QWORD)v10;
  }
  else if ( v9 )
  {
    v45 = v9;
    memcpy(v10, v11, v9);
    v10 = (_WORD *)(v45 + *(_QWORD *)(v7 + 24));
    v37 = *(_QWORD *)(v7 + 16) - (_QWORD)v10;
    *(_QWORD *)(v7 + 24) = v10;
    if ( v37 > 1 )
      goto LABEL_6;
    goto LABEL_41;
  }
  if ( v12 > 1 )
  {
LABEL_6:
    *v10 = 8250;
    *(_QWORD *)(v7 + 24) += 2LL;
    goto LABEL_7;
  }
LABEL_41:
  sub_16E7EE0(v7, ": ", 2u);
LABEL_7:
  sub_1E09CA0(a1 + 352, v3);
  v13 = *(_BYTE **)(v3 + 24);
  if ( (unsigned __int64)v13 >= *(_QWORD *)(v3 + 16) )
  {
    sub_16E7DE0(v3, 10);
  }
  else
  {
    *(_QWORD *)(v3 + 24) = v13 + 1;
    *v13 = 10;
  }
  sub_1E08940(*(_QWORD *)(a1 + 56), a1, v3);
  v14 = *(_QWORD *)(a1 + 72);
  if ( v14 )
    sub_1E0A8F0(v14, v3);
  v15 = 0;
  sub_1E0AE90(*(_QWORD *)(a1 + 64), v3);
  v16 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 16) + 112LL);
  if ( v16 != sub_1D00B10 )
    v15 = v16();
  v17 = *(_QWORD *)(a1 + 40);
  if ( v17 && *(_QWORD *)(v17 + 360) != *(_QWORD *)(v17 + 368) )
  {
    v18 = *(__m128i **)(v3 + 24);
    if ( *(_QWORD *)(v3 + 16) - (_QWORD)v18 <= 0x12u )
    {
      sub_16E7EE0(v3, "Function Live Ins: ", 0x13u);
    }
    else
    {
      v19 = _mm_load_si128((const __m128i *)&xmmword_42EB030);
      v18[1].m128i_i8[2] = 32;
      v18[1].m128i_i16[0] = 14963;
      *v18 = v19;
      *(_QWORD *)(v3 + 24) += 19LL;
    }
    v20 = *(_QWORD *)(a1 + 40);
    v21 = *(unsigned int **)(v20 + 360);
    v44 = *(unsigned int **)(v20 + 368);
    if ( v21 != v44 )
    {
      while ( 1 )
      {
        v22 = *v21;
        v23 = (__int64 *)v48;
        sub_1F4AA00(v48, v22, v15, 0, 0);
        if ( !v49 )
          break;
        v50(v48, v3);
        if ( v49 )
          v49(v48, v48, 3);
        if ( v21[1] )
        {
          v39 = *(_DWORD **)(v3 + 24);
          if ( *(_QWORD *)(v3 + 16) - (_QWORD)v39 <= 3u )
          {
            v40 = sub_16E7EE0(v3, " in ", 4u);
          }
          else
          {
            *v39 = 544106784;
            v40 = v3;
            *(_QWORD *)(v3 + 24) += 4LL;
          }
          v22 = v21[1];
          v23 = v51;
          v43 = v40;
          sub_1F4AA00(v51, v22, v15, 0, 0);
          if ( !v52 )
            break;
          v53(v51, v43);
          if ( v52 )
            v52(v51, v51, 3);
        }
        v21 += 2;
        if ( v44 == v21 )
          goto LABEL_25;
        v25 = *(_WORD **)(v3 + 24);
        if ( *(_QWORD *)(v3 + 16) - (_QWORD)v25 <= 1u )
        {
          sub_16E7EE0(v3, ", ", 2u);
        }
        else
        {
          *v25 = 8236;
          *(_QWORD *)(v3 + 24) += 2LL;
        }
      }
      sub_4263D6(v23, v22, v24);
    }
LABEL_25:
    v26 = *(_BYTE **)(v3 + 24);
    if ( (unsigned __int64)v26 >= *(_QWORD *)(v3 + 16) )
    {
      sub_16E7DE0(v3, 10);
    }
    else
    {
      *(_QWORD *)(v3 + 24) = v26 + 1;
      *v26 = 10;
    }
  }
  sub_154BA10((__int64)v51, *(_QWORD *)(*(_QWORD *)a1 + 40LL), 1);
  sub_154C150((__int64)v51, *(_QWORD *)a1);
  for ( i = *(_QWORD *)(a1 + 328); a1 + 320 != i; i = *(_QWORD *)(i + 8) )
  {
    v28 = *(_BYTE **)(v3 + 24);
    if ( (unsigned __int64)v28 < *(_QWORD *)(v3 + 16) )
    {
      *(_QWORD *)(v3 + 24) = v28 + 1;
      *v28 = 10;
    }
    else
    {
      sub_16E7DE0(v3, 10);
    }
    sub_1DD77E0(i, v3, (__int64)v51, a3, 1u);
  }
  v29 = *(__m128i **)(v3 + 24);
  if ( *(_QWORD *)(v3 + 16) - (_QWORD)v29 <= 0x20u )
  {
    v3 = sub_16E7EE0(v3, "\n# End machine code for function ", 0x21u);
  }
  else
  {
    v30 = _mm_load_si128((const __m128i *)&xmmword_42EB040);
    v29[2].m128i_i8[0] = 32;
    *v29 = v30;
    v29[1] = _mm_load_si128((const __m128i *)&xmmword_42EB050);
    *(_QWORD *)(v3 + 24) += 33LL;
  }
  v31 = sub_1E0A440((__int64 *)a1);
  v33 = *(_BYTE **)(v3 + 24);
  v34 = (char *)v31;
  v35 = *(_QWORD *)(v3 + 16) - (_QWORD)v33;
  if ( v32 > v35 )
  {
    v42 = sub_16E7EE0(v3, v34, v32);
    v33 = *(_BYTE **)(v42 + 24);
    v3 = v42;
    v35 = *(_QWORD *)(v42 + 16) - (_QWORD)v33;
  }
  else if ( v32 )
  {
    v47 = v32;
    memcpy(v33, v34, v32);
    v33 = (_BYTE *)(v47 + *(_QWORD *)(v3 + 24));
    v38 = *(_QWORD *)(v3 + 16) - (_QWORD)v33;
    *(_QWORD *)(v3 + 24) = v33;
    if ( v38 > 2 )
      goto LABEL_38;
LABEL_43:
    sub_16E7EE0(v3, ".\n\n", 3u);
    return sub_154BA40(v51);
  }
  if ( v35 <= 2 )
    goto LABEL_43;
LABEL_38:
  v33[2] = 10;
  *(_WORD *)v33 = 2606;
  *(_QWORD *)(v3 + 24) += 3LL;
  return sub_154BA40(v51);
}
