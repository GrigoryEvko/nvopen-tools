// Function: sub_2E823F0
// Address: 0x2e823f0
//
__int64 (__fastcall *__fastcall sub_2E823F0(__int64 a1, __int64 a2, __int64 a3))(_QWORD *, _QWORD *, __int64)
{
  __int64 v3; // r13
  __m128i *v5; // rdx
  __m128i si128; // xmm0
  __int64 v7; // r14
  const char *v8; // rax
  size_t v9; // rdx
  _WORD *v10; // rdi
  unsigned __int8 *v11; // rsi
  unsigned __int64 v12; // rax
  _BYTE *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r15
  __int64 v16; // rax
  __m128i *v17; // rdx
  __m128i v18; // xmm0
  __int64 v19; // rax
  unsigned int *v20; // rbx
  __int64 v21; // rsi
  _BYTE *v22; // rdi
  __int64 v23; // rdx
  _WORD *v24; // rdx
  _BYTE *v25; // rax
  __int64 i; // r15
  _BYTE *v27; // rax
  __m128i *v28; // rdx
  __m128i v29; // xmm0
  const char *v30; // rax
  size_t v31; // rdx
  _BYTE *v32; // rdi
  unsigned __int8 *v33; // rsi
  unsigned __int64 v34; // rax
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // rax
  _DWORD *v38; // rdx
  __int64 v39; // r10
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // [rsp+0h] [rbp-E0h]
  unsigned int *v43; // [rsp+10h] [rbp-D0h]
  size_t v44; // [rsp+10h] [rbp-D0h]
  size_t v46; // [rsp+18h] [rbp-C8h]
  _BYTE v47[16]; // [rsp+20h] [rbp-C0h] BYREF
  void (__fastcall *v48)(_BYTE *, _BYTE *, __int64); // [rsp+30h] [rbp-B0h]
  void (__fastcall *v49)(_BYTE *, __int64); // [rsp+38h] [rbp-A8h]
  _QWORD v50[2]; // [rsp+40h] [rbp-A0h] BYREF
  void (__fastcall *v51)(_QWORD *, _QWORD *, __int64); // [rsp+50h] [rbp-90h]
  void (__fastcall *v52)(_QWORD *, __int64); // [rsp+58h] [rbp-88h]

  v3 = a2;
  v5 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v5 <= 0x1Bu )
  {
    v7 = sub_CB6200(a2, "# Machine code for function ", 0x1Cu);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_42EB020);
    v7 = a2;
    qmemcpy(&v5[1], "or function ", 12);
    *v5 = si128;
    *(_QWORD *)(a2 + 32) += 28LL;
  }
  v8 = sub_2E791E0((__int64 *)a1);
  v10 = *(_WORD **)(v7 + 32);
  v11 = (unsigned __int8 *)v8;
  v12 = *(_QWORD *)(v7 + 24) - (_QWORD)v10;
  if ( v9 > v12 )
  {
    v40 = sub_CB6200(v7, v11, v9);
    v10 = *(_WORD **)(v40 + 32);
    v7 = v40;
    v12 = *(_QWORD *)(v40 + 24) - (_QWORD)v10;
  }
  else if ( v9 )
  {
    v44 = v9;
    memcpy(v10, v11, v9);
    v10 = (_WORD *)(v44 + *(_QWORD *)(v7 + 32));
    v36 = *(_QWORD *)(v7 + 24) - (_QWORD)v10;
    *(_QWORD *)(v7 + 32) = v10;
    if ( v36 > 1 )
      goto LABEL_6;
    goto LABEL_39;
  }
  if ( v12 > 1 )
  {
LABEL_6:
    *v10 = 8250;
    *(_QWORD *)(v7 + 32) += 2LL;
    goto LABEL_7;
  }
LABEL_39:
  sub_CB6200(v7, (unsigned __int8 *)": ", 2u);
LABEL_7:
  sub_2E78A80((_QWORD *)(a1 + 344), v3);
  v13 = *(_BYTE **)(v3 + 32);
  if ( (unsigned __int64)v13 >= *(_QWORD *)(v3 + 24) )
  {
    sub_CB5D20(v3, 10);
  }
  else
  {
    *(_QWORD *)(v3 + 32) = v13 + 1;
    *v13 = 10;
  }
  sub_2E77130(*(_QWORD *)(a1 + 48), a1, v3);
  v14 = *(_QWORD *)(a1 + 64);
  if ( v14 )
    sub_2E79E40(v14, v3);
  sub_2E7A200(*(_QWORD *)(a1 + 56), v3);
  v15 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 16) + 200LL))(*(_QWORD *)(a1 + 16));
  v16 = *(_QWORD *)(a1 + 32);
  if ( v16 && *(_QWORD *)(v16 + 488) != *(_QWORD *)(v16 + 496) )
  {
    v17 = *(__m128i **)(v3 + 32);
    if ( *(_QWORD *)(v3 + 24) - (_QWORD)v17 <= 0x12u )
    {
      sub_CB6200(v3, "Function Live Ins: ", 0x13u);
    }
    else
    {
      v18 = _mm_load_si128((const __m128i *)&xmmword_42EB030);
      v17[1].m128i_i8[2] = 32;
      v17[1].m128i_i16[0] = 14963;
      *v17 = v18;
      *(_QWORD *)(v3 + 32) += 19LL;
    }
    v19 = *(_QWORD *)(a1 + 32);
    v20 = *(unsigned int **)(v19 + 488);
    v43 = *(unsigned int **)(v19 + 496);
    if ( v20 != v43 )
    {
      while ( 1 )
      {
        v21 = *v20;
        v22 = v47;
        sub_2FF6320(v47, v21, v15, 0, 0);
        if ( !v48 )
          break;
        v49(v47, v3);
        if ( v48 )
          v48(v47, v47, 3);
        if ( v20[1] )
        {
          v38 = *(_DWORD **)(v3 + 32);
          if ( *(_QWORD *)(v3 + 24) - (_QWORD)v38 <= 3u )
          {
            v39 = sub_CB6200(v3, (unsigned __int8 *)" in ", 4u);
          }
          else
          {
            *v38 = 544106784;
            v39 = v3;
            *(_QWORD *)(v3 + 32) += 4LL;
          }
          v21 = v20[1];
          v22 = v50;
          v42 = v39;
          sub_2FF6320(v50, v21, v15, 0, 0);
          if ( !v51 )
            break;
          v52(v50, v42);
          if ( v51 )
            v51(v50, v50, 3);
        }
        v20 += 2;
        if ( v43 == v20 )
          goto LABEL_23;
        v24 = *(_WORD **)(v3 + 32);
        if ( *(_QWORD *)(v3 + 24) - (_QWORD)v24 <= 1u )
        {
          sub_CB6200(v3, (unsigned __int8 *)", ", 2u);
        }
        else
        {
          *v24 = 8236;
          *(_QWORD *)(v3 + 32) += 2LL;
        }
      }
      sub_4263D6(v22, v21, v23);
    }
LABEL_23:
    v25 = *(_BYTE **)(v3 + 32);
    if ( (unsigned __int64)v25 >= *(_QWORD *)(v3 + 24) )
    {
      sub_CB5D20(v3, 10);
    }
    else
    {
      *(_QWORD *)(v3 + 32) = v25 + 1;
      *v25 = 10;
    }
  }
  sub_A558A0((__int64)v50, *(_QWORD *)(*(_QWORD *)a1 + 40LL), 1);
  sub_A564B0((__int64)v50, *(_QWORD *)a1);
  for ( i = *(_QWORD *)(a1 + 328); a1 + 320 != i; i = *(_QWORD *)(i + 8) )
  {
    v27 = *(_BYTE **)(v3 + 32);
    if ( (unsigned __int64)v27 < *(_QWORD *)(v3 + 24) )
    {
      *(_QWORD *)(v3 + 32) = v27 + 1;
      *v27 = 10;
    }
    else
    {
      sub_CB5D20(v3, 10);
    }
    sub_2E38390(i, v3, (__int64)v50, a3, 1u);
  }
  v28 = *(__m128i **)(v3 + 32);
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v28 <= 0x20u )
  {
    v3 = sub_CB6200(v3, "\n# End machine code for function ", 0x21u);
  }
  else
  {
    v29 = _mm_load_si128((const __m128i *)&xmmword_42EB040);
    v28[2].m128i_i8[0] = 32;
    *v28 = v29;
    v28[1] = _mm_load_si128((const __m128i *)&xmmword_42EB050);
    *(_QWORD *)(v3 + 32) += 33LL;
  }
  v30 = sub_2E791E0((__int64 *)a1);
  v32 = *(_BYTE **)(v3 + 32);
  v33 = (unsigned __int8 *)v30;
  v34 = *(_QWORD *)(v3 + 24) - (_QWORD)v32;
  if ( v31 > v34 )
  {
    v41 = sub_CB6200(v3, v33, v31);
    v32 = *(_BYTE **)(v41 + 32);
    v3 = v41;
    v34 = *(_QWORD *)(v41 + 24) - (_QWORD)v32;
  }
  else if ( v31 )
  {
    v46 = v31;
    memcpy(v32, v33, v31);
    v32 = (_BYTE *)(v46 + *(_QWORD *)(v3 + 32));
    v37 = *(_QWORD *)(v3 + 24) - (_QWORD)v32;
    *(_QWORD *)(v3 + 32) = v32;
    if ( v37 > 2 )
      goto LABEL_36;
LABEL_41:
    v33 = (unsigned __int8 *)".\n\n";
    sub_CB6200(v3, (unsigned __int8 *)".\n\n", 3u);
    return sub_A55520(v50, (__int64)v33);
  }
  if ( v34 <= 2 )
    goto LABEL_41;
LABEL_36:
  v32[2] = 10;
  *(_WORD *)v32 = 2606;
  *(_QWORD *)(v3 + 32) += 3LL;
  return sub_A55520(v50, (__int64)v33);
}
