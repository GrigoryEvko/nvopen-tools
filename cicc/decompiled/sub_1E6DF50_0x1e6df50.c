// Function: sub_1E6DF50
// Address: 0x1e6df50
//
_QWORD *__fastcall sub_1E6DF50(__int64 *a1)
{
  __int64 (*v1)(); // r13
  __m128i *v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rax
  _QWORD *v5; // rax
  __m128i *v6; // rdx
  __int64 v7; // rdi
  __m128i si128; // xmm0
  __m128i *v9; // rdx
  __m128i v10; // xmm0
  _QWORD *result; // rax
  __int64 v12; // rax
  _QWORD *v13; // [rsp+0h] [rbp-C0h] BYREF
  __int16 v14; // [rsp+10h] [rbp-B0h]
  _QWORD *v15; // [rsp+20h] [rbp-A0h] BYREF
  __int16 v16; // [rsp+30h] [rbp-90h]
  _QWORD v17[2]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v18; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v19[2]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v20[2]; // [rsp+70h] [rbp-50h] BYREF
  _QWORD v21[2]; // [rsp+80h] [rbp-40h] BYREF
  _OWORD v22[3]; // [rsp+90h] [rbp-30h] BYREF

  v1 = *(__int64 (**)())(*a1 + 16);
  (*(void (__fastcall **)(_QWORD *, __int64 *))(*a1 + 48))(v19, a1);
  v2 = (__m128i *)sub_2241130(v19, 0, 0, "Scheduling-Units Graph for ", 27);
  v21[0] = v22;
  if ( (__m128i *)v2->m128i_i64[0] == &v2[1] )
  {
    v22[0] = _mm_loadu_si128(v2 + 1);
  }
  else
  {
    v21[0] = v2->m128i_i64[0];
    *(_QWORD *)&v22[0] = v2[1].m128i_i64[0];
  }
  v3 = v2->m128i_i64[1];
  v2[1].m128i_i8[0] = 0;
  v21[1] = v3;
  v2->m128i_i64[0] = (__int64)v2[1].m128i_i64;
  v2->m128i_i64[1] = 0;
  v15 = v21;
  v4 = *a1;
  v16 = 260;
  (*(void (__fastcall **)(_QWORD *, __int64 *))(v4 + 48))(v17, a1);
  v13 = v17;
  v14 = 260;
  if ( v1 == sub_1E6D7C0 )
  {
    v5 = sub_16E8CB0();
    v6 = (__m128i *)v5[3];
    v7 = (__int64)v5;
    if ( v5[2] - (_QWORD)v6 <= 0x3Du )
    {
      v12 = sub_16E7EE0((__int64)v5, "ScheduleDAGMI::viewGraph is only available in debug builds on ", 0x3Eu);
      v9 = *(__m128i **)(v12 + 24);
      v7 = v12;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42EC320);
      qmemcpy(&v6[3], "bug builds on ", 14);
      *v6 = si128;
      v6[1] = _mm_load_si128((const __m128i *)&xmmword_42EC330);
      v6[2] = _mm_load_si128((const __m128i *)&xmmword_42EAFD0);
      v9 = (__m128i *)(v5[3] + 62LL);
      v5[3] = v9;
    }
    if ( *(_QWORD *)(v7 + 16) - (_QWORD)v9 <= 0x1Cu )
    {
      sub_16E7EE0(v7, "systems with Graphviz or gv!\n", 0x1Du);
    }
    else
    {
      v10 = _mm_load_si128(xmmword_42EAFE0);
      qmemcpy(&v9[1], "phviz or gv!\n", 13);
      *v9 = v10;
      *(_QWORD *)(v7 + 24) += 29LL;
    }
  }
  else
  {
    ((void (__fastcall *)(__int64 *, _QWORD **, _QWORD **))v1)(a1, &v13, &v15);
  }
  if ( (__int64 *)v17[0] != &v18 )
    j_j___libc_free_0(v17[0], v18 + 1);
  if ( (_OWORD *)v21[0] != v22 )
    j_j___libc_free_0(v21[0], *(_QWORD *)&v22[0] + 1LL);
  result = v20;
  if ( (_QWORD *)v19[0] != v20 )
    return (_QWORD *)j_j___libc_free_0(v19[0], v20[0] + 1LL);
  return result;
}
