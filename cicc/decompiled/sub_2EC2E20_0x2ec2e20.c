// Function: sub_2EC2E20
// Address: 0x2ec2e20
//
void __fastcall sub_2EC2E20(__int64 *a1)
{
  __int64 (*v1)(); // r13
  __m128i *v2; // rax
  unsigned __int64 v3; // rcx
  __int64 v4; // rax
  _QWORD *v5; // rax
  __m128i *v6; // rdx
  __int64 v7; // rdi
  __m128i si128; // xmm0
  __m128i *v9; // rdx
  __m128i v10; // xmm0
  __int64 v11; // rax
  unsigned __int64 v12[2]; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v13; // [rsp+10h] [rbp-D0h] BYREF
  unsigned __int64 v14[2]; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v15; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int64 v16[2]; // [rsp+40h] [rbp-A0h] BYREF
  __m128i v17; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int64 *v18; // [rsp+60h] [rbp-80h] BYREF
  __int16 v19; // [rsp+80h] [rbp-60h]
  unsigned __int64 *v20; // [rsp+90h] [rbp-50h] BYREF
  __int16 v21; // [rsp+B0h] [rbp-30h]

  v1 = *(__int64 (**)())(*a1 + 16);
  (*(void (__fastcall **)(unsigned __int64 *, __int64 *))(*a1 + 56))(v14, a1);
  v2 = (__m128i *)sub_2241130(v14, 0, 0, "Scheduling-Units Graph for ", 0x1Bu);
  v16[0] = (unsigned __int64)&v17;
  if ( (__m128i *)v2->m128i_i64[0] == &v2[1] )
  {
    v17 = _mm_loadu_si128(v2 + 1);
  }
  else
  {
    v16[0] = v2->m128i_i64[0];
    v17.m128i_i64[0] = v2[1].m128i_i64[0];
  }
  v3 = v2->m128i_u64[1];
  v2[1].m128i_i8[0] = 0;
  v16[1] = v3;
  v2->m128i_i64[0] = (__int64)v2[1].m128i_i64;
  v2->m128i_i64[1] = 0;
  v20 = v16;
  v4 = *a1;
  v21 = 260;
  (*(void (__fastcall **)(unsigned __int64 *, __int64 *))(v4 + 56))(v12, a1);
  v18 = v12;
  v19 = 260;
  if ( v1 == sub_2EC29E0 )
  {
    v5 = sub_CB72A0();
    v6 = (__m128i *)v5[4];
    v7 = (__int64)v5;
    if ( v5[3] - (_QWORD)v6 <= 0x3Du )
    {
      v11 = sub_CB6200((__int64)v5, "ScheduleDAGMI::viewGraph is only available in debug builds on ", 0x3Eu);
      v9 = *(__m128i **)(v11 + 32);
      v7 = v11;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42EC320);
      qmemcpy(&v6[3], "bug builds on ", 14);
      *v6 = si128;
      v6[1] = _mm_load_si128((const __m128i *)&xmmword_42EC330);
      v6[2] = _mm_load_si128((const __m128i *)&xmmword_42EAFD0);
      v9 = (__m128i *)(v5[4] + 62LL);
      v5[4] = v9;
    }
    if ( *(_QWORD *)(v7 + 24) - (_QWORD)v9 <= 0x1Cu )
    {
      sub_CB6200(v7, "systems with Graphviz or gv!\n", 0x1Du);
    }
    else
    {
      v10 = _mm_load_si128(xmmword_42EAFE0);
      qmemcpy(&v9[1], "phviz or gv!\n", 13);
      *v9 = v10;
      *(_QWORD *)(v7 + 32) += 29LL;
    }
  }
  else
  {
    ((void (__fastcall *)(__int64 *, unsigned __int64 **, unsigned __int64 **))v1)(a1, &v18, &v20);
  }
  if ( (__int64 *)v12[0] != &v13 )
    j_j___libc_free_0(v12[0]);
  if ( (__m128i *)v16[0] != &v17 )
    j_j___libc_free_0(v16[0]);
  if ( (__int64 *)v14[0] != &v15 )
    j_j___libc_free_0(v14[0]);
}
