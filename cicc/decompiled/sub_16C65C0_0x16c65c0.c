// Function: sub_16C65C0
// Address: 0x16c65c0
//
char __fastcall sub_16C65C0(__int64 *a1, __int64 (*a2)(), __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 (*v6)(); // r12
  __int64 v7; // r13
  const char *v8; // rdi
  const __m128i *v9; // rsi
  __m128i *v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r14
  __int64 *v15; // rdi
  const char *v16; // rdi
  char result; // al
  __int64 v18; // rax
  void *v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rax
  __m128i si128; // xmm0
  const __m128i *v26; // [rsp+0h] [rbp-40h] BYREF
  __int64 v27; // [rsp+8h] [rbp-38h]
  _QWORD v28[6]; // [rsp+10h] [rbp-30h] BYREF

  v6 = a2;
  if ( !qword_4FA0510 )
  {
    a2 = sub_160CFB0;
    a1 = &qword_4FA0510;
    sub_16C1EA0((__int64)&qword_4FA0510, sub_160CFB0, (__int64)sub_160D0B0, a4, a5, a6);
  }
  v7 = qword_4FA0510;
  if ( (unsigned __int8)sub_16D5D40(a1, a2, a3, a4, a5, a6) )
    sub_16C30C0((pthread_mutex_t **)v7);
  else
    ++*(_DWORD *)(v7 + 8);
  v8 = *(const char **)v6;
  v9 = (const __m128i *)&v26;
  LOBYTE(v28[0]) = 0;
  v26 = (const __m128i *)v28;
  v27 = 0;
  if ( (_UNKNOWN *)sub_16F12D0(v8, &v26) == &unk_4FA17DA )
  {
    v18 = sub_16E8CB0(v8, &v26, v10);
    v19 = *(void **)(v18 + 24);
    v20 = v18;
    if ( *(_QWORD *)(v18 + 16) - (_QWORD)v19 <= 0xEu )
    {
      v20 = sub_16E7EE0(v18, "Error opening '", 15);
    }
    else
    {
      qmemcpy(v19, "Error opening '", 15);
      *(_QWORD *)(v18 + 24) += 15LL;
    }
    v21 = sub_16E7EE0(v20, *(const char **)v6, *((_QWORD *)v6 + 1));
    v22 = *(_QWORD *)(v21 + 24);
    v23 = v21;
    if ( (unsigned __int64)(*(_QWORD *)(v21 + 16) - v22) <= 2 )
    {
      v23 = sub_16E7EE0(v21, "': ", 3);
    }
    else
    {
      *(_BYTE *)(v22 + 2) = 32;
      *(_WORD *)v22 = 14887;
      *(_QWORD *)(v21 + 24) += 3LL;
    }
    v9 = v26;
    v24 = sub_16E7EE0(v23, v26->m128i_i8, v27);
    v10 = *(__m128i **)(v24 + 24);
    if ( *(_QWORD *)(v24 + 16) - (_QWORD)v10 <= 0x19u )
    {
      v9 = (const __m128i *)"\n  -load request ignored.\n";
      sub_16E7EE0(v24, "\n  -load request ignored.\n", 26);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42AEF60);
      qmemcpy(&v10[1], " ignored.\n", 10);
      *v10 = si128;
      *(_QWORD *)(v24 + 24) += 26LL;
    }
  }
  else
  {
    if ( !qword_4FA0530 )
    {
      v9 = (const __m128i *)sub_16BA4C0;
      sub_16C1EA0((__int64)&qword_4FA0530, (__int64 (*)(void))sub_16BA4C0, (__int64)sub_16BA4F0, v11, v12, v13);
    }
    v14 = qword_4FA0530;
    v15 = *(__int64 **)(qword_4FA0530 + 8);
    if ( v15 == *(__int64 **)(qword_4FA0530 + 16) )
    {
      v9 = *(const __m128i **)(qword_4FA0530 + 8);
      sub_8FD760((__m128i **)qword_4FA0530, v9, (__int64)v6);
    }
    else
    {
      if ( v15 )
      {
        *v15 = (__int64)(v15 + 2);
        v9 = *(const __m128i **)v6;
        sub_16C6510(v15, *(_BYTE **)v6, *(_QWORD *)v6 + *((_QWORD *)v6 + 1));
        v15 = *(__int64 **)(v14 + 8);
      }
      *(_QWORD *)(v14 + 8) = v15 + 4;
    }
  }
  v16 = (const char *)v26;
  if ( v26 != (const __m128i *)v28 )
  {
    v9 = (const __m128i *)(v28[0] + 1LL);
    j_j___libc_free_0(v26, v28[0] + 1LL);
  }
  result = sub_16D5D40(v16, v9, v10, v11, v12, v13);
  if ( result )
    return sub_16C30E0((pthread_mutex_t **)v7);
  --*(_DWORD *)(v7 + 8);
  return result;
}
