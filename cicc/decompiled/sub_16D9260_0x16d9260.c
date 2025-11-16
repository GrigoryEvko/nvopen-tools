// Function: sub_16D9260
// Address: 0x16d9260
//
char __fastcall sub_16D9260(_QWORD *a1, __int64 (*a2)(), __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *v6; // rbx
  __int64 v7; // r13
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r14
  char result; // al
  __int64 v16[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = (const __m128i *)a2;
  if ( !qword_4FA1610 )
  {
    a2 = sub_160CFB0;
    sub_16C1EA0((__int64)&qword_4FA1610, sub_160CFB0, (__int64)sub_160D0B0, a4, a5, a6);
  }
  v7 = qword_4FA1610;
  if ( (unsigned __int8)sub_16D5D40() )
  {
    sub_16C30C0((pthread_mutex_t **)v7);
    if ( !v6[8].m128i_i8[1] )
      goto LABEL_5;
  }
  else
  {
    ++*(_DWORD *)(v7 + 8);
    if ( !v6[8].m128i_i8[1] )
      goto LABEL_5;
  }
  a2 = (__int64 (*)())v6;
  sub_16D91B0((__int64)(a1 + 9), v6, (__int64)v6[4].m128i_i64, (__int64)v6[6].m128i_i64);
LABEL_5:
  v11 = (__int64 *)v6[9].m128i_i64[0];
  v12 = v6[9].m128i_i64[1];
  v6[8].m128i_i64[1] = 0;
  *v11 = v12;
  v13 = v6[9].m128i_i64[1];
  if ( v13 )
  {
    v12 = v6[9].m128i_i64[0];
    *(_QWORD *)(v13 + 144) = v12;
  }
  if ( !a1[8] && a1[9] != a1[10] )
  {
    sub_16D7600(v16, (__int64)a2, v12, v8, v9, v10);
    v14 = v16[0];
    sub_16D80D0((__int64)a1, v16[0]);
    if ( v14 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
  }
  result = sub_16D5D40();
  if ( result )
    return sub_16C30E0((pthread_mutex_t **)v7);
  --*(_DWORD *)(v7 + 8);
  return result;
}
