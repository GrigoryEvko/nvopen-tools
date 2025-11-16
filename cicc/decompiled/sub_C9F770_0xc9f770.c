// Function: sub_C9F770
// Address: 0xc9f770
//
int __fastcall sub_C9F770(__int64 a1, const __m128i *a2)
{
  pthread_mutex_t *v2; // r13
  unsigned int v3; // eax
  __int64 *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( !qword_4F84F60 )
    sub_C7D570(&qword_4F84F60, sub_CA0780, (__int64)sub_C9FD10);
  v2 = (pthread_mutex_t *)(qword_4F84F60 + 664);
  if ( &_pthread_key_create )
  {
    v3 = pthread_mutex_lock((pthread_mutex_t *)(qword_4F84F60 + 664));
    if ( v3 )
      sub_4264C5(v3);
  }
  if ( a2[9].m128i_i8[1] )
    sub_C9F6B0((__m128i **)(a1 + 72), a2, (__int64)a2[5].m128i_i64, (__int64)a2[7].m128i_i64);
  v4 = (__int64 *)a2[10].m128i_i64[0];
  v5 = a2[10].m128i_i64[1];
  a2[9].m128i_i64[1] = 0;
  *v4 = v5;
  v6 = a2[10].m128i_i64[1];
  if ( v6 )
    *(_QWORD *)(v6 + 160) = a2[10].m128i_i64[0];
  if ( *(_QWORD *)(a1 + 64)
    || (v6 = *(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80) == v6)
    || (sub_C9DED0(v11), LODWORD(v6) = sub_C9EA90(a1, (_QWORD *)v11[0], v7, v8, v9), !v11[0]) )
  {
    if ( !&_pthread_key_create )
      return v6;
  }
  else
  {
    LODWORD(v6) = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11[0] + 8LL))(v11[0]);
    if ( !&_pthread_key_create )
      return v6;
  }
  LODWORD(v6) = pthread_mutex_unlock(v2);
  return v6;
}
