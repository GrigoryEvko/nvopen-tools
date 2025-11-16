// Function: sub_262EA90
// Address: 0x262ea90
//
__int64 __fastcall sub_262EA90(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned __int8 v4; // r15
  __int64 result; // rax
  __int64 v6; // rax
  size_t v7; // r15
  unsigned __int64 v8; // rsi
  size_t v9; // rsi
  char v10; // [rsp+7h] [rbp-59h] BYREF
  __int64 v11; // [rsp+8h] [rbp-58h] BYREF
  __m128i v12; // [rsp+10h] [rbp-50h] BYREF
  const void *v13; // [rsp+20h] [rbp-40h] BYREF
  size_t v14; // [rsp+28h] [rbp-38h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _QWORD, __m128i *, const void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Linkage",
         0,
         0,
         &v12,
         &v13) )
  {
    sub_261B850(a1, (unsigned int *)a2);
    (*(void (__fastcall **)(__int64, const void *))(*(_QWORD *)a1 + 128LL))(a1, v13);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __m128i *, const void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Visibility",
         0,
         0,
         &v12,
         &v13) )
  {
    sub_261B850(a1, (unsigned int *)(a2 + 4));
    (*(void (__fastcall **)(__int64, const void *))(*(_QWORD *)a1 + 128LL))(a1, v13);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __m128i *, const void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "NotEligibleToImport",
         0,
         0,
         &v12,
         &v13) )
  {
    sub_261B670(a1, (_BYTE *)(a2 + 8));
    (*(void (__fastcall **)(__int64, const void *))(*(_QWORD *)a1 + 128LL))(a1, v13);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _QWORD, __m128i *, const void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Live",
         0,
         0,
         &v12,
         &v13) )
  {
    sub_261B670(a1, (_BYTE *)(a2 + 9));
    (*(void (__fastcall **)(__int64, const void *))(*(_QWORD *)a1 + 128LL))(a1, v13);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _QWORD, __m128i *, const void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Local",
         0,
         0,
         &v12,
         &v13) )
  {
    sub_261B670(a1, (_BYTE *)(a2 + 10));
    (*(void (__fastcall **)(__int64, const void *))(*(_QWORD *)a1 + 128LL))(a1, v13);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __m128i *, const void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "CanAutoHide",
         0,
         0,
         &v12,
         &v13) )
  {
    sub_261B670(a1, (_BYTE *)(a2 + 11));
    (*(void (__fastcall **)(__int64, const void *))(*(_QWORD *)a1 + 128LL))(a1, v13);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __m128i *, const void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "ImportType",
         0,
         0,
         &v12,
         &v13) )
  {
    sub_261B850(a1, (unsigned int *)(a2 + 12));
    (*(void (__fastcall **)(__int64, const void *))(*(_QWORD *)a1 + 128LL))(a1, v13);
  }
  v3 = *(_QWORD *)a1;
  v12.m128i_i64[1] = 0;
  v10 = 1;
  v4 = (*(__int64 (__fastcall **)(__int64))(v3 + 16))(a1);
  if ( v4 )
    v4 = *(_BYTE *)(a2 + 24) ^ 1;
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    if ( !*(_BYTE *)(a2 + 24) )
      goto LABEL_19;
  }
  else if ( !*(_BYTE *)(a2 + 24) )
  {
    *(_QWORD *)(a2 + 16) = 0;
    *(_BYTE *)(a2 + 24) = 1;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "Aliasee",
          0,
          v4,
          &v10,
          &v11) )
  {
LABEL_19:
    if ( v10 )
      *(__m128i *)(a2 + 16) = _mm_loadu_si128(&v12);
    goto LABEL_21;
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    goto LABEL_55;
  v6 = sub_CB1000(a1);
  if ( *(_DWORD *)(v6 + 32) != 1 )
    goto LABEL_55;
  v7 = *(_QWORD *)(v6 + 80);
  v13 = *(const void **)(v6 + 72);
  v14 = v7;
  v8 = sub_C93710(&v13, 32, 0xFFFFFFFFFFFFFFFFLL) + 1;
  if ( v8 > v14 )
    v8 = v14;
  v9 = v14 - v7 + v8;
  if ( v9 > v14 )
    v9 = v14;
  if ( sub_9691B0(v13, v9, "<none>", 6) )
    *(__m128i *)(a2 + 16) = _mm_loadu_si128(&v12);
  else
LABEL_55:
    sub_261BC10(a1, (_QWORD *)(a2 + 16));
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v11);
LABEL_21:
  if ( (!(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1)
     || *(_QWORD *)(a2 + 40) != *(_QWORD *)(a2 + 32))
    && (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __m128i *, const void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Refs",
         0,
         0,
         &v12,
         &v13) )
  {
    sub_262E1B0(a1, (__int64 *)(a2 + 32));
    (*(void (__fastcall **)(__int64, const void *))(*(_QWORD *)a1 + 128LL))(a1, v13);
  }
  if ( (!(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1)
     || *(_QWORD *)(a2 + 64) != *(_QWORD *)(a2 + 56))
    && (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __m128i *, const void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "TypeTests",
         0,
         0,
         &v12,
         &v13) )
  {
    sub_262E1B0(a1, (__int64 *)(a2 + 56));
    (*(void (__fastcall **)(__int64, const void *))(*(_QWORD *)a1 + 128LL))(a1, v13);
  }
  if ( (!(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1)
     || *(_QWORD *)(a2 + 88) != *(_QWORD *)(a2 + 80))
    && (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __m128i *, const void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "TypeTestAssumeVCalls",
         0,
         0,
         &v12,
         &v13) )
  {
    sub_262E490(a1, (__int64 *)(a2 + 80));
    (*(void (__fastcall **)(__int64, const void *))(*(_QWORD *)a1 + 128LL))(a1, v13);
  }
  if ( (!(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1)
     || *(_QWORD *)(a2 + 112) != *(_QWORD *)(a2 + 104))
    && (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __m128i *, const void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "TypeCheckedLoadVCalls",
         0,
         0,
         &v12,
         &v13) )
  {
    sub_262E490(a1, (__int64 *)(a2 + 104));
    (*(void (__fastcall **)(__int64, const void *))(*(_QWORD *)a1 + 128LL))(a1, v13);
  }
  if ( (!(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1)
     || *(_QWORD *)(a2 + 136) != *(_QWORD *)(a2 + 128))
    && (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __m128i *, const void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "TypeTestAssumeConstVCalls",
         0,
         0,
         &v12,
         &v13) )
  {
    sub_262E810(a1, (__int64 *)(a2 + 128));
    (*(void (__fastcall **)(__int64, const void *))(*(_QWORD *)a1 + 128LL))(a1, v13);
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1)
    || (result = *(_QWORD *)(a2 + 152), *(_QWORD *)(a2 + 160) != result) )
  {
    result = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __m128i *, const void **))(*(_QWORD *)a1 + 120LL))(
               a1,
               "TypeCheckedLoadConstVCalls",
               0,
               0,
               &v12,
               &v13);
    if ( (_BYTE)result )
    {
      sub_262E810(a1, (__int64 *)(a2 + 152));
      return (*(__int64 (__fastcall **)(__int64, const void *))(*(_QWORD *)a1 + 128LL))(a1, v13);
    }
  }
  return result;
}
