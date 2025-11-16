// Function: sub_CCEEA0
// Address: 0xcceea0
//
__int64 __fastcall sub_CCEEA0(__int64 a1, __int64 a2)
{
  char v2; // al
  _BOOL8 v3; // rcx
  char v4; // al
  _BOOL8 v5; // rcx
  char v6; // al
  _BOOL8 v7; // rcx
  char v8; // al
  _BOOL8 v9; // rcx
  char v10; // al
  _BOOL8 v11; // rcx
  __int64 v12; // rax
  char v13; // al
  _BOOL8 v14; // rcx
  char v15; // al
  _BOOL8 v16; // rcx
  char v17; // al
  __int64 v18; // rcx
  char v19; // al
  __int64 v20; // rcx
  __int64 result; // rax
  __m128i v22; // xmm2
  __int64 v23; // rax
  char v24; // [rsp+Eh] [rbp-72h] BYREF
  char v25; // [rsp+Fh] [rbp-71h] BYREF
  __int64 v26; // [rsp+10h] [rbp-70h] BYREF
  __int64 v27; // [rsp+18h] [rbp-68h] BYREF
  __m128i v28; // [rsp+20h] [rbp-60h] BYREF
  __m128i v29; // [rsp+30h] [rbp-50h] BYREF
  __int64 v30; // [rsp+40h] [rbp-40h]

  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v3 = 0;
  if ( v2 )
    v3 = *(_DWORD *)a2 == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "OptimizerConstBank",
         0,
         v3,
         &v27,
         &v28) )
  {
    sub_CCC2C0(a1, (unsigned int *)a2);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v28.m128i_i64[0]);
  }
  else if ( (_BYTE)v27 )
  {
    *(_DWORD *)a2 = 0;
  }
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v5 = 0;
  if ( v4 )
    v5 = *(_DWORD *)(a2 + 4) == 1;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "DriverConstBank",
         0,
         v5,
         &v27,
         &v28) )
  {
    sub_CCC2C0(a1, (unsigned int *)(a2 + 4));
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v28.m128i_i64[0]);
  }
  else if ( (_BYTE)v27 )
  {
    *(_DWORD *)(a2 + 4) = 1;
  }
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v7 = 0;
  if ( v6 )
    v7 = *(_DWORD *)(a2 + 8) == 1;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "BindlessTextureBank",
         0,
         v7,
         &v27,
         &v28) )
  {
    sub_CCC2C0(a1, (unsigned int *)(a2 + 8));
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v28.m128i_i64[0]);
  }
  else if ( (_BYTE)v27 )
  {
    *(_DWORD *)(a2 + 8) = 1;
  }
  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v9 = 0;
  if ( v8 && !*(_DWORD *)(a2 + 16) && !*(_QWORD *)(a2 + 24) )
    v9 = *(_DWORD *)(a2 + 32) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "LocalMemoryWindow",
         0,
         v9,
         &v27,
         &v28) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
    sub_CCE4B0(a1, a2 + 16);
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v28.m128i_i64[0]);
  }
  else if ( (_BYTE)v27 )
  {
    *(_DWORD *)(a2 + 16) = 0;
    *(_QWORD *)(a2 + 24) = 0;
    *(_DWORD *)(a2 + 32) = 0;
  }
  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v11 = 0;
  if ( v10 && !*(_DWORD *)(a2 + 40) && !*(_QWORD *)(a2 + 48) )
    v11 = *(_DWORD *)(a2 + 56) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int64 *, __m128i *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "SharedMemoryWindow",
         0,
         v11,
         &v27,
         &v28) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
    sub_CCE4B0(a1, a2 + 40);
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v28.m128i_i64[0]);
  }
  else if ( (_BYTE)v27 )
  {
    *(_DWORD *)(a2 + 40) = 0;
    *(_QWORD *)(a2 + 48) = 0;
    *(_DWORD *)(a2 + 56) = 0;
  }
  v12 = *(_QWORD *)a1;
  v30 = 0;
  v28 = 0;
  v29 = 0;
  v13 = (*(__int64 (__fastcall **)(__int64))(v12 + 16))(a1);
  v14 = 0;
  if ( v13
    && !*(_DWORD *)(a2 + 64)
    && !*(_DWORD *)(a2 + 68)
    && !*(_DWORD *)(a2 + 72)
    && !*(_DWORD *)(a2 + 76)
    && !*(_DWORD *)(a2 + 80)
    && !*(_DWORD *)(a2 + 84)
    && !*(_DWORD *)(a2 + 88)
    && !*(_DWORD *)(a2 + 92)
    && !*(_DWORD *)(a2 + 96)
    && (*(_BYTE *)(a2 + 100) & 0xF) == 0 )
  {
    v14 = (*(_DWORD *)(a2 + 100) & 0xFFFFFFF0) == 0;
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int64 *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "ShaderConstIface",
         0,
         v14,
         &v26,
         &v27) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
    sub_CCE780(a1, a2 + 64);
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v27);
  }
  else if ( (_BYTE)v26 )
  {
    v22 = _mm_loadu_si128(&v29);
    v23 = v30;
    *(__m128i *)(a2 + 64) = _mm_loadu_si128(&v28);
    *(_QWORD *)(a2 + 96) = v23;
    *(__m128i *)(a2 + 80) = v22;
  }
  v15 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v16 = 0;
  if ( v15 && !*(_DWORD *)(a2 + 104) && !*(_DWORD *)(a2 + 108) )
    v16 = *(_DWORD *)(a2 + 112) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "VectorizeAndRemapTLD",
         0,
         v16,
         &v24,
         &v26) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
    if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
           a1,
           "Enabled",
           1,
           0,
           &v25,
           &v27) )
    {
      sub_CCC2C0(a1, (unsigned int *)(a2 + 104));
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v27);
    }
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
           a1,
           "NewTsPtrStart",
           1,
           0,
           &v25,
           &v27) )
    {
      sub_CCC2C0(a1, (unsigned int *)(a2 + 108));
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v27);
    }
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
           a1,
           "NewTsPtrEnd",
           1,
           0,
           &v25,
           &v27) )
    {
      sub_CCC2C0(a1, (unsigned int *)(a2 + 112));
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v27);
    }
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v26);
  }
  else if ( v24 )
  {
    *(_QWORD *)(a2 + 104) = 0;
    *(_DWORD *)(a2 + 112) = 0;
  }
  v17 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v18 = 0;
  if ( v17 )
    v18 = *(_BYTE *)(a2 + 116) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, __int64 *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "ELFControlsDCI",
         0,
         v18,
         &v26,
         &v27) )
  {
    sub_CCCDD0(a1, (_BYTE *)(a2 + 116));
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v27);
  }
  else if ( (_BYTE)v26 )
  {
    *(_BYTE *)(a2 + 116) = 0;
  }
  v19 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v20 = 0;
  if ( v19 )
    v20 = *(_BYTE *)(a2 + 117) ^ 1u;
  result = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, __int64, __int64 *, __int64 *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "DiscardDefaultValueOutputs",
             0,
             v20,
             &v26,
             &v27);
  if ( (_BYTE)result )
  {
    sub_CCCDD0(a1, (_BYTE *)(a2 + 117));
    return (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v27);
  }
  else if ( (_BYTE)v26 )
  {
    *(_BYTE *)(a2 + 117) = 0;
  }
  return result;
}
