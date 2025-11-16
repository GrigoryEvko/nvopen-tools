// Function: sub_1C23370
// Address: 0x1c23370
//
__int64 __fastcall sub_1C23370(__int64 a1, __int64 a2)
{
  char v2; // al
  _BOOL8 v3; // rcx
  unsigned int v4; // eax
  char v5; // al
  _BOOL8 v6; // rcx
  char v7; // al
  _BOOL8 v8; // rcx
  char v9; // al
  _BOOL8 v10; // rcx
  char v11; // al
  _BOOL8 v12; // rcx
  const __m128i *v13; // rax
  __int64 result; // rax
  __int64 v15; // rax
  __m128i *v16; // r10
  char v17; // al
  __int64 v18; // r10
  __int64 v19; // rax
  unsigned __int8 (__fastcall *v20)(__int64, const char *, _BOOL8); // r14
  char v21; // al
  _BOOL8 v22; // rdx
  unsigned __int8 (__fastcall *v23)(__int64, const char *, _BOOL8); // r14
  char v24; // al
  _BOOL8 v25; // rdx
  unsigned __int8 (__fastcall *v26)(__int64, const char *, _BOOL8); // r14
  char v27; // al
  _BOOL8 v28; // rdx
  __int64 v29; // [rsp+8h] [rbp-178h]
  char v30; // [rsp+17h] [rbp-169h] BYREF
  __int64 v31; // [rsp+18h] [rbp-168h] BYREF
  __int128 v32; // [rsp+20h] [rbp-160h] BYREF
  _BYTE *v33; // [rsp+30h] [rbp-150h] BYREF
  __int64 v34; // [rsp+38h] [rbp-148h]
  _QWORD v35[2]; // [rsp+40h] [rbp-140h] BYREF
  _BYTE v36[128]; // [rsp+50h] [rbp-130h] BYREF
  __m128i v37; // [rsp+D0h] [rbp-B0h]
  __m128i v38; // [rsp+E0h] [rbp-A0h]
  __m128i v39; // [rsp+F0h] [rbp-90h]
  __m128i v40; // [rsp+100h] [rbp-80h]
  __m128i v41; // [rsp+110h] [rbp-70h]
  __m128i v42; // [rsp+120h] [rbp-60h]
  __m128i v43; // [rsp+130h] [rbp-50h]
  __int64 v44; // [rsp+140h] [rbp-40h]

  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v3 = 0;
  if ( v2 && !*(_DWORD *)a2 )
    v3 = *(_DWORD *)(a2 + 4) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, _BYTE **, _BYTE *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Version",
         0,
         v3,
         &v33,
         v36) )
  {
    sub_1C14930(a1, (unsigned int *)a2);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v36);
    v4 = *(_DWORD *)a2;
LABEL_7:
    if ( v4 )
      goto LABEL_8;
    goto LABEL_39;
  }
  if ( !(_BYTE)v33 )
  {
    v4 = *(_DWORD *)a2;
    goto LABEL_7;
  }
  *(_QWORD *)a2 = 0;
LABEL_39:
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, _BYTE **, _BYTE *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Version",
         1,
         0,
         &v33,
         v36) )
  {
    sub_1C14710(a1, (unsigned int *)a2);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v36);
  }
LABEL_8:
  v5 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v6 = 0;
  if ( v5 && !*(_DWORD *)(a2 + 8) )
    v6 = *(_DWORD *)(a2 + 12) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, _BYTE **, _BYTE *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "NvvmIRVersion",
         0,
         v6,
         &v33,
         v36) )
  {
    sub_1C14930(a1, (unsigned int *)(a2 + 8));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v36);
  }
  else if ( (_BYTE)v33 )
  {
    *(_QWORD *)(a2 + 8) = 0;
  }
  v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v8 = 0;
  if ( v7 && !*(_DWORD *)(a2 + 16) )
    v8 = *(_DWORD *)(a2 + 20) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, _BYTE **, _BYTE *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "NvvmDebugVersion",
         0,
         v8,
         &v33,
         v36) )
  {
    sub_1C14930(a1, (unsigned int *)(a2 + 16));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v36);
  }
  else if ( (_BYTE)v33 )
  {
    *(_QWORD *)(a2 + 16) = 0;
  }
  v9 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v10 = 0;
  if ( v9 && !*(_DWORD *)(a2 + 24) )
    v10 = *(_DWORD *)(a2 + 28) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, _BYTE **, _BYTE *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "LlvmVersion",
         0,
         v10,
         &v33,
         v36) )
  {
    sub_1C14930(a1, (unsigned int *)(a2 + 24));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v36);
  }
  else if ( (_BYTE)v33 )
  {
    *(_QWORD *)(a2 + 24) = 0;
  }
  v11 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v12 = 0;
  if ( v11 )
    v12 = *(_DWORD *)(a2 + 32) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, _BYTE **, _BYTE *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "IRLevel",
         0,
         v12,
         &v33,
         v36) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v20 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v21 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v22 = 0;
    if ( v21 )
      v22 = *(_DWORD *)(a2 + 32) == 0;
    if ( v20(a1, "NVVM_IR_LEVEL_UNIFIED_AFTER_DCI", v22) )
      *(_DWORD *)(a2 + 32) = 0;
    v23 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v24 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v25 = 0;
    if ( v24 )
      v25 = *(_DWORD *)(a2 + 32) == 1;
    if ( v23(a1, "NVVM_IR_LEVEL_LTO", v25) )
      *(_DWORD *)(a2 + 32) = 1;
    v26 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v27 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v28 = 0;
    if ( v27 )
      v28 = *(_DWORD *)(a2 + 32) == 2;
    if ( v26(a1, "NVVM_IR_LEVEL_OPTIX", v28) )
      *(_DWORD *)(a2 + 32) = 2;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v36);
  }
  else if ( (_BYTE)v33 )
  {
    *(_DWORD *)(a2 + 32) = 0;
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v13 = *(const __m128i **)(a2 + 80);
    *(__m128i *)v36 = _mm_loadu_si128(v13);
    *(__m128i *)&v36[16] = _mm_loadu_si128(v13 + 1);
    *(__m128i *)&v36[32] = _mm_loadu_si128(v13 + 2);
    *(__m128i *)&v36[48] = _mm_loadu_si128(v13 + 3);
    *(__m128i *)&v36[64] = _mm_loadu_si128(v13 + 4);
    *(__m128i *)&v36[80] = _mm_loadu_si128(v13 + 5);
    *(__m128i *)&v36[96] = _mm_loadu_si128(v13 + 6);
    *(__m128i *)&v36[112] = _mm_loadu_si128(v13 + 7);
    v37 = _mm_loadu_si128(v13 + 8);
    v38 = _mm_loadu_si128(v13 + 9);
    v39 = _mm_loadu_si128(v13 + 10);
    v40 = _mm_loadu_si128(v13 + 11);
    v41 = _mm_loadu_si128(v13 + 12);
    v42 = _mm_loadu_si128(v13 + 13);
    v43 = _mm_loadu_si128(v13 + 14);
    v44 = v13[15].m128i_i64[0];
    if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, __int128 *, _BYTE **))(*(_QWORD *)a1 + 120LL))(
           a1,
           "Options",
           1,
           0,
           &v32,
           &v33) )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
      sub_1C21CE0(a1, (__int64)v36);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
      (*(void (__fastcall **)(__int64, _BYTE *))(*(_QWORD *)a1 + 128LL))(a1, v33);
    }
  }
  else
  {
    v15 = sub_16E4080(a1);
    v16 = (__m128i *)sub_145CBF0(*(__int64 **)(v15 + 8), 248, 8);
    memset(v36, 0, 0x78u);
    v29 = (__int64)v16;
    sub_1C13890(v16, (const __m128i *)v36);
    v17 = (*(__int64 (__fastcall **)(__int64, char *, __int64, _QWORD, __int128 *, _BYTE **))(*(_QWORD *)a1 + 120LL))(
            a1,
            "Options",
            1,
            0,
            &v32,
            &v33);
    v18 = v29;
    if ( v17 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
      sub_1C21CE0(a1, v29);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
      (*(void (__fastcall **)(__int64, _BYTE *))(*(_QWORD *)a1 + 128LL))(a1, v33);
      v18 = v29;
    }
    *(_QWORD *)(a2 + 80) = v18;
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, _BYTE **, _BYTE *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "IsBinary",
         1,
         0,
         &v33,
         v36) )
  {
    sub_1C14360(a1, (_BYTE *)(a2 + 72));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v36);
  }
  result = (*(__int64 (__fastcall **)(__int64, char *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "Module",
             1,
             0,
             &v30,
             &v31);
  if ( (_BYTE)result )
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    {
      v34 = 0;
      v33 = v35;
      LOBYTE(v35[0]) = 0;
      *(_DWORD *)&v36[32] = 1;
      memset(&v36[8], 0, 24);
      *(_QWORD *)v36 = &unk_49EFBE0;
      *(_QWORD *)&v36[40] = &v33;
      sub_16E4080(a1);
      sub_16E7EE0((__int64)v36, *(char **)(a2 + 40), *(_QWORD *)(a2 + 48));
      if ( *(_QWORD *)&v36[24] != *(_QWORD *)&v36[8] )
        sub_16E7BA0((__int64 *)v36);
      v32 = **(_OWORD **)&v36[40];
      (*(void (__fastcall **)(__int64, __int128 *, __int64))(*(_QWORD *)a1 + 216LL))(a1, &v32, 1);
      sub_16E7BC0((__int64 *)v36);
      if ( v33 != (_BYTE *)v35 )
        j_j___libc_free_0(v33, v35[0] + 1LL);
    }
    else
    {
      v19 = *(_QWORD *)a1;
      v33 = 0;
      v34 = 0;
      (*(void (__fastcall **)(__int64, _BYTE **, __int64))(v19 + 216))(a1, &v33, 1);
      sub_16E4080(a1);
      *(_QWORD *)v36 = &v36[16];
      if ( v33 )
      {
        sub_1C13960((__int64 *)v36, v33, (__int64)&v33[v34]);
      }
      else
      {
        *(_QWORD *)&v36[8] = 0;
        v36[16] = 0;
      }
      sub_2240AE0(a2 + 40, v36);
      if ( *(_BYTE **)v36 != &v36[16] )
        j_j___libc_free_0(*(_QWORD *)v36, *(_QWORD *)&v36[16] + 1LL);
    }
    return (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v31);
  }
  return result;
}
