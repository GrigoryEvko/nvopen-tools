// Function: sub_CDCA30
// Address: 0xcdca30
//
__int64 __fastcall sub_CDCA30(__int64 a1, __int64 a2)
{
  char v3; // al
  _BOOL8 v4; // rcx
  unsigned int v5; // eax
  char v6; // al
  _BOOL8 v7; // rcx
  char v8; // al
  _BOOL8 v9; // rcx
  char v10; // al
  _BOOL8 v11; // rcx
  char v12; // al
  _BOOL8 v13; // rcx
  const __m128i *v14; // rax
  __int64 result; // rax
  __int64 *v16; // rdi
  __int64 v17; // rax
  __m128i *v18; // r10
  char v19; // al
  __int64 v20; // r10
  __int64 v21; // rax
  unsigned __int8 (__fastcall *v22)(__int64, const char *, _BOOL8); // r15
  char v23; // al
  _BOOL8 v24; // rdx
  unsigned __int8 (__fastcall *v25)(__int64, const char *, _BOOL8); // r15
  char v26; // al
  _BOOL8 v27; // rdx
  unsigned __int8 (__fastcall *v28)(__int64, const char *, _BOOL8); // r15
  char v29; // al
  _BOOL8 v30; // rdx
  __int64 v31; // [rsp+8h] [rbp-178h]
  char v32; // [rsp+17h] [rbp-169h] BYREF
  __int64 v33; // [rsp+18h] [rbp-168h] BYREF
  _QWORD v34[2]; // [rsp+20h] [rbp-160h] BYREF
  _BYTE *v35; // [rsp+30h] [rbp-150h] BYREF
  __int64 v36; // [rsp+38h] [rbp-148h]
  _QWORD v37[2]; // [rsp+40h] [rbp-140h] BYREF
  _BYTE v38[128]; // [rsp+50h] [rbp-130h] BYREF
  __m128i v39; // [rsp+D0h] [rbp-B0h]
  __m128i v40; // [rsp+E0h] [rbp-A0h]
  __m128i v41; // [rsp+F0h] [rbp-90h]
  __m128i v42; // [rsp+100h] [rbp-80h]
  __m128i v43; // [rsp+110h] [rbp-70h]
  __m128i v44; // [rsp+120h] [rbp-60h]
  __m128i v45; // [rsp+130h] [rbp-50h]
  __int64 v46; // [rsp+140h] [rbp-40h]

  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v4 = 0;
  if ( v3 && !*(_DWORD *)a2 )
    v4 = *(_DWORD *)(a2 + 4) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, _BYTE **, _BYTE *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Version",
         0,
         v4,
         &v35,
         v38) )
  {
    sub_CCC540(a1, (unsigned int *)a2);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v38);
    v5 = *(_DWORD *)a2;
LABEL_7:
    if ( v5 )
      goto LABEL_8;
    goto LABEL_39;
  }
  if ( !(_BYTE)v35 )
  {
    v5 = *(_DWORD *)a2;
    goto LABEL_7;
  }
  *(_QWORD *)a2 = 0;
LABEL_39:
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, _BYTE **, _BYTE *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Version",
         1,
         0,
         &v35,
         v38) )
  {
    sub_CCC2C0(a1, (unsigned int *)a2);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v38);
  }
LABEL_8:
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v7 = 0;
  if ( v6 && !*(_DWORD *)(a2 + 8) )
    v7 = *(_DWORD *)(a2 + 12) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, _BYTE **, _BYTE *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "NvvmIRVersion",
         0,
         v7,
         &v35,
         v38) )
  {
    sub_CCC540(a1, (unsigned int *)(a2 + 8));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v38);
  }
  else if ( (_BYTE)v35 )
  {
    *(_QWORD *)(a2 + 8) = 0;
  }
  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v9 = 0;
  if ( v8 && !*(_DWORD *)(a2 + 16) )
    v9 = *(_DWORD *)(a2 + 20) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, _BYTE **, _BYTE *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "NvvmDebugVersion",
         0,
         v9,
         &v35,
         v38) )
  {
    sub_CCC540(a1, (unsigned int *)(a2 + 16));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v38);
  }
  else if ( (_BYTE)v35 )
  {
    *(_QWORD *)(a2 + 16) = 0;
  }
  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v11 = 0;
  if ( v10 && !*(_DWORD *)(a2 + 24) )
    v11 = *(_DWORD *)(a2 + 28) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, _BYTE **, _BYTE *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "LlvmVersion",
         0,
         v11,
         &v35,
         v38) )
  {
    sub_CCC540(a1, (unsigned int *)(a2 + 24));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v38);
  }
  else if ( (_BYTE)v35 )
  {
    *(_QWORD *)(a2 + 24) = 0;
  }
  v12 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v13 = 0;
  if ( v12 )
    v13 = *(_DWORD *)(a2 + 32) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, _BYTE **, _BYTE *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "IRLevel",
         0,
         v13,
         &v35,
         v38) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v22 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v23 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v24 = 0;
    if ( v23 )
      v24 = *(_DWORD *)(a2 + 32) == 0;
    if ( v22(a1, "NVVM_IR_LEVEL_UNIFIED_AFTER_DCI", v24) )
      *(_DWORD *)(a2 + 32) = 0;
    v25 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v26 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v27 = 0;
    if ( v26 )
      v27 = *(_DWORD *)(a2 + 32) == 1;
    if ( v25(a1, "NVVM_IR_LEVEL_LTO", v27) )
      *(_DWORD *)(a2 + 32) = 1;
    v28 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v29 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v30 = 0;
    if ( v29 )
      v30 = *(_DWORD *)(a2 + 32) == 2;
    if ( v28(a1, "NVVM_IR_LEVEL_OPTIX", v30) )
      *(_DWORD *)(a2 + 32) = 2;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v38);
  }
  else if ( (_BYTE)v35 )
  {
    *(_DWORD *)(a2 + 32) = 0;
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v14 = *(const __m128i **)(a2 + 80);
    *(__m128i *)v38 = _mm_loadu_si128(v14);
    *(__m128i *)&v38[16] = _mm_loadu_si128(v14 + 1);
    *(__m128i *)&v38[32] = _mm_loadu_si128(v14 + 2);
    *(__m128i *)&v38[48] = _mm_loadu_si128(v14 + 3);
    *(__m128i *)&v38[64] = _mm_loadu_si128(v14 + 4);
    *(__m128i *)&v38[80] = _mm_loadu_si128(v14 + 5);
    *(__m128i *)&v38[96] = _mm_loadu_si128(v14 + 6);
    *(__m128i *)&v38[112] = _mm_loadu_si128(v14 + 7);
    v39 = _mm_loadu_si128(v14 + 8);
    v40 = _mm_loadu_si128(v14 + 9);
    v41 = _mm_loadu_si128(v14 + 10);
    v42 = _mm_loadu_si128(v14 + 11);
    v43 = _mm_loadu_si128(v14 + 12);
    v44 = _mm_loadu_si128(v14 + 13);
    v45 = _mm_loadu_si128(v14 + 14);
    v46 = v14[15].m128i_i64[0];
    if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, _QWORD *, _BYTE **))(*(_QWORD *)a1 + 120LL))(
           a1,
           "Options",
           1,
           0,
           v34,
           &v35) )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
      sub_CDB4D0(a1, (__int64)v38);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
      (*(void (__fastcall **)(__int64, _BYTE *))(*(_QWORD *)a1 + 128LL))(a1, v35);
    }
  }
  else
  {
    v16 = *(__int64 **)(sub_CB0A70(a1) + 8);
    v17 = *v16;
    v16[10] += 248;
    v18 = (__m128i *)((v17 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    if ( v16[1] >= (unsigned __int64)&v18[15].m128i_u64[1] && v17 )
      *v16 = (__int64)&v18[15].m128i_i64[1];
    else
      v18 = (__m128i *)sub_9D1E70((__int64)v16, 248, 248, 3);
    memset(v38, 0, 0x78u);
    v31 = (__int64)v18;
    sub_CCBB10(v18, (const __m128i *)v38);
    v19 = (*(__int64 (__fastcall **)(__int64, char *, __int64, _QWORD, _QWORD *, _BYTE **))(*(_QWORD *)a1 + 120LL))(
            a1,
            "Options",
            1,
            0,
            v34,
            &v35);
    v20 = v31;
    if ( v19 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
      sub_CDB4D0(a1, v31);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
      (*(void (__fastcall **)(__int64, _BYTE *))(*(_QWORD *)a1 + 128LL))(a1, v35);
      v20 = v31;
    }
    *(_QWORD *)(a2 + 80) = v20;
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, _BYTE **, _BYTE *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "IsBinary",
         1,
         0,
         &v35,
         v38) )
  {
    sub_CCCDD0(a1, (_BYTE *)(a2 + 72));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v38);
  }
  result = (*(__int64 (__fastcall **)(__int64, char *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "Module",
             1,
             0,
             &v32,
             &v33);
  if ( (_BYTE)result )
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    {
      *(_QWORD *)&v38[40] = 0x100000000LL;
      v35 = v37;
      *(_QWORD *)v38 = &unk_49DD210;
      v36 = 0;
      LOBYTE(v37[0]) = 0;
      memset(&v38[8], 0, 32);
      *(_QWORD *)&v38[48] = &v35;
      sub_CB5980((__int64)v38, 0, 0, 0);
      sub_CB0A70(a1);
      sub_CB6200((__int64)v38, *(unsigned __int8 **)(a2 + 40), *(_QWORD *)(a2 + 48));
      v34[0] = v35;
      v34[1] = v36;
      (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 224LL))(a1, v34);
      *(_QWORD *)v38 = &unk_49DD210;
      sub_CB5840((__int64)v38);
      if ( v35 != (_BYTE *)v37 )
        j_j___libc_free_0(v35, v37[0] + 1LL);
    }
    else
    {
      v21 = *(_QWORD *)a1;
      v35 = 0;
      v36 = 0;
      (*(void (__fastcall **)(__int64, _BYTE **))(v21 + 224))(a1, &v35);
      sub_CB0A70(a1);
      *(_QWORD *)v38 = &v38[16];
      if ( v35 )
      {
        sub_CCBBE0((__int64 *)v38, v35, (__int64)&v35[v36]);
      }
      else
      {
        *(_QWORD *)&v38[8] = 0;
        v38[16] = 0;
      }
      sub_2240AE0(a2 + 40, v38);
      if ( *(_BYTE **)v38 != &v38[16] )
        j_j___libc_free_0(*(_QWORD *)v38, *(_QWORD *)&v38[16] + 1LL);
    }
    return (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v33);
  }
  return result;
}
