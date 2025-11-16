// Function: sub_257A720
// Address: 0x257a720
//
__int64 __fastcall sub_257A720(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v8; // xmm0
  _QWORD *v9; // rax
  __int64 v10; // r13
  __int64 (__fastcall *v11)(__int64); // rax
  _BYTE *v12; // rdi
  unsigned __int8 (*v13)(void); // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // r14
  __int64 (__fastcall *v24)(__int64); // rax
  _BYTE *v25; // rdi
  unsigned __int8 (*v26)(void); // rax
  __int64 (__fastcall *v27)(__int64); // rax
  _BYTE *v28; // rdi
  void (*v29)(void); // rax
  int v30; // ebx
  __m128i v34; // [rsp+20h] [rbp-70h] BYREF
  char v35; // [rsp+3Fh] [rbp-51h] BYREF
  void *v36; // [rsp+40h] [rbp-50h] BYREF
  __m128i v37; // [rsp+48h] [rbp-48h]

  v34.m128i_i64[0] = a2;
  v34.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v34) )
    v34.m128i_i64[1] = 0;
  v8 = _mm_load_si128(&v34);
  v36 = &unk_438A658;
  v37 = v8;
  v9 = sub_25134D0(a1 + 136, (__int64 *)&v36);
  if ( v9 )
  {
    v10 = v9[3];
    if ( v10 )
    {
      if ( a5 != 2 && a4 )
      {
        v11 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL);
        if ( v11 == sub_2505FA0 )
          v12 = (_BYTE *)(v10 + 88);
        else
          v12 = (_BYTE *)v11(v10);
        v13 = *(unsigned __int8 (**)(void))(*(_QWORD *)v12 + 16LL);
        if ( (char *)v13 != (char *)sub_2506C20 )
        {
          if ( !v13() )
            goto LABEL_15;
          goto LABEL_14;
        }
        if ( v12[8] != 0xFF && v12[9] != 0xFF && v12[10] != 0xFF && v12[11] != 0xFF )
LABEL_14:
          sub_250ED80(a1, v10, a4, a5);
      }
LABEL_15:
      if ( a6 && *(_DWORD *)(a1 + 3552) == 1 )
        sub_251C580(a1, v10);
      return v10;
    }
  }
  v15 = *(_QWORD *)(a1 + 4376);
  if ( v15 )
  {
    v36 = &unk_438A658;
    if ( !sub_2517B80(v15, (__int64 *)&v36) )
      return 0;
  }
  v16 = sub_25096F0(&v34);
  v17 = v16;
  if ( v16 )
  {
    if ( (unsigned __int8)sub_B2D610(v16, 20) || (unsigned __int8)sub_B2D610(v17, 48) )
      return 0;
  }
  if ( !(unsigned __int8)sub_254F6C0(a1, v34.m128i_i64, &v35) )
    return 0;
  v10 = sub_25665C0(&v34, a1);
  v36 = &unk_438A658;
  v37 = _mm_loadu_si128((const __m128i *)(v10 + 72));
  *sub_2519B70(a1 + 136, (__int64)&v36) = v10;
  if ( *(_DWORD *)(a1 + 3552) <= 1u )
  {
    v36 = (void *)(v10 & 0xFFFFFFFFFFFFFFFBLL);
    sub_257A300(a1 + 224, (unsigned __int64 *)&v36, v18, v19, v20, v21);
    if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v10) )
      goto LABEL_43;
  }
  v36 = (void *)v10;
  v22 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250B230, (__int64)&v36);
  ++*(_DWORD *)(a1 + 3556);
  v23 = v22;
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v10 + 24LL))(v10, a1);
  --*(_DWORD *)(a1 + 3556);
  if ( v23 )
    sub_C9AF60(v23);
  if ( v35 )
  {
    if ( a7 )
    {
      v30 = *(_DWORD *)(a1 + 3552);
      *(_DWORD *)(a1 + 3552) = 1;
      sub_251C580(a1, v10);
      *(_DWORD *)(a1 + 3552) = v30;
    }
    if ( a4 )
    {
      v24 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL);
      if ( v24 == sub_2505FA0 )
        v25 = (_BYTE *)(v10 + 88);
      else
        v25 = (_BYTE *)v24(v10);
      v26 = *(unsigned __int8 (**)(void))(*(_QWORD *)v25 + 16LL);
      if ( (char *)v26 == (char *)sub_2506C20 )
      {
        if ( v25[8] == 0xFF || v25[9] == 0xFF || v25[10] == 0xFF || v25[11] == 0xFF )
          return v10;
      }
      else if ( !v26() )
      {
        return v10;
      }
      sub_250ED80(a1, v10, a4, a5);
    }
  }
  else
  {
LABEL_43:
    v27 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL);
    if ( v27 == sub_2505FA0 )
      v28 = (_BYTE *)(v10 + 88);
    else
      v28 = (_BYTE *)v27(v10);
    v29 = *(void (**)(void))(*(_QWORD *)v28 + 40LL);
    if ( (char *)v29 == (char *)sub_2505DD0 )
      v28[12] = 1;
    else
      v29();
  }
  return v10;
}
