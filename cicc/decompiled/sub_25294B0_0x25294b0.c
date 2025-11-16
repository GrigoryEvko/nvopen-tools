// Function: sub_25294B0
// Address: 0x25294b0
//
__int64 __fastcall sub_25294B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v8; // xmm0
  _QWORD *v9; // rax
  __int64 v10; // r13
  __int64 (__fastcall *v11)(__int64); // rax
  _BYTE *v12; // rdi
  bool (__fastcall *v13)(__int64); // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r14
  int v23; // eax
  void (*v24)(); // rdx
  __int64 (__fastcall *v25)(__int64); // rax
  _BYTE *v26; // rdi
  unsigned __int8 (*v27)(void); // rax
  __int64 (__fastcall *v28)(__int64); // rax
  _BYTE *v29; // rdi
  void (*v30)(void); // rax
  int v31; // ebx
  __m128i v35; // [rsp+20h] [rbp-70h] BYREF
  char v36; // [rsp+3Fh] [rbp-51h] BYREF
  void *v37; // [rsp+40h] [rbp-50h] BYREF
  __m128i v38; // [rsp+48h] [rbp-48h]

  v35.m128i_i64[0] = a2;
  v35.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v35) )
    v35.m128i_i64[1] = 0;
  v8 = _mm_load_si128(&v35);
  v37 = &unk_438A668;
  v38 = v8;
  v9 = sub_25134D0(a1 + 136, (__int64 *)&v37);
  if ( v9 )
  {
    v10 = v9[3];
    if ( v10 )
    {
      if ( a5 != 2 && a4 )
      {
        v11 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL);
        if ( v11 == sub_2505EC0 )
          v12 = (_BYTE *)(v10 + 88);
        else
          v12 = (_BYTE *)v11(v10);
        v13 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v12 + 16LL);
        if ( v13 == sub_2505EE0 )
        {
          if ( !v12[9] )
            goto LABEL_11;
LABEL_39:
          sub_250ED80(a1, v10, a4, a5);
          if ( !a6 )
            return v10;
LABEL_12:
          if ( *(_DWORD *)(a1 + 3552) == 1 )
            sub_251C580(a1, v10);
          return v10;
        }
        if ( ((unsigned __int8 (*)(void))v13)() )
          goto LABEL_39;
      }
LABEL_11:
      if ( !a6 )
        return v10;
      goto LABEL_12;
    }
  }
  if ( (unsigned int)((char)sub_2509800(&v35) - 4) > 1
    && *(_BYTE *)(sub_250D180(v35.m128i_i64, (__int64)&v37) + 8) != 14 )
  {
    return 0;
  }
  v15 = *(_QWORD *)(a1 + 4376);
  if ( v15 )
  {
    v37 = &unk_438A668;
    if ( !sub_2517B80(v15, (__int64 *)&v37) )
      return 0;
  }
  v16 = sub_25096F0(&v35);
  v17 = v16;
  if ( v16 )
  {
    if ( (unsigned __int8)sub_B2D610(v16, 20) || (unsigned __int8)sub_B2D610(v17, 48) )
      return 0;
  }
  if ( !(unsigned __int8)sub_250CDD0(a1, v35.m128i_i64, &v36) )
    return 0;
  v10 = sub_2566680(&v35, a1);
  v37 = &unk_438A668;
  v38 = _mm_loadu_si128((const __m128i *)(v10 + 72));
  *sub_2519B70(a1 + 136, (__int64)&v37) = v10;
  if ( *(_DWORD *)(a1 + 3552) <= 1u )
  {
    v37 = (void *)(v10 & 0xFFFFFFFFFFFFFFFBLL);
    sub_251B630(a1 + 224, (unsigned __int64 *)&v37, v18, v19, v20, v21);
    if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v10) )
      goto LABEL_43;
  }
  v37 = (void *)v10;
  v22 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_2509EA0, (__int64)&v37);
  v23 = *(_DWORD *)(a1 + 3556);
  *(_DWORD *)(a1 + 3556) = v23 + 1;
  v24 = *(void (**)())(*(_QWORD *)v10 + 24LL);
  if ( v24 != nullsub_1516 )
  {
    ((void (__fastcall *)(__int64, __int64))v24)(v10, a1);
    v23 = *(_DWORD *)(a1 + 3556) - 1;
  }
  *(_DWORD *)(a1 + 3556) = v23;
  if ( v22 )
    sub_C9AF60(v22);
  if ( v36 )
  {
    if ( a7 )
    {
      v31 = *(_DWORD *)(a1 + 3552);
      *(_DWORD *)(a1 + 3552) = 1;
      sub_251C580(a1, v10);
      *(_DWORD *)(a1 + 3552) = v31;
    }
    if ( a4 )
    {
      v25 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL);
      if ( v25 == sub_2505EC0 )
        v26 = (_BYTE *)(v10 + 88);
      else
        v26 = (_BYTE *)v25(v10);
      v27 = *(unsigned __int8 (**)(void))(*(_QWORD *)v26 + 16LL);
      if ( (char *)v27 == (char *)sub_2505EE0 )
      {
        if ( !v26[9] )
          return v10;
      }
      else if ( !v27() )
      {
        return v10;
      }
      sub_250ED80(a1, v10, a4, a5);
    }
  }
  else
  {
LABEL_43:
    v28 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL);
    if ( v28 == sub_2505EC0 )
      v29 = (_BYTE *)(v10 + 88);
    else
      v29 = (_BYTE *)v28(v10);
    v30 = *(void (**)(void))(*(_QWORD *)v29 + 40LL);
    if ( (char *)v30 == (char *)sub_2505ED0 )
      v29[9] = v29[8];
    else
      v30();
  }
  return v10;
}
