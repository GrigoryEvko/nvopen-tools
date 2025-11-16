// Function: sub_257C550
// Address: 0x257c550
//
__int64 __fastcall sub_257C550(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v8; // xmm0
  _QWORD *v9; // rax
  __int64 v10; // r13
  __int64 (__fastcall *v11)(__int64); // rax
  _DWORD *v12; // rdi
  bool (__fastcall *v13)(__int64); // rax
  __int64 i; // rsi
  int v16; // ecx
  unsigned __int8 v17; // al
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 (__fastcall *v27)(__int64); // rax
  _DWORD *v28; // rdi
  unsigned __int8 (*v29)(void); // rax
  __int64 (__fastcall *v30)(__int64); // rax
  _DWORD *v31; // rdi
  void (*v32)(void); // rax
  int v33; // ebx
  __m128i v37; // [rsp+20h] [rbp-70h] BYREF
  char v38; // [rsp+3Fh] [rbp-51h] BYREF
  void *v39; // [rsp+40h] [rbp-50h] BYREF
  __m128i v40; // [rsp+48h] [rbp-48h]

  v37.m128i_i64[0] = a2;
  v37.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v37) )
    v37.m128i_i64[1] = 0;
  v8 = _mm_load_si128(&v37);
  v39 = &unk_438A662;
  v40 = v8;
  v9 = sub_25134D0(a1 + 136, (__int64 *)&v39);
  if ( v9 )
  {
    v10 = v9[3];
    if ( v10 )
    {
      if ( a5 != 2 && a4 )
      {
        v11 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL);
        if ( v11 == sub_2505FF0 )
          v12 = (_DWORD *)(v10 + 88);
        else
          v12 = (_DWORD *)v11(v10);
        v13 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v12 + 16LL);
        if ( v13 == sub_2506010 )
        {
          if ( !v12[3] )
            goto LABEL_11;
LABEL_29:
          sub_250ED80(a1, v10, a4, a5);
          if ( !a6 )
            return v10;
LABEL_12:
          if ( *(_DWORD *)(a1 + 3552) == 1 )
            sub_251C580(a1, v10);
          return v10;
        }
        if ( ((unsigned __int8 (*)(void))v13)() )
          goto LABEL_29;
      }
LABEL_11:
      if ( !a6 )
        return v10;
      goto LABEL_12;
    }
  }
  for ( i = sub_250D180(v37.m128i_i64, (__int64)&v39); ; i = **(_QWORD **)(i + 16) )
  {
    v16 = *(unsigned __int8 *)(i + 8);
    v17 = *(_BYTE *)(i + 8);
    if ( (unsigned int)(v16 - 17) <= 1 )
      v17 = *(_BYTE *)(**(_QWORD **)(i + 16) + 8LL);
    if ( v17 <= 3u || v17 == 5 || (v17 & 0xFD) == 4 )
      break;
    if ( (_BYTE)v16 != 16 )
      return 0;
  }
  v18 = *(_QWORD *)(a1 + 4376);
  if ( v18 )
  {
    v39 = &unk_438A662;
    if ( !sub_2517B80(v18, (__int64 *)&v39) )
      return 0;
  }
  v19 = sub_25096F0(&v37);
  v20 = v19;
  if ( v19 )
  {
    if ( (unsigned __int8)sub_B2D610(v19, 20) || (unsigned __int8)sub_B2D610(v20, 48) )
      return 0;
  }
  if ( !(unsigned __int8)sub_254F6C0(a1, v37.m128i_i64, &v38) )
    return 0;
  v10 = (__int64)sub_2564490(&v37, a1);
  v39 = &unk_438A662;
  v40 = _mm_loadu_si128((const __m128i *)(v10 + 72));
  *sub_2519B70(a1 + 136, (__int64)&v39) = v10;
  if ( *(_DWORD *)(a1 + 3552) <= 1u )
  {
    v39 = (void *)(v10 & 0xFFFFFFFFFFFFFFFBLL);
    sub_257A300(a1 + 224, (unsigned __int64 *)&v39, v21, v22, v23, v24);
    if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v10) )
      goto LABEL_47;
  }
  v39 = (void *)v10;
  v25 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250BBD0, (__int64)&v39);
  ++*(_DWORD *)(a1 + 3556);
  v26 = v25;
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v10 + 24LL))(v10, a1);
  --*(_DWORD *)(a1 + 3556);
  if ( v26 )
    sub_C9AF60(v26);
  if ( v38 )
  {
    if ( a7 )
    {
      v33 = *(_DWORD *)(a1 + 3552);
      *(_DWORD *)(a1 + 3552) = 1;
      sub_251C580(a1, v10);
      *(_DWORD *)(a1 + 3552) = v33;
    }
    if ( a4 )
    {
      v27 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL);
      if ( v27 == sub_2505FF0 )
        v28 = (_DWORD *)(v10 + 88);
      else
        v28 = (_DWORD *)v27(v10);
      v29 = *(unsigned __int8 (**)(void))(*(_QWORD *)v28 + 16LL);
      if ( (char *)v29 == (char *)sub_2506010 )
      {
        if ( !v28[3] )
          return v10;
      }
      else if ( !v29() )
      {
        return v10;
      }
      sub_250ED80(a1, v10, a4, a5);
    }
  }
  else
  {
LABEL_47:
    v30 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL);
    if ( v30 == sub_2505FF0 )
      v31 = (_DWORD *)(v10 + 88);
    else
      v31 = (_DWORD *)v30(v10);
    v32 = *(void (**)(void))(*(_QWORD *)v31 + 40LL);
    if ( (char *)v32 == (char *)sub_2506000 )
      v31[3] = v31[2];
    else
      v32();
  }
  return v10;
}
