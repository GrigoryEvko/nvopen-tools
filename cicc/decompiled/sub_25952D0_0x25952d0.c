// Function: sub_25952D0
// Address: 0x25952d0
//
__int64 __fastcall sub_25952D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v8; // xmm0
  _QWORD *v9; // rax
  __int64 v10; // r13
  __int64 (__fastcall *v11)(__int64); // rax
  __int64 v12; // rdi
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
  __int64 v25; // rdi
  unsigned __int8 (*v26)(void); // rax
  __int64 (__fastcall *v27)(__int64); // rax
  _BYTE *v28; // rdi
  void (*v29)(void); // rax
  char v30; // al
  __int64 v31; // rsi
  __int64 v32; // rdi
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
  v39 = &unk_438A65E;
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
        if ( v11 == sub_2505F70 )
          v12 = v10 + 88;
        else
          v12 = v11(v10);
        v13 = *(unsigned __int8 (**)(void))(*(_QWORD *)v12 + 16LL);
        if ( (char *)v13 != (char *)sub_2505F80 )
        {
          if ( !v13() )
            goto LABEL_13;
          goto LABEL_12;
        }
        if ( *(_DWORD *)(v12 + 72) || *(_BYTE *)(v12 + 48) )
LABEL_12:
          sub_250ED80(a1, v10, a4, a5);
      }
LABEL_13:
      if ( a6 && *(_DWORD *)(a1 + 3552) == 1 )
        sub_251C580(a1, v10);
      return v10;
    }
  }
  v15 = *(_QWORD *)(a1 + 4376);
  if ( v15 )
  {
    v39 = &unk_438A65E;
    if ( !sub_2517B80(v15, (__int64 *)&v39) )
      return 0;
  }
  v16 = sub_25096F0(&v37);
  v17 = v16;
  if ( v16 )
  {
    if ( (unsigned __int8)sub_B2D610(v16, 20) || (unsigned __int8)sub_B2D610(v17, 48) )
      return 0;
  }
  if ( !(unsigned __int8)sub_254F6C0(a1, v37.m128i_i64, &v38) )
    return 0;
  v10 = sub_25780C0(&v37, a1);
  v39 = &unk_438A65E;
  v40 = _mm_loadu_si128((const __m128i *)(v10 + 72));
  *sub_2519B70(a1 + 136, (__int64)&v39) = v10;
  if ( *(_DWORD *)(a1 + 3552) <= 1u )
  {
    v39 = (void *)(v10 & 0xFFFFFFFFFFFFFFFBLL);
    sub_257A300(a1 + 224, (unsigned __int64 *)&v39, v18, v19, v20, v21);
    if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v10) )
      goto LABEL_39;
  }
  v39 = (void *)v10;
  v22 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250B0F0, (__int64)&v39);
  ++*(_DWORD *)(a1 + 3556);
  v23 = v22;
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v10 + 24LL))(v10, a1);
  --*(_DWORD *)(a1 + 3556);
  if ( v23 )
    sub_C9AF60(v23);
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
      v24 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL);
      if ( v24 == sub_2505F70 )
        v25 = v10 + 88;
      else
        v25 = v24(v10);
      v26 = *(unsigned __int8 (**)(void))(*(_QWORD *)v25 + 16LL);
      if ( (char *)v26 == (char *)sub_2505F80 )
      {
        if ( !*(_DWORD *)(v25 + 72) && !*(_BYTE *)(v25 + 48) )
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
LABEL_39:
    v27 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL);
    if ( v27 == sub_2505F70 )
      v28 = (_BYTE *)(v10 + 88);
    else
      v28 = (_BYTE *)v27(v10);
    v29 = *(void (**)(void))(*(_QWORD *)v28 + 40LL);
    if ( (char *)v29 == (char *)sub_2506110 )
    {
      v30 = v28[8];
      v28[88] = 1;
      v31 = (__int64)(v28 + 16);
      v32 = (__int64)(v28 + 56);
      *(_BYTE *)(v32 - 8) = v30;
      sub_255D9B0(v32, v31);
    }
    else
    {
      v29();
    }
  }
  return v10;
}
