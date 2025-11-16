// Function: sub_257B000
// Address: 0x257b000
//
__int64 __fastcall sub_257B000(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v8; // xmm0
  _QWORD *v9; // rax
  __int64 v10; // r13
  __int64 (__fastcall *v11)(__int64); // rax
  _BYTE *v12; // rdi
  __int64 (__fastcall *v13)(__int64); // rax
  char v14; // al
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // r14
  int v24; // eax
  void (*v25)(); // rdx
  __int64 (__fastcall *v26)(__int64); // rax
  _BYTE *v27; // rdi
  __int64 (__fastcall *v28)(__int64); // rax
  __int64 (__fastcall *v30)(__int64); // rax
  __int64 v31; // rdi
  int v32; // ebx
  __m128i v36; // [rsp+20h] [rbp-70h] BYREF
  char v37; // [rsp+3Fh] [rbp-51h] BYREF
  void *v38; // [rsp+40h] [rbp-50h] BYREF
  __m128i v39; // [rsp+48h] [rbp-48h]

  v36.m128i_i64[0] = a2;
  v36.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v36) )
    v36.m128i_i64[1] = 0;
  v8 = _mm_load_si128(&v36);
  v38 = &unk_438A675;
  v39 = v8;
  v9 = sub_25134D0(a1 + 136, (__int64 *)&v38);
  if ( v9 && (v10 = v9[3]) != 0 )
  {
    if ( a5 != 2
      && a4
      && ((v11 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL), v11 != sub_2505F20)
        ? (v12 = (_BYTE *)v11(v10))
        : (v12 = (_BYTE *)(v10 + 88)),
          (v13 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 16LL), v13 != sub_2505E30)
        ? (v14 = ((__int64 (*)(void))v13)())
        : (v14 = v12[9]),
          v14) )
    {
      sub_250ED80(a1, v10, a4, a5);
      if ( !a6 )
        return v10;
    }
    else if ( !a6 )
    {
      return v10;
    }
    if ( *(_DWORD *)(a1 + 3552) == 1 )
      sub_251C580(a1, v10);
  }
  else
  {
    v16 = *(_QWORD *)(a1 + 4376);
    if ( v16 )
    {
      v38 = &unk_438A675;
      if ( !sub_2517B80(v16, (__int64 *)&v38) )
        return 0;
    }
    v17 = sub_25096F0(&v36);
    v18 = v17;
    if ( v17 )
    {
      if ( (unsigned __int8)sub_B2D610(v17, 20) || (unsigned __int8)sub_B2D610(v18, 48) )
        return 0;
    }
    if ( !(unsigned __int8)sub_254F6C0(a1, v36.m128i_i64, &v37) )
    {
      return 0;
    }
    else
    {
      v10 = sub_25663F0(&v36, a1);
      v38 = &unk_438A675;
      v39 = _mm_loadu_si128((const __m128i *)(v10 + 72));
      *sub_2519B70(a1 + 136, (__int64)&v38) = v10;
      if ( *(_DWORD *)(a1 + 3552) <= 1u )
      {
        v38 = (void *)(v10 & 0xFFFFFFFFFFFFFFFBLL);
        sub_257A300(a1 + 224, (unsigned __int64 *)&v38, v19, v20, v21, v22);
        if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v10) )
          goto LABEL_42;
      }
      v38 = (void *)v10;
      v23 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250AFB0, (__int64)&v38);
      v24 = *(_DWORD *)(a1 + 3556);
      *(_DWORD *)(a1 + 3556) = v24 + 1;
      v25 = *(void (**)())(*(_QWORD *)v10 + 24LL);
      if ( v25 != nullsub_1516 )
      {
        ((void (__fastcall *)(__int64, __int64))v25)(v10, a1);
        v24 = *(_DWORD *)(a1 + 3556) - 1;
      }
      *(_DWORD *)(a1 + 3556) = v24;
      if ( v23 )
        sub_C9AF60(v23);
      if ( v37 )
      {
        if ( a7 )
        {
          v32 = *(_DWORD *)(a1 + 3552);
          *(_DWORD *)(a1 + 3552) = 1;
          sub_251C580(a1, v10);
          *(_DWORD *)(a1 + 3552) = v32;
        }
        if ( a4 )
        {
          v26 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL);
          v27 = (_BYTE *)(v26 == sub_2505F20 ? v10 + 88 : v26(v10));
          v28 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v27 + 16LL);
          if ( v28 == sub_2505E30 ? v27[9] : ((__int64 (*)(void))v28)() )
            sub_250ED80(a1, v10, a4, a5);
        }
      }
      else
      {
LABEL_42:
        v30 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL);
        if ( v30 == sub_2505F20 )
          v31 = v10 + 88;
        else
          v31 = v30(v10);
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v31 + 40LL))(v31);
      }
    }
  }
  return v10;
}
