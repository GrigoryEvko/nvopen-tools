// Function: sub_251C7D0
// Address: 0x251c7d0
//
__int64 __fastcall sub_251C7D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v8; // xmm0
  __int64 v9; // rsi
  _QWORD *v10; // rax
  __int64 v11; // r13
  __int64 (__fastcall *v12)(__int64); // rax
  _BYTE *v13; // rdi
  __int64 (__fastcall *v14)(__int64); // rax
  char v15; // al
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // r13
  unsigned __int8 *v20; // r13
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r14
  int v26; // eax
  void (*v27)(); // rdx
  __int64 (__fastcall *v28)(__int64); // rax
  _BYTE *v29; // rdi
  __int64 (__fastcall *v30)(__int64); // rax
  __int64 (__fastcall *v32)(__int64); // rax
  _BYTE *v33; // rdi
  void (*v34)(void); // rax
  int v35; // ebx
  char v38; // [rsp+18h] [rbp-68h]
  __m128i v40; // [rsp+20h] [rbp-60h] BYREF
  void *v41; // [rsp+30h] [rbp-50h] BYREF
  __m128i v42; // [rsp+38h] [rbp-48h]

  v40.m128i_i64[0] = a2;
  v40.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v40) )
    v40.m128i_i64[1] = 0;
  v8 = _mm_load_si128(&v40);
  v9 = (__int64)&v41;
  v41 = &unk_438A661;
  v42 = v8;
  v10 = sub_25134D0(a1 + 136, (__int64 *)&v41);
  if ( v10 && (v11 = v10[3]) != 0 )
  {
    if ( a5 != 2
      && a4
      && ((v12 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL), v12 != sub_2505F20)
        ? (v13 = (_BYTE *)v12(v11))
        : (v13 = (_BYTE *)(v11 + 88)),
          (v14 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v13 + 16LL), v14 != sub_2505E30)
        ? (v15 = ((__int64 (*)(void))v14)())
        : (v15 = v13[9]),
          v15) )
    {
      sub_250ED80(a1, v11, a4, a5);
      if ( !a6 )
        return v11;
    }
    else if ( !a6 )
    {
      return v11;
    }
    if ( *(_DWORD *)(a1 + 3552) == 1 )
      sub_251C580(a1, v11);
  }
  else
  {
    v17 = *(_QWORD *)(a1 + 4376);
    if ( v17 )
    {
      v9 = (__int64)&v41;
      v41 = &unk_438A661;
      if ( !sub_2517B80(v17, (__int64 *)&v41) )
        return 0;
    }
    v18 = sub_25096F0(&v40);
    v19 = v18;
    if ( v18 )
    {
      if ( (unsigned __int8)sub_B2D610(v18, 20) )
        return 0;
      v9 = 48;
      if ( (unsigned __int8)sub_B2D610(v19, 48) )
        return 0;
    }
    if ( *(_DWORD *)(a1 + 3556) > unk_4FEEF68 )
    {
      return 0;
    }
    else
    {
      if ( (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) > 1
        && (v20 = sub_250CBE0(v40.m128i_i64, v9), (v38 = sub_250CC70(a1, v40.m128i_i64)) != 0) )
      {
        if ( v20 )
        {
          v38 = *(_BYTE *)(a1 + 4296);
          if ( !v38 )
          {
            v38 = sub_2506F10(*(_QWORD *)(a1 + 200), (__int64)v20);
            if ( !v38 )
              v38 = sub_2508DC0(a1, &v40);
          }
        }
      }
      else
      {
        v38 = 0;
      }
      v11 = sub_25623E0(&v40, a1);
      v41 = &unk_438A661;
      v42 = _mm_loadu_si128((const __m128i *)(v11 + 72));
      *sub_2519B70(a1 + 136, (__int64)&v41) = v11;
      if ( *(_DWORD *)(a1 + 3552) <= 1u )
      {
        v41 = (void *)(v11 & 0xFFFFFFFFFFFFFFFBLL);
        sub_251B630(a1 + 224, (unsigned __int64 *)&v41, v21, v22, v23, v24);
        if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v11) )
          goto LABEL_51;
      }
      v41 = (void *)v11;
      v25 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250A610, (__int64)&v41);
      v26 = *(_DWORD *)(a1 + 3556);
      *(_DWORD *)(a1 + 3556) = v26 + 1;
      v27 = *(void (**)())(*(_QWORD *)v11 + 24LL);
      if ( v27 != nullsub_1516 )
      {
        ((void (__fastcall *)(__int64, __int64))v27)(v11, a1);
        v26 = *(_DWORD *)(a1 + 3556) - 1;
      }
      *(_DWORD *)(a1 + 3556) = v26;
      if ( v25 )
        sub_C9AF60(v25);
      if ( v38 )
      {
        if ( a7 )
        {
          v35 = *(_DWORD *)(a1 + 3552);
          *(_DWORD *)(a1 + 3552) = 1;
          sub_251C580(a1, v11);
          *(_DWORD *)(a1 + 3552) = v35;
        }
        if ( a4 )
        {
          v28 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
          v29 = (_BYTE *)(v28 == sub_2505F20 ? v11 + 88 : v28(v11));
          v30 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v29 + 16LL);
          if ( v30 == sub_2505E30 ? v29[9] : ((__int64 (*)(void))v30)() )
            sub_250ED80(a1, v11, a4, a5);
        }
      }
      else
      {
LABEL_51:
        v32 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
        if ( v32 == sub_2505F20 )
          v33 = (_BYTE *)(v11 + 88);
        else
          v33 = (_BYTE *)v32(v11);
        v34 = *(void (**)(void))(*(_QWORD *)v33 + 40LL);
        if ( (char *)v34 == (char *)sub_2505E20 )
          v33[9] = v33[8];
        else
          v34();
      }
    }
  }
  return v11;
}
