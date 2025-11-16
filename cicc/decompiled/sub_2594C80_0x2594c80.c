// Function: sub_2594C80
// Address: 0x2594c80
//
__int64 __fastcall sub_2594C80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
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
  unsigned __int8 v21; // cl
  unsigned __int64 v22; // rdx
  unsigned __int8 v23; // cl
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 v29; // r14
  __int64 (__fastcall *v30)(__int64); // rax
  _BYTE *v31; // rdi
  __int64 (__fastcall *v32)(__int64); // rax
  __int64 (__fastcall *v34)(__int64); // rax
  __int64 v35; // rdi
  int v36; // ebx
  unsigned __int8 *v38; // [rsp+8h] [rbp-78h]
  __m128i v41; // [rsp+20h] [rbp-60h] BYREF
  void *v42; // [rsp+30h] [rbp-50h] BYREF
  __m128i v43; // [rsp+38h] [rbp-48h]

  v41.m128i_i64[0] = a2;
  v41.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v41) )
    v41.m128i_i64[1] = 0;
  v8 = _mm_load_si128(&v41);
  v9 = (__int64)&v42;
  v42 = &unk_438A677;
  v43 = v8;
  v10 = sub_25134D0(a1 + 136, (__int64 *)&v42);
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
      v9 = (__int64)&v42;
      v42 = &unk_438A677;
      if ( !sub_2517B80(v17, (__int64 *)&v42) )
        return 0;
    }
    v18 = sub_25096F0(&v41);
    v19 = v18;
    if ( v18 )
    {
      if ( (unsigned __int8)sub_B2D610(v18, 20) )
        return 0;
      v9 = 48;
      if ( (unsigned __int8)sub_B2D610(v19, 48) )
        return 0;
    }
    if ( *(_DWORD *)(a1 + 3556) > dword_4FEEF68[0] || (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) <= 1 )
      return 0;
    v20 = sub_250CBE0(v41.m128i_i64, v9);
    v21 = sub_2509800(&v41);
    if ( v21 <= 7u && ((1LL << v21) & 0xA8) != 0 )
    {
      v22 = v41.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
      if ( (v41.m128i_i8[0] & 3) == 3 )
        v22 = *(_QWORD *)(v22 + 24);
      if ( **(_BYTE **)(v22 - 32) == 25 )
        return 0;
    }
    v38 = sub_250CBE0(v41.m128i_i64, v9);
    v23 = sub_2509800(&v41);
    if ( v23 <= 6u && ((1LL << v23) & 0x54) != 0 && !(unsigned __int8)sub_254F400(a1, v38) )
      return 0;
    if ( v20
      && !*(_BYTE *)(a1 + 4296)
      && !(unsigned __int8)sub_253A110(*(_QWORD *)(a1 + 200), (__int64)v20)
      && !(unsigned __int8)sub_254D3B0(a1, &v41) )
    {
      return 0;
    }
    else
    {
      v11 = sub_25625A0(&v41, a1);
      v42 = &unk_438A677;
      v43 = _mm_loadu_si128((const __m128i *)(v11 + 72));
      *sub_2519B70(a1 + 136, (__int64)&v42) = v11;
      if ( *(_DWORD *)(a1 + 3552) > 1u
        || (v42 = (void *)(v11 & 0xFFFFFFFFFFFFFFFBLL),
            sub_257A300(a1 + 224, (unsigned __int64 *)&v42, v24, v25, v26, v27),
            *(_DWORD *)(a1 + 3552))
        || (unsigned __int8)sub_250E880(a1, v11) )
      {
        v42 = (void *)v11;
        v28 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250A9C0, (__int64)&v42);
        ++*(_DWORD *)(a1 + 3556);
        v29 = v28;
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v11 + 24LL))(v11, a1);
        --*(_DWORD *)(a1 + 3556);
        if ( v29 )
          sub_C9AF60(v29);
        if ( a7 )
        {
          v36 = *(_DWORD *)(a1 + 3552);
          *(_DWORD *)(a1 + 3552) = 1;
          sub_251C580(a1, v11);
          *(_DWORD *)(a1 + 3552) = v36;
        }
        if ( a4 )
        {
          v30 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
          v31 = (_BYTE *)(v30 == sub_2505F20 ? v11 + 88 : v30(v11));
          v32 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v31 + 16LL);
          if ( v32 == sub_2505E30 ? v31[9] : ((__int64 (*)(void))v32)() )
            sub_250ED80(a1, v11, a4, a5);
        }
      }
      else
      {
        v34 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
        if ( v34 == sub_2505F20 )
          v35 = v11 + 88;
        else
          v35 = v34(v11);
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v35 + 40LL))(v35);
      }
    }
  }
  return v11;
}
