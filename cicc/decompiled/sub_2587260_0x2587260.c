// Function: sub_2587260
// Address: 0x2587260
//
__int64 __fastcall sub_2587260(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v8; // xmm0
  __int64 v9; // rsi
  _QWORD *v10; // rax
  __int64 v11; // r13
  __int64 (__fastcall *v12)(__int64); // rax
  _BYTE *v13; // rdi
  __int64 (__fastcall *v14)(__int64); // rax
  char v15; // al
  __int64 v17; // rax
  int v18; // ecx
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // r13
  unsigned __int8 *v22; // r13
  unsigned __int8 v23; // cl
  unsigned __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // r14
  __int64 (__fastcall *v31)(__int64); // rax
  _BYTE *v32; // rdi
  __int64 (__fastcall *v33)(__int64); // rax
  __int64 (__fastcall *v35)(__int64); // rax
  __int64 v36; // rdi
  int v37; // ebx
  char v40; // [rsp+18h] [rbp-68h]
  __m128i v42; // [rsp+20h] [rbp-60h] BYREF
  void *v43; // [rsp+30h] [rbp-50h] BYREF
  __m128i v44; // [rsp+38h] [rbp-48h]

  v42.m128i_i64[0] = a2;
  v42.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v42) )
    v42.m128i_i64[1] = 0;
  v8 = _mm_load_si128(&v42);
  v9 = (__int64)&v43;
  v43 = &unk_438A669;
  v44 = v8;
  v10 = sub_25134D0(a1 + 136, (__int64 *)&v43);
  if ( !v10 || (v11 = v10[3]) == 0 )
  {
    v17 = sub_250D180(v42.m128i_i64, (__int64)&v43);
    v18 = *(unsigned __int8 *)(v17 + 8);
    if ( (unsigned int)(v18 - 17) <= 1 )
      LOBYTE(v18) = *(_BYTE *)(**(_QWORD **)(v17 + 16) + 8LL);
    if ( (_BYTE)v18 != 14 )
      return 0;
    v19 = *(_QWORD *)(a1 + 4376);
    if ( v19 )
    {
      v9 = (__int64)&v43;
      v43 = &unk_438A669;
      if ( !sub_2517B80(v19, (__int64 *)&v43) )
        return 0;
    }
    v20 = sub_25096F0(&v42);
    v21 = v20;
    if ( v20 )
    {
      if ( (unsigned __int8)sub_B2D610(v20, 20) )
        return 0;
      v9 = 48;
      if ( (unsigned __int8)sub_B2D610(v21, 48) )
        return 0;
    }
    if ( *(_DWORD *)(a1 + 3556) > dword_4FEEF68[0] )
      return 0;
    if ( (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) > 1 )
    {
      v22 = sub_250CBE0(v42.m128i_i64, v9);
      v23 = sub_2509800(&v42);
      if ( v23 > 7u || ((1LL << v23) & 0xA8) == 0 )
        goto LABEL_34;
      v24 = v42.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
      if ( (v42.m128i_i8[0] & 3) == 3 )
        v24 = *(_QWORD *)(v24 + 24);
      if ( **(_BYTE **)(v24 - 32) != 25 )
      {
LABEL_34:
        if ( (v23 & 0xFD) == 4 )
        {
          if ( (v22[32] & 0xFu) - 7 <= 1 && (unsigned __int8)sub_250CC70(a1, v42.m128i_i64) )
          {
LABEL_37:
            if ( *(_BYTE *)(a1 + 4296) || (unsigned __int8)sub_253A110(*(_QWORD *)(a1 + 200), (__int64)v22) )
              goto LABEL_39;
            v40 = sub_254D3B0(a1, &v42);
LABEL_40:
            v11 = sub_2562AC0(&v42, a1);
            v43 = &unk_438A669;
            v44 = _mm_loadu_si128((const __m128i *)(v11 + 72));
            *sub_2519B70(a1 + 136, (__int64)&v43) = v11;
            if ( *(_DWORD *)(a1 + 3552) <= 1u )
            {
              v43 = (void *)(v11 & 0xFFFFFFFFFFFFFFFBLL);
              sub_257A300(a1 + 224, (unsigned __int64 *)&v43, v25, v26, v27, v28);
              if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v11) )
                goto LABEL_57;
            }
            v43 = (void *)v11;
            v29 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250BA90, (__int64)&v43);
            ++*(_DWORD *)(a1 + 3556);
            v30 = v29;
            (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v11 + 24LL))(v11, a1);
            --*(_DWORD *)(a1 + 3556);
            if ( v30 )
              sub_C9AF60(v30);
            if ( v40 )
            {
              if ( a7 )
              {
                v37 = *(_DWORD *)(a1 + 3552);
                *(_DWORD *)(a1 + 3552) = 1;
                sub_251C580(a1, v11);
                *(_DWORD *)(a1 + 3552) = v37;
              }
              if ( a4 )
              {
                v31 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
                v32 = (_BYTE *)(v31 == sub_2505F20 ? v11 + 88 : v31(v11));
                v33 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v32 + 16LL);
                if ( v33 == sub_2505E30 ? v32[9] : ((__int64 (*)(void))v33)() )
                  sub_250ED80(a1, v11, a4, a5);
              }
            }
            else
            {
LABEL_57:
              v35 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
              if ( v35 == sub_2505F20 )
                v36 = v11 + 88;
              else
                v36 = v35(v11);
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v36 + 40LL))(v36);
            }
            return v11;
          }
        }
        else if ( (unsigned __int8)sub_250CC70(a1, v42.m128i_i64) )
        {
          if ( !v22 )
          {
LABEL_39:
            v40 = 1;
            goto LABEL_40;
          }
          goto LABEL_37;
        }
      }
    }
    v40 = 0;
    goto LABEL_40;
  }
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
  return v11;
}
