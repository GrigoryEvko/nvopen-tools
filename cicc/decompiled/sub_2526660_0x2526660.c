// Function: sub_2526660
// Address: 0x2526660
//
__int64 __fastcall sub_2526660(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
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
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r14
  int v28; // eax
  void (*v29)(); // rdx
  __int64 (__fastcall *v30)(__int64); // rax
  _BYTE *v31; // rdi
  __int64 (__fastcall *v32)(__int64); // rax
  __int64 (__fastcall *v34)(__int64); // rax
  _BYTE *v35; // rdi
  void (*v36)(void); // rax
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
  v43 = &unk_438A664;
  v44 = v8;
  v10 = sub_25134D0(a1 + 136, (__int64 *)&v43);
  if ( !v10 || (v11 = v10[3]) == 0 )
  {
    v17 = *(_QWORD *)(a1 + 4376);
    if ( v17 )
    {
      v9 = (__int64)&v43;
      v43 = &unk_438A664;
      if ( !sub_2517B80(v17, (__int64 *)&v43) )
        return 0;
    }
    v18 = sub_25096F0(&v42);
    v19 = v18;
    if ( v18 )
    {
      if ( (unsigned __int8)sub_B2D610(v18, 20) )
        return 0;
      v9 = 48;
      if ( (unsigned __int8)sub_B2D610(v19, 48) )
        return 0;
    }
    if ( *(_DWORD *)(a1 + 3556) > dword_4FEEF68[0] )
      return 0;
    if ( (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) > 1 )
    {
      v20 = sub_250CBE0(v42.m128i_i64, v9);
      v21 = sub_2509800(&v42);
      if ( v21 > 7u || ((1LL << v21) & 0xA8) == 0 )
        goto LABEL_28;
      v22 = v42.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
      if ( (v42.m128i_i8[0] & 3) == 3 )
        v22 = *(_QWORD *)(v22 + 24);
      if ( **(_BYTE **)(v22 - 32) != 25 )
      {
LABEL_28:
        if ( (v21 & 0xFD) == 4 )
        {
          if ( (v20[32] & 0xFu) - 7 <= 1 && (unsigned __int8)sub_250CC70(a1, v42.m128i_i64) )
          {
LABEL_31:
            if ( *(_BYTE *)(a1 + 4296) || (unsigned __int8)sub_2506F10(*(_QWORD *)(a1 + 200), (__int64)v20) )
              goto LABEL_33;
            v40 = sub_2508DC0(a1, &v42);
LABEL_38:
            v11 = sub_2563EA0(&v42, a1);
            v43 = &unk_438A664;
            v44 = _mm_loadu_si128((const __m128i *)(v11 + 72));
            *sub_2519B70(a1 + 136, (__int64)&v43) = v11;
            if ( *(_DWORD *)(a1 + 3552) <= 1u )
            {
              v43 = (void *)(v11 & 0xFFFFFFFFFFFFFFFBLL);
              sub_251B630(a1 + 224, (unsigned __int64 *)&v43, v23, v24, v25, v26);
              if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v11) )
                goto LABEL_55;
            }
            v43 = (void *)v11;
            v27 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250A3B0, (__int64)&v43);
            v28 = *(_DWORD *)(a1 + 3556);
            *(_DWORD *)(a1 + 3556) = v28 + 1;
            v29 = *(void (**)())(*(_QWORD *)v11 + 24LL);
            if ( v29 != nullsub_1516 )
            {
              ((void (__fastcall *)(__int64, __int64))v29)(v11, a1);
              v28 = *(_DWORD *)(a1 + 3556) - 1;
            }
            *(_DWORD *)(a1 + 3556) = v28;
            if ( v27 )
              sub_C9AF60(v27);
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
                v30 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
                v31 = (_BYTE *)(v30 == sub_2505DE0 ? v11 + 88 : v30(v11));
                v32 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v31 + 16LL);
                if ( v32 == sub_2505E60 ? v31[17] : ((__int64 (*)(void))v32)() )
                  sub_250ED80(a1, v11, a4, a5);
              }
            }
            else
            {
LABEL_55:
              v34 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
              if ( v34 == sub_2505DE0 )
                v35 = (_BYTE *)(v11 + 88);
              else
                v35 = (_BYTE *)v34(v11);
              v36 = *(void (**)(void))(*(_QWORD *)v35 + 40LL);
              if ( (char *)v36 == (char *)sub_2505EB0 )
                v35[17] = v35[16];
              else
                v36();
            }
            return v11;
          }
        }
        else if ( (unsigned __int8)sub_250CC70(a1, v42.m128i_i64) )
        {
          if ( !v20 )
          {
LABEL_33:
            v40 = 1;
            goto LABEL_38;
          }
          goto LABEL_31;
        }
      }
    }
    v40 = 0;
    goto LABEL_38;
  }
  if ( a5 != 2
    && a4
    && ((v12 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL), v12 != sub_2505DE0)
      ? (v13 = (_BYTE *)v12(v11))
      : (v13 = (_BYTE *)(v11 + 88)),
        (v14 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v13 + 16LL), v14 != sub_2505E60)
      ? (v15 = ((__int64 (*)(void))v14)())
      : (v15 = v13[17]),
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
