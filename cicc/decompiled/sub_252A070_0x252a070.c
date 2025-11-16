// Function: sub_252A070
// Address: 0x252a070
//
__int64 __fastcall sub_252A070(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v8; // xmm0
  __int64 v9; // rsi
  _QWORD *v10; // rax
  __int64 v11; // r13
  __int64 (__fastcall *v12)(__int64); // rax
  _DWORD *v13; // rdi
  bool (__fastcall *v14)(__int64); // rax
  __int64 v16; // rax
  int v17; // ecx
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r13
  unsigned __int8 *v21; // r13
  unsigned __int8 v22; // cl
  unsigned __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // r14
  int v29; // eax
  void (*v30)(); // rdx
  __int64 (__fastcall *v31)(__int64); // rax
  _DWORD *v32; // rdi
  unsigned __int8 (*v33)(void); // rax
  __int64 (__fastcall *v34)(__int64); // rax
  _DWORD *v35; // rdi
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
  v43 = &unk_438A667;
  v44 = v8;
  v10 = sub_25134D0(a1 + 136, (__int64 *)&v43);
  if ( v10 )
  {
    v11 = v10[3];
    if ( v11 )
    {
      if ( a5 != 2 && a4 )
      {
        v12 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
        if ( v12 == sub_2505F40 )
          v13 = (_DWORD *)(v11 + 88);
        else
          v13 = (_DWORD *)v12(v11);
        v14 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v13 + 16LL);
        if ( v14 == sub_2505F60 )
        {
          if ( !v13[3] )
            goto LABEL_11;
LABEL_52:
          sub_250ED80(a1, v11, a4, a5);
          if ( !a6 )
            return v11;
LABEL_12:
          if ( *(_DWORD *)(a1 + 3552) == 1 )
            sub_251C580(a1, v11);
          return v11;
        }
        if ( ((unsigned __int8 (*)(void))v14)() )
          goto LABEL_52;
      }
LABEL_11:
      if ( !a6 )
        return v11;
      goto LABEL_12;
    }
  }
  if ( (unsigned int)((char)sub_2509800(&v42) - 4) > 1 )
  {
    v16 = sub_250D180(v42.m128i_i64, (__int64)&v43);
    v17 = *(unsigned __int8 *)(v16 + 8);
    if ( (unsigned int)(v17 - 17) <= 1 )
      LOBYTE(v17) = *(_BYTE *)(**(_QWORD **)(v16 + 16) + 8LL);
    if ( (_BYTE)v17 != 14 )
      return 0;
  }
  v18 = *(_QWORD *)(a1 + 4376);
  if ( v18 )
  {
    v9 = (__int64)&v43;
    v43 = &unk_438A667;
    if ( !sub_2517B80(v18, (__int64 *)&v43) )
      return 0;
  }
  v19 = sub_25096F0(&v42);
  v20 = v19;
  if ( v19 )
  {
    if ( (unsigned __int8)sub_B2D610(v19, 20) )
      return 0;
    v9 = 48;
    if ( (unsigned __int8)sub_B2D610(v20, 48) )
      return 0;
  }
  if ( *(_DWORD *)(a1 + 3556) > dword_4FEEF68[0] )
    return 0;
  if ( (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) <= 1 )
    goto LABEL_61;
  v21 = sub_250CBE0(v42.m128i_i64, v9);
  v22 = sub_2509800(&v42);
  if ( v22 > 7u || ((1LL << v22) & 0xA8) == 0 )
  {
    if ( (unsigned __int8)sub_250CC70(a1, v42.m128i_i64) )
    {
      if ( !v21 )
      {
LABEL_36:
        v40 = 1;
        goto LABEL_37;
      }
      goto LABEL_34;
    }
LABEL_61:
    v40 = 0;
    goto LABEL_37;
  }
  if ( !v21 )
    goto LABEL_61;
  v23 = v42.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
  if ( (v42.m128i_i8[0] & 3) == 3 )
    v23 = *(_QWORD *)(v23 + 24);
  if ( **(_BYTE **)(v23 - 32) == 25 || !(unsigned __int8)sub_250CC70(a1, v42.m128i_i64) )
    goto LABEL_61;
LABEL_34:
  if ( *(_BYTE *)(a1 + 4296) || (unsigned __int8)sub_2506F10(*(_QWORD *)(a1 + 200), (__int64)v21) )
    goto LABEL_36;
  v40 = sub_2508DC0(a1, &v42);
LABEL_37:
  v11 = sub_2561CA0(&v42, a1);
  v43 = &unk_438A667;
  v44 = _mm_loadu_si128((const __m128i *)(v11 + 72));
  *sub_2519B70(a1 + 136, (__int64)&v43) = v11;
  if ( *(_DWORD *)(a1 + 3552) <= 1u )
  {
    v43 = (void *)(v11 & 0xFFFFFFFFFFFFFFFBLL);
    sub_251B630(a1 + 224, (unsigned __int64 *)&v43, v24, v25, v26, v27);
    if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v11) )
      goto LABEL_57;
  }
  v43 = (void *)v11;
  v28 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_2509D60, (__int64)&v43);
  v29 = *(_DWORD *)(a1 + 3556);
  *(_DWORD *)(a1 + 3556) = v29 + 1;
  v30 = *(void (**)())(*(_QWORD *)v11 + 24LL);
  if ( v30 != nullsub_1516 )
  {
    ((void (__fastcall *)(__int64, __int64))v30)(v11, a1);
    v29 = *(_DWORD *)(a1 + 3556) - 1;
  }
  *(_DWORD *)(a1 + 3556) = v29;
  if ( v28 )
    sub_C9AF60(v28);
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
      if ( v31 == sub_2505F40 )
        v32 = (_DWORD *)(v11 + 88);
      else
        v32 = (_DWORD *)v31(v11);
      v33 = *(unsigned __int8 (**)(void))(*(_QWORD *)v32 + 16LL);
      if ( (char *)v33 == (char *)sub_2505F60 )
      {
        if ( !v32[3] )
          return v11;
      }
      else if ( !v33() )
      {
        return v11;
      }
      sub_250ED80(a1, v11, a4, a5);
    }
  }
  else
  {
LABEL_57:
    v34 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
    if ( v34 == sub_2505F40 )
      v35 = (_DWORD *)(v11 + 88);
    else
      v35 = (_DWORD *)v34(v11);
    v36 = *(void (**)(void))(*(_QWORD *)v35 + 40LL);
    if ( (char *)v36 == (char *)sub_2505F50 )
      v35[3] = v35[2];
    else
      v36();
  }
  return v11;
}
