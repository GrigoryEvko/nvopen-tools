// Function: sub_252BBE0
// Address: 0x252bbe0
//
__int64 __fastcall sub_252BBE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
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
  __int64 v29; // r14
  int v30; // eax
  void (*v31)(); // rdx
  __int64 (__fastcall *v32)(__int64); // rax
  _BYTE *v33; // rdi
  __int64 (__fastcall *v34)(__int64); // rax
  __int64 (__fastcall *v36)(__int64); // rax
  _BYTE *v37; // rdi
  void (*v38)(void); // rax
  int v39; // ebx
  __m128i v43; // [rsp+20h] [rbp-60h] BYREF
  void *v44; // [rsp+30h] [rbp-50h] BYREF
  __m128i v45; // [rsp+38h] [rbp-48h]

  v43.m128i_i64[0] = a2;
  v43.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v43) )
    v43.m128i_i64[1] = 0;
  v8 = _mm_load_si128(&v43);
  v9 = (__int64)&v44;
  v44 = &unk_438A67A;
  v45 = v8;
  v10 = sub_25134D0(a1 + 136, (__int64 *)&v44);
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
    if ( (unsigned int)((char)sub_2509800(&v43) - 4) > 1 )
    {
      v17 = sub_250D180(v43.m128i_i64, (__int64)&v44);
      v18 = *(unsigned __int8 *)(v17 + 8);
      if ( (unsigned int)(v18 - 17) <= 1 )
        LOBYTE(v18) = *(_BYTE *)(**(_QWORD **)(v17 + 16) + 8LL);
      if ( (_BYTE)v18 != 14 )
        return 0;
    }
    v19 = *(_QWORD *)(a1 + 4376);
    if ( v19 )
    {
      v9 = (__int64)&v44;
      v44 = &unk_438A67A;
      if ( !sub_2517B80(v19, (__int64 *)&v44) )
        return 0;
    }
    v20 = sub_25096F0(&v43);
    v21 = v20;
    if ( v20 )
    {
      if ( (unsigned __int8)sub_B2D610(v20, 20) )
        return 0;
      v9 = 48;
      if ( (unsigned __int8)sub_B2D610(v21, 48) )
        return 0;
    }
    if ( *(_DWORD *)(a1 + 3556) > dword_4FEEF68[0] || (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) <= 1 )
      return 0;
    v22 = sub_250CBE0(v43.m128i_i64, v9);
    v23 = sub_2509800(&v43);
    if ( v23 <= 7u && ((1LL << v23) & 0xA8) != 0 )
    {
      v24 = v43.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
      if ( (v43.m128i_i8[0] & 3) == 3 )
        v24 = *(_QWORD *)(v24 + 24);
      if ( **(_BYTE **)(v24 - 32) == 25 )
        return 0;
    }
    if ( !(unsigned __int8)sub_250CC70(a1, v43.m128i_i64)
      || v22
      && !*(_BYTE *)(a1 + 4296)
      && !(unsigned __int8)sub_2506F10(*(_QWORD *)(a1 + 200), (__int64)v22)
      && !(unsigned __int8)sub_2508DC0(a1, &v43) )
    {
      return 0;
    }
    else
    {
      v11 = sub_25617A0(&v43, a1);
      v44 = &unk_438A67A;
      v45 = _mm_loadu_si128((const __m128i *)(v11 + 72));
      *sub_2519B70(a1 + 136, (__int64)&v44) = v11;
      if ( *(_DWORD *)(a1 + 3552) > 1u
        || (v44 = (void *)(v11 & 0xFFFFFFFFFFFFFFFBLL),
            sub_251B630(a1 + 224, (unsigned __int64 *)&v44, v25, v26, v27, v28),
            *(_DWORD *)(a1 + 3552))
        || (unsigned __int8)sub_250E880(a1, v11) )
      {
        v44 = (void *)v11;
        v29 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_2509890, (__int64)&v44);
        v30 = *(_DWORD *)(a1 + 3556);
        *(_DWORD *)(a1 + 3556) = v30 + 1;
        v31 = *(void (**)())(*(_QWORD *)v11 + 24LL);
        if ( v31 != nullsub_1516 )
        {
          ((void (__fastcall *)(__int64, __int64))v31)(v11, a1);
          v30 = *(_DWORD *)(a1 + 3556) - 1;
        }
        *(_DWORD *)(a1 + 3556) = v30;
        if ( v29 )
          sub_C9AF60(v29);
        if ( a7 )
        {
          v39 = *(_DWORD *)(a1 + 3552);
          *(_DWORD *)(a1 + 3552) = 1;
          sub_251C580(a1, v11);
          *(_DWORD *)(a1 + 3552) = v39;
        }
        if ( a4 )
        {
          v32 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
          v33 = (_BYTE *)(v32 == sub_2505F20 ? v11 + 88 : v32(v11));
          v34 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v33 + 16LL);
          if ( v34 == sub_2505E30 ? v33[9] : ((__int64 (*)(void))v34)() )
            sub_250ED80(a1, v11, a4, a5);
        }
      }
      else
      {
        v36 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
        if ( v36 == sub_2505F20 )
          v37 = (_BYTE *)(v11 + 88);
        else
          v37 = (_BYTE *)v36(v11);
        v38 = *(void (**)(void))(*(_QWORD *)v37 + 40LL);
        if ( (char *)v38 == (char *)sub_2505E20 )
          v37[9] = v37[8];
        else
          v38();
      }
    }
  }
  return v11;
}
