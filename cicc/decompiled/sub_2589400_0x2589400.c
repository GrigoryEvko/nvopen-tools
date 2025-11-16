// Function: sub_2589400
// Address: 0x2589400
//
__int64 __fastcall sub_2589400(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v8; // xmm0
  _QWORD *v9; // rax
  __int64 v10; // r13
  __int64 (__fastcall *v11)(__int64); // rax
  _DWORD *v12; // rdi
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
  _DWORD *v25; // rdi
  unsigned __int8 (*v26)(void); // rax
  __int64 (__fastcall *v27)(__int64); // rax
  __int64 v28; // r12
  __int64 (__fastcall *v29)(__int64); // rax
  unsigned int v30; // eax
  __int64 v31; // rdx
  unsigned int v32; // eax
  __int64 v33; // rdx
  int v34; // ebx
  char v37; // [rsp+18h] [rbp-68h]
  __m128i v39; // [rsp+20h] [rbp-60h] BYREF
  void *v40; // [rsp+30h] [rbp-50h] BYREF
  __m128i v41; // [rsp+38h] [rbp-48h]

  v39.m128i_i64[0] = a2;
  v39.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v39) )
    v39.m128i_i64[1] = 0;
  v8 = _mm_load_si128(&v39);
  v40 = &unk_438A666;
  v41 = v8;
  v9 = sub_25134D0(a1 + 136, (__int64 *)&v40);
  if ( v9 )
  {
    v10 = v9[3];
    if ( v10 )
    {
      if ( a5 != 2 && a4 )
      {
        v11 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL);
        if ( v11 == sub_2534AB0 )
          v12 = (_DWORD *)(v10 + 88);
        else
          v12 = (_DWORD *)v11(v10);
        v13 = *(unsigned __int8 (**)(void))(*(_QWORD *)v12 + 16LL);
        if ( (char *)v13 == (char *)sub_2535A50 )
        {
          if ( v12[2] && !sub_AAF760((__int64)(v12 + 4)) )
          {
LABEL_22:
            sub_250ED80(a1, v10, a4, a5);
            if ( !a6 )
              return v10;
LABEL_12:
            if ( *(_DWORD *)(a1 + 3552) == 1 )
              sub_251C580(a1, v10);
            return v10;
          }
        }
        else if ( v13() )
        {
          goto LABEL_22;
        }
      }
      if ( !a6 )
        return v10;
      goto LABEL_12;
    }
  }
  if ( *(_BYTE *)(sub_250D180(v39.m128i_i64, (__int64)&v40) + 8) != 12 )
    return 0;
  v15 = *(_QWORD *)(a1 + 4376);
  if ( v15 )
  {
    v40 = &unk_438A666;
    if ( !sub_2517B80(v15, (__int64 *)&v40) )
      return 0;
  }
  v16 = sub_25096F0(&v39);
  v17 = v16;
  if ( v16 )
  {
    if ( (unsigned __int8)sub_B2D610(v16, 20) || (unsigned __int8)sub_B2D610(v17, 48) )
      return 0;
  }
  if ( *(_DWORD *)(a1 + 3556) > dword_4FEEF68[0] )
    return 0;
  v37 = sub_2561480(a1, v39.m128i_i64);
  v10 = sub_25636F0(&v39, a1);
  v40 = &unk_438A666;
  v41 = _mm_loadu_si128((const __m128i *)(v10 + 72));
  *sub_2519B70(a1 + 136, (__int64)&v40) = v10;
  if ( *(_DWORD *)(a1 + 3552) <= 1u )
  {
    v40 = (void *)(v10 & 0xFFFFFFFFFFFFFFFBLL);
    sub_257A300(a1 + 224, (unsigned __int64 *)&v40, v18, v19, v20, v21);
    if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v10) )
      goto LABEL_44;
  }
  v40 = (void *)v10;
  v22 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_2554AF0, (__int64)&v40);
  ++*(_DWORD *)(a1 + 3556);
  v23 = v22;
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v10 + 24LL))(v10, a1);
  --*(_DWORD *)(a1 + 3556);
  if ( v23 )
    sub_C9AF60(v23);
  if ( v37 )
  {
    if ( a7 )
    {
      v34 = *(_DWORD *)(a1 + 3552);
      *(_DWORD *)(a1 + 3552) = 1;
      sub_251C580(a1, v10);
      *(_DWORD *)(a1 + 3552) = v34;
    }
    if ( a4 )
    {
      v24 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL);
      if ( v24 == sub_2534AB0 )
        v25 = (_DWORD *)(v10 + 88);
      else
        v25 = (_DWORD *)v24(v10);
      v26 = *(unsigned __int8 (**)(void))(*(_QWORD *)v25 + 16LL);
      if ( (char *)v26 == (char *)sub_2535A50 )
      {
        if ( !v25[2] || sub_AAF760((__int64)(v25 + 4)) )
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
LABEL_44:
    v27 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL);
    if ( v27 == sub_2534AB0 )
      v28 = v10 + 88;
    else
      v28 = v27(v10);
    v29 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v28 + 40LL);
    if ( v29 == sub_2539D60 )
    {
      if ( *(_DWORD *)(v28 + 24) > 0x40u || (v30 = *(_DWORD *)(v28 + 56), v30 > 0x40) )
      {
        sub_C43990(v28 + 16, v28 + 48);
      }
      else
      {
        v31 = *(_QWORD *)(v28 + 48);
        *(_DWORD *)(v28 + 24) = v30;
        *(_QWORD *)(v28 + 16) = v31;
      }
      if ( *(_DWORD *)(v28 + 40) > 0x40u || (v32 = *(_DWORD *)(v28 + 72), v32 > 0x40) )
      {
        sub_C43990(v28 + 32, v28 + 64);
      }
      else
      {
        v33 = *(_QWORD *)(v28 + 64);
        *(_DWORD *)(v28 + 40) = v32;
        *(_QWORD *)(v28 + 32) = v33;
      }
    }
    else
    {
      v29(v28);
    }
  }
  return v10;
}
