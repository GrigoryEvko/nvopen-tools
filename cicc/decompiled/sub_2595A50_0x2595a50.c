// Function: sub_2595A50
// Address: 0x2595a50
//
__int64 __fastcall sub_2595A50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v8; // xmm0
  _QWORD *v9; // rax
  __int64 v10; // r13
  __int64 (__fastcall *v11)(__int64); // rax
  _BYTE *v12; // rdi
  __int64 (__fastcall *v13)(__int64); // rax
  char v14; // al
  _BYTE *v16; // rax
  __int64 v17; // r13
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
  _BYTE *v28; // rdi
  __int64 (__fastcall *v29)(__int64); // rax
  __int64 (__fastcall *v31)(__int64); // rax
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
  v39 = &unk_438A65A;
  v40 = v8;
  v9 = sub_25134D0(a1 + 136, (__int64 *)&v39);
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
    if ( (unsigned __int8)sub_2509800(&v37) != 5 )
      return 0;
    v16 = (_BYTE *)sub_2509740(&v37);
    v17 = (__int64)v16;
    if ( *v16 != 85 )
      return 0;
    if ( !sub_B491E0((__int64)v16) )
      return 0;
    if ( sub_B49200(v17) )
      return 0;
    v18 = *(_QWORD *)(a1 + 4376);
    if ( v18 )
    {
      v39 = &unk_438A65A;
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
    {
      return 0;
    }
    else
    {
      v10 = sub_2565F60(&v37, a1);
      v39 = &unk_438A65A;
      v40 = _mm_loadu_si128((const __m128i *)(v10 + 72));
      *sub_2519B70(a1 + 136, (__int64)&v39) = v10;
      if ( *(_DWORD *)(a1 + 3552) <= 1u )
      {
        v39 = (void *)(v10 & 0xFFFFFFFFFFFFFFFBLL);
        sub_257A300(a1 + 224, (unsigned __int64 *)&v39, v21, v22, v23, v24);
        if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v10) )
          goto LABEL_46;
      }
      v39 = (void *)v10;
      v25 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250BD00, (__int64)&v39);
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
          v28 = (_BYTE *)(v27 == sub_2505F20 ? v10 + 88 : v27(v10));
          v29 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v28 + 16LL);
          if ( v29 == sub_2505E30 ? v28[9] : ((__int64 (*)(void))v29)() )
            sub_250ED80(a1, v10, a4, a5);
        }
      }
      else
      {
LABEL_46:
        v31 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL);
        if ( v31 == sub_2505F20 )
          v32 = v10 + 88;
        else
          v32 = v31(v10);
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v32 + 40LL))(v32);
      }
    }
  }
  return v10;
}
