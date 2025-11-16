// Function: sub_258DCE0
// Address: 0x258dce0
//
__int64 __fastcall sub_258DCE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v8; // xmm0
  _QWORD *v9; // rax
  __int64 v10; // r13
  _DWORD *v11; // rdi
  bool (__fastcall *v12)(__int64); // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // r14
  _DWORD *v23; // rdi
  bool (__fastcall *v24)(__int64); // rax
  __int64 v25; // rdi
  __int64 (__fastcall *v26)(__int64); // rax
  int v27; // ebx
  __m128i v31; // [rsp+20h] [rbp-70h] BYREF
  char v32; // [rsp+3Fh] [rbp-51h] BYREF
  void *v33; // [rsp+40h] [rbp-50h] BYREF
  __m128i v34; // [rsp+48h] [rbp-48h]

  v31.m128i_i64[0] = a2;
  v31.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v31) )
    v31.m128i_i64[1] = 0;
  v8 = _mm_load_si128(&v31);
  v33 = &unk_438A66E;
  v34 = v8;
  v9 = sub_25134D0(a1 + 136, (__int64 *)&v33);
  if ( v9 )
  {
    v10 = v9[3];
    if ( v10 )
    {
      if ( a5 != 2 && a4 )
      {
        v11 = (_DWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL))(v10);
        v12 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v11 + 16LL);
        if ( v12 == sub_2505DB0 )
        {
          if ( !v11[5] )
            goto LABEL_9;
LABEL_20:
          sub_250ED80(a1, v10, a4, a5);
          if ( !a6 )
            return v10;
LABEL_10:
          if ( *(_DWORD *)(a1 + 3552) == 1 )
            sub_251C580(a1, v10);
          return v10;
        }
        if ( v12((__int64)v11) )
          goto LABEL_20;
      }
LABEL_9:
      if ( !a6 )
        return v10;
      goto LABEL_10;
    }
  }
  if ( *(_BYTE *)(sub_250D180(v31.m128i_i64, (__int64)&v33) + 8) != 14 )
    return 0;
  v14 = *(_QWORD *)(a1 + 4376);
  if ( v14 )
  {
    v33 = &unk_438A66E;
    if ( !sub_2517B80(v14, (__int64 *)&v33) )
      return 0;
  }
  v15 = sub_25096F0(&v31);
  v16 = v15;
  if ( v15 )
  {
    if ( (unsigned __int8)sub_B2D610(v15, 20) || (unsigned __int8)sub_B2D610(v16, 48) )
      return 0;
  }
  if ( !(unsigned __int8)sub_254F6C0(a1, v31.m128i_i64, &v32) )
    return 0;
  v10 = sub_2562CD0(&v31, a1);
  v33 = &unk_438A66E;
  v34 = _mm_loadu_si128((const __m128i *)(v10 + 72));
  *sub_2519B70(a1 + 136, (__int64)&v33) = v10;
  if ( *(_DWORD *)(a1 + 3552) <= 1u )
  {
    v33 = (void *)(v10 & 0xFFFFFFFFFFFFFFFBLL);
    sub_257A300(a1 + 224, (unsigned __int64 *)&v33, v17, v18, v19, v20);
    if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v10) )
      goto LABEL_36;
  }
  v33 = (void *)v10;
  v21 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250B700, (__int64)&v33);
  ++*(_DWORD *)(a1 + 3556);
  v22 = v21;
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v10 + 24LL))(v10, a1);
  --*(_DWORD *)(a1 + 3556);
  if ( v22 )
    sub_C9AF60(v22);
  if ( v32 )
  {
    if ( a7 )
    {
      v27 = *(_DWORD *)(a1 + 3552);
      *(_DWORD *)(a1 + 3552) = 1;
      sub_251C580(a1, v10);
      *(_DWORD *)(a1 + 3552) = v27;
    }
    if ( a4 )
    {
      v23 = (_DWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL))(v10);
      v24 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v23 + 16LL);
      if ( v24 == sub_2505DB0 )
      {
        if ( !v23[5] )
          return v10;
      }
      else if ( !v24((__int64)v23) )
      {
        return v10;
      }
      sub_250ED80(a1, v10, a4, a5);
    }
  }
  else
  {
LABEL_36:
    v25 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL))(v10);
    v26 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v25 + 40LL);
    if ( v26 == sub_2505DC0 )
    {
      *(_DWORD *)(v25 + 20) = *(_DWORD *)(v25 + 16);
      *(_BYTE *)(v25 + 81) = *(_BYTE *)(v25 + 80);
    }
    else
    {
      v26(v25);
    }
  }
  return v10;
}
