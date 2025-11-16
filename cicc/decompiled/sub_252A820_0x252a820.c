// Function: sub_252A820
// Address: 0x252a820
//
__int64 __fastcall sub_252A820(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v8; // xmm0
  _QWORD *v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v13; // rax
  int v14; // ecx
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r14
  int v23; // eax
  void (*v24)(); // rdx
  __int64 v25; // rax
  __int64 v26; // rax
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
  v33 = &unk_438A65F;
  v34 = v8;
  v9 = sub_25134D0(a1 + 136, (__int64 *)&v33);
  if ( v9 && (v10 = v9[3]) != 0 )
  {
    if ( a5 != 2
      && a4
      && (v11 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL))(v10),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v11 + 16LL))(v11)) )
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
    v13 = sub_250D180(v31.m128i_i64, (__int64)&v33);
    v14 = *(unsigned __int8 *)(v13 + 8);
    if ( (unsigned int)(v14 - 17) <= 1 )
      LOBYTE(v14) = *(_BYTE *)(**(_QWORD **)(v13 + 16) + 8LL);
    if ( (_BYTE)v14 != 14 )
      return 0;
    v15 = *(_QWORD *)(a1 + 4376);
    if ( v15 )
    {
      v33 = &unk_438A65F;
      if ( !sub_2517B80(v15, (__int64 *)&v33) )
        return 0;
    }
    v16 = sub_25096F0(&v31);
    v17 = v16;
    if ( v16 )
    {
      if ( (unsigned __int8)sub_B2D610(v16, 20) || (unsigned __int8)sub_B2D610(v17, 48) )
        return 0;
    }
    if ( !(unsigned __int8)sub_250CDD0(a1, v31.m128i_i64, &v32) )
    {
      return 0;
    }
    else
    {
      v10 = sub_25646A0(&v31, a1);
      v33 = &unk_438A65F;
      v34 = _mm_loadu_si128((const __m128i *)(v10 + 72));
      *sub_2519B70(a1 + 136, (__int64)&v33) = v10;
      if ( *(_DWORD *)(a1 + 3552) <= 1u )
      {
        v33 = (void *)(v10 & 0xFFFFFFFFFFFFFFFBLL);
        sub_251B630(a1 + 224, (unsigned __int64 *)&v33, v18, v19, v20, v21);
        if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v10) )
          goto LABEL_37;
      }
      v33 = (void *)v10;
      v22 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_2509C30, (__int64)&v33);
      v23 = *(_DWORD *)(a1 + 3556);
      *(_DWORD *)(a1 + 3556) = v23 + 1;
      v24 = *(void (**)())(*(_QWORD *)v10 + 24LL);
      if ( v24 != nullsub_1516 )
      {
        ((void (__fastcall *)(__int64, __int64))v24)(v10, a1);
        v23 = *(_DWORD *)(a1 + 3556) - 1;
      }
      *(_DWORD *)(a1 + 3556) = v23;
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
          v25 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL))(v10);
          if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v25 + 16LL))(v25) )
            sub_250ED80(a1, v10, a4, a5);
        }
      }
      else
      {
LABEL_37:
        v26 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL))(v10);
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v26 + 40LL))(v26);
      }
    }
  }
  return v10;
}
