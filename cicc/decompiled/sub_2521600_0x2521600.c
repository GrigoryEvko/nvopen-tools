// Function: sub_2521600
// Address: 0x2521600
//
void __fastcall sub_2521600(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v9; // xmm0
  void **v10; // rsi
  _QWORD *v11; // rax
  __int64 (__fastcall *v12)(__int64); // rax
  _DWORD *v13; // rdi
  bool (__fastcall *v14)(__int64); // rax
  __int64 i; // rsi
  int v16; // ecx
  unsigned __int8 v17; // al
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // r14
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r15
  int v26; // eax
  void (*v27)(); // rdx
  __int64 (__fastcall *v28)(__int64); // rax
  _DWORD *v29; // rdi
  unsigned __int8 (*v30)(void); // rax
  __int64 (__fastcall *v31)(__int64); // rax
  _DWORD *v32; // rdi
  void (*v33)(void); // rax
  int v34; // ebx
  __int64 v36; // [rsp+10h] [rbp-80h]
  __m128i v38; // [rsp+20h] [rbp-70h] BYREF
  char v39; // [rsp+3Fh] [rbp-51h] BYREF
  void *v40; // [rsp+40h] [rbp-50h] BYREF
  __m128i v41; // [rsp+48h] [rbp-48h]

  v38.m128i_i64[0] = a2;
  v38.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v38) )
    v38.m128i_i64[1] = 0;
  v9 = _mm_load_si128(&v38);
  v10 = &v40;
  v40 = &unk_438A662;
  v41 = v9;
  v11 = sub_25134D0(a1 + 136, (__int64 *)&v40);
  if ( v11 )
  {
    v10 = (void **)v11[3];
    if ( v10 )
    {
      if ( a5 != 2 && a4 )
      {
        v12 = (__int64 (__fastcall *)(__int64))*((_QWORD *)*v10 + 5);
        if ( v12 == sub_2505FF0 )
          v13 = v10 + 11;
        else
          v13 = (_DWORD *)v12((__int64)v10);
        v14 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v13 + 16LL);
        if ( v14 == sub_2506010 )
        {
          if ( !v13[3] )
            goto LABEL_11;
LABEL_43:
          sub_250ED80(a1, (__int64)v10, a4, a5);
          if ( !a6 )
            return;
          goto LABEL_12;
        }
        if ( ((unsigned __int8 (*)(void))v14)() )
          goto LABEL_43;
      }
LABEL_11:
      if ( !a6 )
        return;
LABEL_12:
      if ( *(_DWORD *)(a1 + 3552) == 1 )
        sub_251C580(a1, (__int64)v10);
      return;
    }
  }
  for ( i = sub_250D180(v38.m128i_i64, (__int64)v10); ; i = **(_QWORD **)(i + 16) )
  {
    v16 = *(unsigned __int8 *)(i + 8);
    v17 = *(_BYTE *)(i + 8);
    if ( (unsigned int)(v16 - 17) <= 1 )
      v17 = *(_BYTE *)(**(_QWORD **)(i + 16) + 8LL);
    if ( v17 <= 3u || v17 == 5 || (v17 & 0xFD) == 4 )
      break;
    if ( (_BYTE)v16 != 16 )
      return;
  }
  v18 = *(_QWORD *)(a1 + 4376);
  if ( !v18 || (v40 = &unk_438A662, sub_2517B80(v18, (__int64 *)&v40)) )
  {
    v19 = sub_25096F0(&v38);
    if ( !v19 || (v36 = v19, !(unsigned __int8)sub_B2D610(v19, 20)) && !(unsigned __int8)sub_B2D610(v36, 48) )
    {
      if ( (unsigned __int8)sub_250CDD0(a1, v38.m128i_i64, &v39) )
      {
        v20 = sub_2564490(&v38, a1);
        v40 = &unk_438A662;
        v41 = _mm_loadu_si128((const __m128i *)(v20 + 72));
        *sub_2519B70(a1 + 136, (__int64)&v40) = v20;
        if ( *(_DWORD *)(a1 + 3552) <= 1u )
        {
          v40 = (void *)(v20 & 0xFFFFFFFFFFFFFFFBLL);
          sub_251B630(a1 + 224, (unsigned __int64 *)&v40, v21, v22, v23, v24);
          if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v20) )
            goto LABEL_48;
        }
        v40 = (void *)v20;
        v25 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250BBD0, (__int64)&v40);
        v26 = *(_DWORD *)(a1 + 3556);
        *(_DWORD *)(a1 + 3556) = v26 + 1;
        v27 = *(void (**)())(*(_QWORD *)v20 + 24LL);
        if ( v27 != nullsub_1516 )
        {
          ((void (__fastcall *)(__int64, __int64))v27)(v20, a1);
          v26 = *(_DWORD *)(a1 + 3556) - 1;
        }
        *(_DWORD *)(a1 + 3556) = v26;
        if ( v25 )
          sub_C9AF60(v25);
        if ( v39 )
        {
          if ( a7 )
          {
            v34 = *(_DWORD *)(a1 + 3552);
            *(_DWORD *)(a1 + 3552) = 1;
            sub_251C580(a1, v20);
            *(_DWORD *)(a1 + 3552) = v34;
          }
          if ( a4 )
          {
            v28 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v20 + 40LL);
            if ( v28 == sub_2505FF0 )
              v29 = (_DWORD *)(v20 + 88);
            else
              v29 = (_DWORD *)v28(v20);
            v30 = *(unsigned __int8 (**)(void))(*(_QWORD *)v29 + 16LL);
            if ( (char *)v30 == (char *)sub_2506010 )
            {
              if ( !v29[3] )
                return;
            }
            else if ( !v30() )
            {
              return;
            }
            sub_250ED80(a1, v20, a4, a5);
          }
        }
        else
        {
LABEL_48:
          v31 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v20 + 40LL);
          if ( v31 == sub_2505FF0 )
            v32 = (_DWORD *)(v20 + 88);
          else
            v32 = (_DWORD *)v31(v20);
          v33 = *(void (**)(void))(*(_QWORD *)v32 + 40LL);
          if ( (char *)v33 == (char *)sub_2506000 )
            v32[3] = v32[2];
          else
            v33();
        }
      }
    }
  }
}
