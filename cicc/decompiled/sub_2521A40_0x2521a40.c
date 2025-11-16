// Function: sub_2521A40
// Address: 0x2521a40
//
void __fastcall sub_2521A40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v9; // xmm0
  void **v10; // rsi
  _QWORD *v11; // rax
  __int64 (__fastcall *v12)(__int64); // rax
  _DWORD *v13; // rdi
  bool (__fastcall *v14)(__int64); // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r15
  int v23; // eax
  void (*v24)(); // rdx
  __int64 (__fastcall *v25)(__int64); // rax
  _DWORD *v26; // rdi
  unsigned __int8 (*v27)(void); // rax
  __int64 (__fastcall *v28)(__int64); // rax
  __int64 v29; // rdi
  void (*v30)(void); // rax
  int v31; // ebx
  __int64 v33; // [rsp+10h] [rbp-80h]
  __m128i v35; // [rsp+20h] [rbp-70h] BYREF
  char v36; // [rsp+3Fh] [rbp-51h] BYREF
  void *v37; // [rsp+40h] [rbp-50h] BYREF
  __m128i v38; // [rsp+48h] [rbp-48h]

  v35.m128i_i64[0] = a2;
  v35.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v35) )
    v35.m128i_i64[1] = 0;
  v9 = _mm_load_si128(&v35);
  v10 = &v37;
  v37 = &unk_438A66E;
  v38 = v9;
  v11 = sub_25134D0(a1 + 136, (__int64 *)&v37);
  if ( v11 )
  {
    v10 = (void **)v11[3];
    if ( v10 )
    {
      if ( a5 != 2 && a4 )
      {
        v12 = (__int64 (__fastcall *)(__int64))*((_QWORD *)*v10 + 5);
        if ( v12 == sub_2505FE0 )
          v13 = v10 + 11;
        else
          v13 = (_DWORD *)v12((__int64)v10);
        v14 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v13 + 16LL);
        if ( v14 == sub_2505DB0 )
        {
          if ( !v13[5] )
            goto LABEL_11;
LABEL_36:
          sub_250ED80(a1, (__int64)v10, a4, a5);
          if ( !a6 )
            return;
          goto LABEL_12;
        }
        if ( ((unsigned __int8 (*)(void))v14)() )
          goto LABEL_36;
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
  if ( *(_BYTE *)(sub_250D180(v35.m128i_i64, (__int64)v10) + 8) == 14 )
  {
    v15 = *(_QWORD *)(a1 + 4376);
    if ( !v15 || (v37 = &unk_438A66E, sub_2517B80(v15, (__int64 *)&v37)) )
    {
      v16 = sub_25096F0(&v35);
      if ( !v16 || (v33 = v16, !(unsigned __int8)sub_B2D610(v16, 20)) && !(unsigned __int8)sub_B2D610(v33, 48) )
      {
        if ( (unsigned __int8)sub_250CDD0(a1, v35.m128i_i64, &v36) )
        {
          v17 = sub_2562CD0(&v35, a1);
          v37 = &unk_438A66E;
          v38 = _mm_loadu_si128((const __m128i *)(v17 + 72));
          *sub_2519B70(a1 + 136, (__int64)&v37) = v17;
          if ( *(_DWORD *)(a1 + 3552) <= 1u )
          {
            v37 = (void *)(v17 & 0xFFFFFFFFFFFFFFFBLL);
            sub_251B630(a1 + 224, (unsigned __int64 *)&v37, v18, v19, v20, v21);
            if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v17) )
              goto LABEL_42;
          }
          v37 = (void *)v17;
          v22 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250B700, (__int64)&v37);
          v23 = *(_DWORD *)(a1 + 3556);
          *(_DWORD *)(a1 + 3556) = v23 + 1;
          v24 = *(void (**)())(*(_QWORD *)v17 + 24LL);
          if ( v24 != nullsub_1516 )
          {
            ((void (__fastcall *)(__int64, __int64))v24)(v17, a1);
            v23 = *(_DWORD *)(a1 + 3556) - 1;
          }
          *(_DWORD *)(a1 + 3556) = v23;
          if ( v22 )
            sub_C9AF60(v22);
          if ( v36 )
          {
            if ( a7 )
            {
              v31 = *(_DWORD *)(a1 + 3552);
              *(_DWORD *)(a1 + 3552) = 1;
              sub_251C580(a1, v17);
              *(_DWORD *)(a1 + 3552) = v31;
            }
            if ( a4 )
            {
              v25 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v17 + 40LL);
              if ( v25 == sub_2505FE0 )
                v26 = (_DWORD *)(v17 + 88);
              else
                v26 = (_DWORD *)v25(v17);
              v27 = *(unsigned __int8 (**)(void))(*(_QWORD *)v26 + 16LL);
              if ( (char *)v27 == (char *)sub_2505DB0 )
              {
                if ( !v26[5] )
                  return;
              }
              else if ( !v27() )
              {
                return;
              }
              sub_250ED80(a1, v17, a4, a5);
            }
          }
          else
          {
LABEL_42:
            v28 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v17 + 40LL);
            if ( v28 == sub_2505FE0 )
              v29 = v17 + 88;
            else
              v29 = v28(v17);
            v30 = *(void (**)(void))(*(_QWORD *)v29 + 40LL);
            if ( (char *)v30 == (char *)sub_2505DC0 )
            {
              *(_DWORD *)(v29 + 20) = *(_DWORD *)(v29 + 16);
              *(_BYTE *)(v29 + 81) = *(_BYTE *)(v29 + 80);
            }
            else
            {
              v30();
            }
          }
        }
      }
    }
  }
}
