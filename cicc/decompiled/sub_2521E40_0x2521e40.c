// Function: sub_2521E40
// Address: 0x2521e40
//
void __fastcall sub_2521E40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v9; // xmm0
  void **v10; // rsi
  _QWORD *v11; // rax
  __int64 (__fastcall *v12)(__int64); // rax
  _QWORD *v13; // rdi
  bool (__fastcall *v14)(__int64); // rax
  __int64 v15; // rax
  int v16; // ecx
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // r15
  int v25; // eax
  void (*v26)(); // rdx
  __int64 (__fastcall *v27)(__int64); // rax
  _QWORD *v28; // rdi
  unsigned __int8 (*v29)(void); // rax
  __int64 (__fastcall *v30)(__int64); // rax
  _QWORD *v31; // rdi
  void (*v32)(void); // rax
  int v33; // ebx
  __int64 v35; // [rsp+10h] [rbp-80h]
  __m128i v37; // [rsp+20h] [rbp-70h] BYREF
  char v38; // [rsp+3Fh] [rbp-51h] BYREF
  void *v39; // [rsp+40h] [rbp-50h] BYREF
  __m128i v40; // [rsp+48h] [rbp-48h]

  v37.m128i_i64[0] = a2;
  v37.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v37) )
    v37.m128i_i64[1] = 0;
  v9 = _mm_load_si128(&v37);
  v10 = &v39;
  v39 = &unk_438A66D;
  v40 = v9;
  v11 = sub_25134D0(a1 + 136, (__int64 *)&v39);
  if ( v11 )
  {
    v10 = (void **)v11[3];
    if ( v10 )
    {
      if ( a5 != 2 && a4 )
      {
        v12 = (__int64 (__fastcall *)(__int64))*((_QWORD *)*v10 + 5);
        if ( v12 == sub_2505FB0 )
          v13 = v10 + 11;
        else
          v13 = (_QWORD *)v12((__int64)v10);
        v14 = *(bool (__fastcall **)(__int64))(*v13 + 16LL);
        if ( v14 == sub_2505FD0 )
        {
          if ( v13[2] == 1 )
            goto LABEL_11;
LABEL_38:
          sub_250ED80(a1, (__int64)v10, a4, a5);
          if ( !a6 )
            return;
          goto LABEL_12;
        }
        if ( ((unsigned __int8 (*)(void))v14)() )
          goto LABEL_38;
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
  v15 = sub_250D180(v37.m128i_i64, (__int64)v10);
  v16 = *(unsigned __int8 *)(v15 + 8);
  if ( (unsigned int)(v16 - 17) <= 1 )
    LOBYTE(v16) = *(_BYTE *)(**(_QWORD **)(v15 + 16) + 8LL);
  if ( (_BYTE)v16 == 14 )
  {
    v17 = *(_QWORD *)(a1 + 4376);
    if ( !v17 || (v39 = &unk_438A66D, sub_2517B80(v17, (__int64 *)&v39)) )
    {
      v18 = sub_25096F0(&v37);
      if ( !v18 || (v35 = v18, !(unsigned __int8)sub_B2D610(v18, 20)) && !(unsigned __int8)sub_B2D610(v35, 48) )
      {
        if ( (unsigned __int8)sub_250CDD0(a1, v37.m128i_i64, &v38) )
        {
          v19 = sub_25630E0(&v37, a1);
          v39 = &unk_438A66D;
          v40 = _mm_loadu_si128((const __m128i *)(v19 + 72));
          *sub_2519B70(a1 + 136, (__int64)&v39) = v19;
          if ( *(_DWORD *)(a1 + 3552) <= 1u )
          {
            v39 = (void *)(v19 & 0xFFFFFFFFFFFFFFFBLL);
            sub_251B630(a1 + 224, (unsigned __int64 *)&v39, v20, v21, v22, v23);
            if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v19) )
              goto LABEL_43;
          }
          v39 = (void *)v19;
          v24 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250B840, (__int64)&v39);
          v25 = *(_DWORD *)(a1 + 3556);
          *(_DWORD *)(a1 + 3556) = v25 + 1;
          v26 = *(void (**)())(*(_QWORD *)v19 + 24LL);
          if ( v26 != nullsub_1516 )
          {
            ((void (__fastcall *)(__int64, __int64))v26)(v19, a1);
            v25 = *(_DWORD *)(a1 + 3556) - 1;
          }
          *(_DWORD *)(a1 + 3556) = v25;
          if ( v24 )
            sub_C9AF60(v24);
          if ( v38 )
          {
            if ( a7 )
            {
              v33 = *(_DWORD *)(a1 + 3552);
              *(_DWORD *)(a1 + 3552) = 1;
              sub_251C580(a1, v19);
              *(_DWORD *)(a1 + 3552) = v33;
            }
            if ( a4 )
            {
              v27 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v19 + 40LL);
              if ( v27 == sub_2505FB0 )
                v28 = (_QWORD *)(v19 + 88);
              else
                v28 = (_QWORD *)v27(v19);
              v29 = *(unsigned __int8 (**)(void))(*v28 + 16LL);
              if ( (char *)v29 == (char *)sub_2505FD0 )
              {
                if ( v28[2] == 1 )
                  return;
              }
              else if ( !v29() )
              {
                return;
              }
              sub_250ED80(a1, v19, a4, a5);
            }
          }
          else
          {
LABEL_43:
            v30 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v19 + 40LL);
            if ( v30 == sub_2505FB0 )
              v31 = (_QWORD *)(v19 + 88);
            else
              v31 = (_QWORD *)v30(v19);
            v32 = *(void (**)(void))(*v31 + 40LL);
            if ( (char *)v32 == (char *)sub_2505FC0 )
              v31[2] = v31[1];
            else
              v32();
          }
        }
      }
    }
  }
}
