// Function: sub_2522250
// Address: 0x2522250
//
void __fastcall sub_2522250(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v9; // xmm0
  _QWORD *v10; // rax
  __int64 v11; // rsi
  __int64 (__fastcall *v12)(__int64); // rax
  __int64 v13; // rdi
  unsigned __int8 (*v14)(void); // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r14
  int v23; // eax
  void (*v24)(); // rdx
  __int64 (__fastcall *v25)(__int64); // rax
  __int64 v26; // rdi
  unsigned __int8 (*v27)(void); // rax
  __int64 (__fastcall *v28)(__int64); // rax
  __int64 v29; // r15
  __int64 (__fastcall *v30)(__int64); // rax
  char v31; // al
  __int64 v32; // rsi
  __int64 v33; // rdi
  unsigned int v34; // eax
  void *v35; // rax
  __int64 v36; // rdx
  const void *v37; // rsi
  int v38; // ebx
  __int64 v40; // [rsp+10h] [rbp-80h]
  __m128i v42; // [rsp+20h] [rbp-70h] BYREF
  char v43; // [rsp+3Fh] [rbp-51h] BYREF
  void *v44; // [rsp+40h] [rbp-50h] BYREF
  __m128i v45; // [rsp+48h] [rbp-48h]

  v42.m128i_i64[0] = a2;
  v42.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v42) )
    v42.m128i_i64[1] = 0;
  v9 = _mm_load_si128(&v42);
  v44 = &unk_438A65E;
  v45 = v9;
  v10 = sub_25134D0(a1 + 136, (__int64 *)&v44);
  if ( v10 )
  {
    v11 = v10[3];
    if ( v11 )
    {
      if ( a5 != 2 && a4 )
      {
        v12 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
        if ( v12 == sub_2505F70 )
          v13 = v11 + 88;
        else
          v13 = v12(v11);
        v14 = *(unsigned __int8 (**)(void))(*(_QWORD *)v13 + 16LL);
        if ( (char *)v14 == (char *)sub_2505F80 )
        {
          if ( *(_DWORD *)(v13 + 72) || *(_BYTE *)(v13 + 48) )
            goto LABEL_12;
        }
        else if ( v14() )
        {
LABEL_12:
          sub_250ED80(a1, v11, a4, a5);
        }
      }
      if ( a6 )
      {
        if ( *(_DWORD *)(a1 + 3552) == 1 )
          sub_251C580(a1, v11);
      }
      return;
    }
  }
  v15 = *(_QWORD *)(a1 + 4376);
  if ( !v15 || (v44 = &unk_438A65E, sub_2517B80(v15, (__int64 *)&v44)) )
  {
    v16 = sub_25096F0(&v42);
    if ( !v16 || (v40 = v16, !(unsigned __int8)sub_B2D610(v16, 20)) && !(unsigned __int8)sub_B2D610(v40, 48) )
    {
      if ( (unsigned __int8)sub_250CDD0(a1, v42.m128i_i64, &v43) )
      {
        v17 = sub_25780C0(&v42, a1);
        v44 = &unk_438A65E;
        v45 = _mm_loadu_si128((const __m128i *)(v17 + 72));
        *sub_2519B70(a1 + 136, (__int64)&v44) = v17;
        if ( *(_DWORD *)(a1 + 3552) <= 1u )
        {
          v44 = (void *)(v17 & 0xFFFFFFFFFFFFFFFBLL);
          sub_251B630(a1 + 224, (unsigned __int64 *)&v44, v18, v19, v20, v21);
          if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v17) )
            goto LABEL_40;
        }
        v44 = (void *)v17;
        v22 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250B0F0, (__int64)&v44);
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
        if ( v43 )
        {
          if ( a7 )
          {
            v38 = *(_DWORD *)(a1 + 3552);
            *(_DWORD *)(a1 + 3552) = 1;
            sub_251C580(a1, v17);
            *(_DWORD *)(a1 + 3552) = v38;
          }
          if ( a4 )
          {
            v25 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v17 + 40LL);
            if ( v25 == sub_2505F70 )
              v26 = v17 + 88;
            else
              v26 = v25(v17);
            v27 = *(unsigned __int8 (**)(void))(*(_QWORD *)v26 + 16LL);
            if ( (char *)v27 == (char *)sub_2505F80 )
            {
              if ( !*(_DWORD *)(v26 + 72) && !*(_BYTE *)(v26 + 48) )
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
LABEL_40:
          v28 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v17 + 40LL);
          if ( v28 == sub_2505F70 )
            v29 = v17 + 88;
          else
            v29 = v28(v17);
          v30 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v29 + 40LL);
          if ( v30 == sub_2506110 )
          {
            v31 = *(_BYTE *)(v29 + 8);
            v32 = *(unsigned int *)(v29 + 80);
            *(_BYTE *)(v29 + 88) = 1;
            v33 = *(_QWORD *)(v29 + 64);
            *(_BYTE *)(v29 + 48) = v31;
            sub_C7D6A0(v33, 16 * v32, 8);
            v34 = *(_DWORD *)(v29 + 40);
            *(_DWORD *)(v29 + 80) = v34;
            if ( v34 )
            {
              v35 = (void *)sub_C7D670(16LL * v34, 8);
              v36 = *(unsigned int *)(v29 + 80);
              v37 = *(const void **)(v29 + 24);
              *(_QWORD *)(v29 + 64) = v35;
              *(_QWORD *)(v29 + 72) = *(_QWORD *)(v29 + 32);
              memcpy(v35, v37, 16 * v36);
            }
            else
            {
              *(_QWORD *)(v29 + 64) = 0;
              *(_QWORD *)(v29 + 72) = 0;
            }
          }
          else
          {
            v30(v29);
          }
        }
      }
    }
  }
}
