// Function: sub_269D9A0
// Address: 0x269d9a0
//
void __fastcall sub_269D9A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v9; // xmm0
  __int64 v10; // rsi
  _QWORD *v11; // rax
  __int64 (__fastcall *v12)(__int64); // rax
  _BYTE *v13; // rdi
  __int64 (__fastcall *v14)(__int64); // rax
  char v15; // al
  __int64 v16; // rcx
  int v17; // edx
  __int64 v18; // rax
  __int64 v19; // rcx
  int v20; // eax
  int v21; // edx
  int v22; // edi
  unsigned int v23; // eax
  __int64 v24; // rax
  unsigned __int8 v25; // cl
  unsigned __int64 v26; // rdx
  __int64 v27; // r14
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r15
  int v33; // eax
  void (*v34)(); // rdx
  __int64 (__fastcall *v35)(__int64); // rax
  _BYTE *v36; // rdi
  __int64 (__fastcall *v37)(__int64); // rax
  __int64 (__fastcall *v39)(__int64); // rax
  __int64 v40; // rdi
  int v41; // ebx
  __int64 v42; // rax
  __int64 v44; // [rsp+10h] [rbp-70h]
  unsigned __int8 *v45; // [rsp+10h] [rbp-70h]
  char v46; // [rsp+10h] [rbp-70h]
  __m128i v48; // [rsp+20h] [rbp-60h] BYREF
  void *v49; // [rsp+30h] [rbp-50h] BYREF
  __m128i v50; // [rsp+38h] [rbp-48h]

  v48.m128i_i64[0] = a2;
  v48.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v48) )
    v48.m128i_i64[1] = 0;
  v9 = _mm_load_si128(&v48);
  v10 = (__int64)&v49;
  v49 = &unk_438A65C;
  v50 = v9;
  v11 = sub_25134D0(a1 + 136, (__int64 *)&v49);
  if ( !v11 || (v10 = v11[3]) == 0 )
  {
    if ( (unsigned __int8)sub_2509800(&v48) == 2 )
      v16 = **(_QWORD **)(*((_QWORD *)sub_250CBE0(v48.m128i_i64, v10) + 3) + 16LL);
    else
      v16 = *(_QWORD *)(sub_250D070(&v48) + 8);
    v17 = *(unsigned __int8 *)(v16 + 8);
    if ( (unsigned int)(v17 - 17) <= 1 )
      LOBYTE(v17) = *(_BYTE *)(**(_QWORD **)(v16 + 16) + 8LL);
    if ( (_BYTE)v17 != 14 )
      return;
    v18 = *(_QWORD *)(a1 + 4376);
    if ( v18 )
    {
      v19 = *(_QWORD *)(v18 + 8);
      v20 = *(_DWORD *)(v18 + 24);
      if ( !v20 )
        return;
      v21 = v20 - 1;
      v22 = 1;
      v23 = (v20 - 1) & (((unsigned int)&unk_438A65C >> 9) ^ ((unsigned int)&unk_438A65C >> 4));
      v10 = *(_QWORD *)(v19 + 8LL * v23);
      if ( (_UNKNOWN *)v10 != &unk_438A65C )
      {
        while ( v10 != -4096 )
        {
          v23 = v21 & (v22 + v23);
          v10 = *(_QWORD *)(v19 + 8LL * v23);
          if ( (_UNKNOWN *)v10 == &unk_438A65C )
            goto LABEL_23;
          ++v22;
        }
        return;
      }
    }
LABEL_23:
    v24 = sub_25096F0(&v48);
    if ( v24 )
    {
      v44 = v24;
      if ( (unsigned __int8)sub_B2D610(v24, 20) )
        return;
      v10 = 48;
      if ( (unsigned __int8)sub_B2D610(v44, 48) )
        return;
    }
    if ( *(_DWORD *)(a1 + 3556) > dword_4FEEF68[0] )
      return;
    if ( (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) > 1 )
    {
      v45 = sub_250CBE0(v48.m128i_i64, v10);
      v25 = sub_2509800(&v48);
      if ( v25 > 7u || ((1LL << v25) & 0xA8) == 0 )
        goto LABEL_33;
      v26 = v48.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
      if ( (v48.m128i_i8[0] & 3) == 3 )
        v26 = *(_QWORD *)(v26 + 24);
      if ( **(_BYTE **)(v26 - 32) != 25 )
      {
LABEL_33:
        if ( (v25 & 0xFD) == 4 )
        {
          if ( (v45[32] & 0xFu) - 7 <= 1 && (unsigned __int8)sub_250CC70(a1, v48.m128i_i64) )
          {
LABEL_36:
            if ( *(_BYTE *)(a1 + 4296) || (unsigned __int8)sub_266EE70(*(_QWORD *)(a1 + 200), (__int64)v45) )
              goto LABEL_38;
            v42 = sub_25096F0(&v48);
            v46 = sub_266EE70(*(_QWORD *)(a1 + 200), v42);
LABEL_39:
            v27 = sub_2564B80(&v48, a1);
            v49 = &unk_438A65C;
            v50 = _mm_loadu_si128((const __m128i *)(v27 + 72));
            *sub_2519B70(a1 + 136, (__int64)&v49) = v27;
            if ( *(_DWORD *)(a1 + 3552) <= 1u )
            {
              v49 = (void *)(v27 & 0xFFFFFFFFFFFFFFFBLL);
              sub_269CF50(a1 + 224, (unsigned __int64 *)&v49, v28, v29, v30, v31);
              if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v27) )
                goto LABEL_65;
            }
            v49 = (void *)v27;
            v32 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250BE40, (__int64)&v49);
            v33 = *(_DWORD *)(a1 + 3556);
            *(_DWORD *)(a1 + 3556) = v33 + 1;
            v34 = *(void (**)())(*(_QWORD *)v27 + 24LL);
            if ( v34 != nullsub_1516 )
            {
              ((void (__fastcall *)(__int64, __int64))v34)(v27, a1);
              v33 = *(_DWORD *)(a1 + 3556) - 1;
            }
            *(_DWORD *)(a1 + 3556) = v33;
            if ( v32 )
              sub_C9AF60(v32);
            if ( v46 )
            {
              if ( a7 )
              {
                v41 = *(_DWORD *)(a1 + 3552);
                *(_DWORD *)(a1 + 3552) = 1;
                sub_251C580(a1, v27);
                *(_DWORD *)(a1 + 3552) = v41;
              }
              if ( a4 )
              {
                v35 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v27 + 40LL);
                v36 = (_BYTE *)(v35 == sub_2505F20 ? v27 + 88 : v35(v27));
                v37 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v36 + 16LL);
                if ( v37 == sub_2505E30 ? v36[9] : ((__int64 (*)(void))v37)() )
                  sub_250ED80(a1, v27, a4, a5);
              }
            }
            else
            {
LABEL_65:
              v39 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v27 + 40LL);
              if ( v39 == sub_2505F20 )
                v40 = v27 + 88;
              else
                v40 = v39(v27);
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v40 + 40LL))(v40);
            }
            return;
          }
        }
        else if ( (unsigned __int8)sub_250CC70(a1, v48.m128i_i64) )
        {
          if ( !v45 )
          {
LABEL_38:
            v46 = 1;
            goto LABEL_39;
          }
          goto LABEL_36;
        }
      }
    }
    v46 = 0;
    goto LABEL_39;
  }
  if ( a5 != 2
    && a4
    && ((v12 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 40LL), v12 != sub_2505F20)
      ? (v13 = (_BYTE *)v12(v10))
      : (v13 = (_BYTE *)(v10 + 88)),
        (v14 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v13 + 16LL), v14 != sub_2505E30)
      ? (v15 = ((__int64 (*)(void))v14)())
      : (v15 = v13[9]),
        v15) )
  {
    sub_250ED80(a1, v10, a4, a5);
    if ( !a6 )
      return;
  }
  else if ( !a6 )
  {
    return;
  }
  if ( *(_DWORD *)(a1 + 3552) == 1 )
    sub_251C580(a1, v10);
}
