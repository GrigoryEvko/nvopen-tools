// Function: sub_2521100
// Address: 0x2521100
//
void __fastcall sub_2521100(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v9; // xmm0
  __int64 v10; // rsi
  _QWORD *v11; // rax
  __int64 (__fastcall *v12)(__int64); // rax
  _BYTE *v13; // rdi
  __int64 (__fastcall *v14)(__int64); // rax
  char v15; // al
  __int64 v16; // rax
  int v17; // ecx
  __int64 v18; // rdi
  __int64 v19; // rax
  unsigned __int8 v20; // cl
  unsigned __int64 v21; // rdx
  __int64 v22; // r14
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r15
  int v28; // eax
  void (*v29)(); // rdx
  __int64 (__fastcall *v30)(__int64); // rax
  _BYTE *v31; // rdi
  __int64 (__fastcall *v32)(__int64); // rax
  __int64 (__fastcall *v34)(__int64); // rax
  _BYTE *v35; // rdi
  void (*v36)(void); // rax
  int v37; // ebx
  __int64 v39; // [rsp+10h] [rbp-70h]
  unsigned __int8 *v40; // [rsp+10h] [rbp-70h]
  char v41; // [rsp+10h] [rbp-70h]
  __m128i v43; // [rsp+20h] [rbp-60h] BYREF
  void *v44; // [rsp+30h] [rbp-50h] BYREF
  __m128i v45; // [rsp+38h] [rbp-48h]

  v43.m128i_i64[0] = a2;
  v43.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v43) )
    v43.m128i_i64[1] = 0;
  v9 = _mm_load_si128(&v43);
  v10 = (__int64)&v44;
  v44 = &unk_438A65C;
  v45 = v9;
  v11 = sub_25134D0(a1 + 136, (__int64 *)&v44);
  if ( !v11 || (v10 = v11[3]) == 0 )
  {
    v16 = sub_250D180(v43.m128i_i64, v10);
    v17 = *(unsigned __int8 *)(v16 + 8);
    if ( (unsigned int)(v17 - 17) <= 1 )
      LOBYTE(v17) = *(_BYTE *)(**(_QWORD **)(v16 + 16) + 8LL);
    if ( (_BYTE)v17 != 14 )
      return;
    v18 = *(_QWORD *)(a1 + 4376);
    if ( v18 )
    {
      v10 = (__int64)&v44;
      v44 = &unk_438A65C;
      if ( !sub_2517B80(v18, (__int64 *)&v44) )
        return;
    }
    v19 = sub_25096F0(&v43);
    if ( v19 )
    {
      v39 = v19;
      if ( (unsigned __int8)sub_B2D610(v19, 20) )
        return;
      v10 = 48;
      if ( (unsigned __int8)sub_B2D610(v39, 48) )
        return;
    }
    if ( *(_DWORD *)(a1 + 3556) > dword_4FEEF68[0] )
      return;
    if ( (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) > 1 )
    {
      v40 = sub_250CBE0(v43.m128i_i64, v10);
      v20 = sub_2509800(&v43);
      if ( v20 > 7u || ((1LL << v20) & 0xA8) == 0 )
        goto LABEL_30;
      v21 = v43.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
      if ( (v43.m128i_i8[0] & 3) == 3 )
        v21 = *(_QWORD *)(v21 + 24);
      if ( **(_BYTE **)(v21 - 32) != 25 )
      {
LABEL_30:
        if ( (v20 & 0xFD) == 4 )
        {
          if ( (v40[32] & 0xFu) - 7 <= 1 && (unsigned __int8)sub_250CC70(a1, v43.m128i_i64) )
          {
LABEL_33:
            if ( *(_BYTE *)(a1 + 4296) || (unsigned __int8)sub_2506F10(*(_QWORD *)(a1 + 200), (__int64)v40) )
              goto LABEL_35;
            v41 = sub_2508DC0(a1, &v43);
LABEL_36:
            v22 = sub_2564B80(&v43, a1);
            v44 = &unk_438A65C;
            v45 = _mm_loadu_si128((const __m128i *)(v22 + 72));
            *sub_2519B70(a1 + 136, (__int64)&v44) = v22;
            if ( *(_DWORD *)(a1 + 3552) <= 1u )
            {
              v44 = (void *)(v22 & 0xFFFFFFFFFFFFFFFBLL);
              sub_251B630(a1 + 224, (unsigned __int64 *)&v44, v23, v24, v25, v26);
              if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v22) )
                goto LABEL_58;
            }
            v44 = (void *)v22;
            v27 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250BE40, (__int64)&v44);
            v28 = *(_DWORD *)(a1 + 3556);
            *(_DWORD *)(a1 + 3556) = v28 + 1;
            v29 = *(void (**)())(*(_QWORD *)v22 + 24LL);
            if ( v29 != nullsub_1516 )
            {
              ((void (__fastcall *)(__int64, __int64))v29)(v22, a1);
              v28 = *(_DWORD *)(a1 + 3556) - 1;
            }
            *(_DWORD *)(a1 + 3556) = v28;
            if ( v27 )
              sub_C9AF60(v27);
            if ( v41 )
            {
              if ( a7 )
              {
                v37 = *(_DWORD *)(a1 + 3552);
                *(_DWORD *)(a1 + 3552) = 1;
                sub_251C580(a1, v22);
                *(_DWORD *)(a1 + 3552) = v37;
              }
              if ( a4 )
              {
                v30 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v22 + 40LL);
                v31 = (_BYTE *)(v30 == sub_2505F20 ? v22 + 88 : v30(v22));
                v32 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v31 + 16LL);
                if ( v32 == sub_2505E30 ? v31[9] : ((__int64 (*)(void))v32)() )
                  sub_250ED80(a1, v22, a4, a5);
              }
            }
            else
            {
LABEL_58:
              v34 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v22 + 40LL);
              if ( v34 == sub_2505F20 )
                v35 = (_BYTE *)(v22 + 88);
              else
                v35 = (_BYTE *)v34(v22);
              v36 = *(void (**)(void))(*(_QWORD *)v35 + 40LL);
              if ( (char *)v36 == (char *)sub_2505E20 )
                v35[9] = v35[8];
              else
                v36();
            }
            return;
          }
        }
        else if ( (unsigned __int8)sub_250CC70(a1, v43.m128i_i64) )
        {
          if ( !v40 )
          {
LABEL_35:
            v41 = 1;
            goto LABEL_36;
          }
          goto LABEL_33;
        }
      }
    }
    v41 = 0;
    goto LABEL_36;
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
