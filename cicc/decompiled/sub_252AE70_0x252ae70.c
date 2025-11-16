// Function: sub_252AE70
// Address: 0x252ae70
//
__int64 __fastcall sub_252AE70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v8; // xmm0
  __int64 v9; // rsi
  _QWORD *v10; // rax
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v14; // rax
  int v15; // ecx
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r13
  unsigned __int8 *v19; // r13
  unsigned __int8 v20; // cl
  unsigned __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // r14
  int v27; // eax
  void (*v28)(); // rdx
  __int64 v29; // rax
  __int64 v30; // rax
  int v31; // ebx
  char v34; // [rsp+18h] [rbp-68h]
  __m128i v36; // [rsp+20h] [rbp-60h] BYREF
  void *v37; // [rsp+30h] [rbp-50h] BYREF
  __m128i v38; // [rsp+38h] [rbp-48h]

  v36.m128i_i64[0] = a2;
  v36.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v36) )
    v36.m128i_i64[1] = 0;
  v8 = _mm_load_si128(&v36);
  v9 = (__int64)&v37;
  v37 = &unk_438A65D;
  v38 = v8;
  v10 = sub_25134D0(a1 + 136, (__int64 *)&v37);
  if ( !v10 || (v11 = v10[3]) == 0 )
  {
    v14 = sub_250D180(v36.m128i_i64, (__int64)&v37);
    v15 = *(unsigned __int8 *)(v14 + 8);
    if ( (unsigned int)(v15 - 17) <= 1 )
      LOBYTE(v15) = *(_BYTE *)(**(_QWORD **)(v14 + 16) + 8LL);
    if ( (_BYTE)v15 != 14 )
      return 0;
    v16 = *(_QWORD *)(a1 + 4376);
    if ( v16 )
    {
      v9 = (__int64)&v37;
      v37 = &unk_438A65D;
      if ( !sub_2517B80(v16, (__int64 *)&v37) )
        return 0;
    }
    v17 = sub_25096F0(&v36);
    v18 = v17;
    if ( v17 )
    {
      if ( (unsigned __int8)sub_B2D610(v17, 20) )
        return 0;
      v9 = 48;
      if ( (unsigned __int8)sub_B2D610(v18, 48) )
        return 0;
    }
    if ( *(_DWORD *)(a1 + 3556) > dword_4FEEF68[0] )
      return 0;
    if ( (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) > 1 )
    {
      v19 = sub_250CBE0(v36.m128i_i64, v9);
      v20 = sub_2509800(&v36);
      if ( v20 > 7u || ((1LL << v20) & 0xA8) == 0 )
        goto LABEL_30;
      v21 = v36.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
      if ( (v36.m128i_i8[0] & 3) == 3 )
        v21 = *(_QWORD *)(v21 + 24);
      if ( **(_BYTE **)(v21 - 32) != 25 )
      {
LABEL_30:
        if ( (v20 & 0xFD) == 4 )
        {
          if ( (v19[32] & 0xFu) - 7 <= 1 && (unsigned __int8)sub_250CC70(a1, v36.m128i_i64) )
          {
LABEL_33:
            if ( *(_BYTE *)(a1 + 4296) || (unsigned __int8)sub_2506F10(*(_QWORD *)(a1 + 200), (__int64)v19) )
              goto LABEL_35;
            v34 = sub_2508DC0(a1, &v36);
LABEL_36:
            v11 = sub_2565830(&v36, a1);
            v37 = &unk_438A65D;
            v38 = _mm_loadu_si128((const __m128i *)(v11 + 72));
            *sub_2519B70(a1 + 136, (__int64)&v37) = v11;
            if ( *(_DWORD *)(a1 + 3552) <= 1u )
            {
              v37 = (void *)(v11 & 0xFFFFFFFFFFFFFFFBLL);
              sub_251B630(a1 + 224, (unsigned __int64 *)&v37, v22, v23, v24, v25);
              if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v11) )
                goto LABEL_49;
            }
            v37 = (void *)v11;
            v26 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_2509AE0, (__int64)&v37);
            v27 = *(_DWORD *)(a1 + 3556);
            *(_DWORD *)(a1 + 3556) = v27 + 1;
            v28 = *(void (**)())(*(_QWORD *)v11 + 24LL);
            if ( v28 != nullsub_1516 )
            {
              ((void (__fastcall *)(__int64, __int64))v28)(v11, a1);
              v27 = *(_DWORD *)(a1 + 3556) - 1;
            }
            *(_DWORD *)(a1 + 3556) = v27;
            if ( v26 )
              sub_C9AF60(v26);
            if ( v34 )
            {
              if ( a7 )
              {
                v31 = *(_DWORD *)(a1 + 3552);
                *(_DWORD *)(a1 + 3552) = 1;
                sub_251C580(a1, v11);
                *(_DWORD *)(a1 + 3552) = v31;
              }
              if ( a4 )
              {
                v29 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL))(v11);
                if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v29 + 16LL))(v29) )
                  sub_250ED80(a1, v11, a4, a5);
              }
            }
            else
            {
LABEL_49:
              v30 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL))(v11);
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v30 + 40LL))(v30);
            }
            return v11;
          }
        }
        else if ( (unsigned __int8)sub_250CC70(a1, v36.m128i_i64) )
        {
          if ( !v19 )
          {
LABEL_35:
            v34 = 1;
            goto LABEL_36;
          }
          goto LABEL_33;
        }
      }
    }
    v34 = 0;
    goto LABEL_36;
  }
  if ( a5 != 2
    && a4
    && (v12 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL))(v11),
        (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v12 + 16LL))(v12)) )
  {
    sub_250ED80(a1, v11, a4, a5);
    if ( !a6 )
      return v11;
  }
  else if ( !a6 )
  {
    return v11;
  }
  if ( *(_DWORD *)(a1 + 3552) == 1 )
    sub_251C580(a1, v11);
  return v11;
}
