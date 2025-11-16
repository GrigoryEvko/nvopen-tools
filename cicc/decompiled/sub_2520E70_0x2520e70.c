// Function: sub_2520E70
// Address: 0x2520e70
//
__int64 __fastcall sub_2520E70(__int64 *a1, unsigned __int64 a2)
{
  __int64 v2; // rbx
  __m128i v3; // rax
  __m128i v4; // xmm0
  _QWORD *v5; // rax
  __int64 v7; // rax
  int v8; // ecx
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r13
  int v18; // eax
  void (*v19)(); // rdx
  int v20; // r13d
  __int64 (__fastcall *v21)(__int64); // rax
  _BYTE *v22; // rdi
  void (*v23)(void); // rax
  __int64 v24; // [rsp+8h] [rbp-78h]
  char v25; // [rsp+1Fh] [rbp-61h] BYREF
  __m128i v26; // [rsp+20h] [rbp-60h] BYREF
  void *v27; // [rsp+30h] [rbp-50h] BYREF
  __m128i v28; // [rsp+38h] [rbp-48h]

  v2 = *a1;
  v3.m128i_i64[0] = sub_250D2C0(a2, 0);
  v26 = v3;
  if ( !(unsigned __int8)sub_250E300(v2, &v26) )
    v26.m128i_i64[1] = 0;
  v4 = _mm_loadu_si128(&v26);
  v27 = &unk_438A65B;
  v28 = v4;
  v5 = sub_25134D0(v2 + 136, (__int64 *)&v27);
  if ( !v5 || !v5[3] )
  {
    v7 = sub_250D180(v26.m128i_i64, (__int64)&v27);
    v8 = *(unsigned __int8 *)(v7 + 8);
    if ( (unsigned int)(v8 - 17) <= 1 )
      LOBYTE(v8) = *(_BYTE *)(**(_QWORD **)(v7 + 16) + 8LL);
    if ( (_BYTE)v8 == 14 )
    {
      v9 = *(_QWORD *)(v2 + 4376);
      if ( !v9 || (v27 = &unk_438A65B, sub_2517B80(v9, (__int64 *)&v27)) )
      {
        v10 = sub_25096F0(&v26);
        if ( !v10 || (v24 = v10, !(unsigned __int8)sub_B2D610(v10, 20)) && !(unsigned __int8)sub_B2D610(v24, 48) )
        {
          if ( (unsigned __int8)sub_250CDD0(v2, v26.m128i_i64, &v25) )
          {
            v11 = sub_2564D90(&v26, v2);
            v27 = &unk_438A65B;
            v12 = v11;
            v28 = _mm_loadu_si128((const __m128i *)(v11 + 72));
            *sub_2519B70(v2 + 136, (__int64)&v27) = v11;
            if ( *(_DWORD *)(v2 + 3552) <= 1u )
            {
              v27 = (void *)(v12 & 0xFFFFFFFFFFFFFFFBLL);
              sub_251B630(v2 + 224, (unsigned __int64 *)&v27, v13, v14, v15, v16);
              if ( !*(_DWORD *)(v2 + 3552) && !(unsigned __int8)sub_250E880(v2, v12) )
                goto LABEL_24;
            }
            v27 = (void *)v12;
            v17 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_250BF70, (__int64)&v27);
            v18 = *(_DWORD *)(v2 + 3556);
            *(_DWORD *)(v2 + 3556) = v18 + 1;
            v19 = *(void (**)())(*(_QWORD *)v12 + 24LL);
            if ( v19 != nullsub_1516 )
            {
              ((void (__fastcall *)(__int64, __int64))v19)(v12, v2);
              v18 = *(_DWORD *)(v2 + 3556) - 1;
            }
            *(_DWORD *)(v2 + 3556) = v18;
            if ( v17 )
              sub_C9AF60(v17);
            if ( v25 )
            {
              v20 = *(_DWORD *)(v2 + 3552);
              *(_DWORD *)(v2 + 3552) = 1;
              sub_251C580(v2, v12);
              *(_DWORD *)(v2 + 3552) = v20;
            }
            else
            {
LABEL_24:
              v21 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 40LL);
              if ( v21 == sub_2505F20 )
                v22 = (_BYTE *)(v12 + 88);
              else
                v22 = (_BYTE *)v21(v12);
              v23 = *(void (**)(void))(*(_QWORD *)v22 + 40LL);
              if ( (char *)v23 == (char *)sub_2505E20 )
                v22[9] = v22[8];
              else
                v23();
            }
          }
        }
      }
    }
  }
  return 1;
}
