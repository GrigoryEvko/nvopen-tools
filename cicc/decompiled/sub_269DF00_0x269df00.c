// Function: sub_269DF00
// Address: 0x269df00
//
__int64 __fastcall sub_269DF00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  __m128i v8; // xmm0
  __int64 v9; // rsi
  _QWORD *v10; // rax
  __int64 v11; // rbx
  __int64 (__fastcall *v12)(__int64); // rax
  _BYTE *v13; // rdi
  __int64 (__fastcall *v14)(__int64); // rax
  char v15; // al
  __int64 v17; // rax
  __int64 v18; // rcx
  int v19; // eax
  int v20; // edx
  int v21; // edi
  unsigned int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rbx
  unsigned __int8 *v25; // rbx
  unsigned __int8 v26; // cl
  unsigned __int64 v27; // rdx
  __int64 v28; // r10
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // r13
  int v34; // eax
  void (*v35)(); // rdx
  __int64 (__fastcall *v36)(__int64); // rax
  _BYTE *v37; // rdi
  __int64 (__fastcall *v38)(__int64); // rax
  __int64 (__fastcall *v40)(__int64); // rax
  __int64 v41; // rdi
  int v42; // r13d
  unsigned __int64 v43; // rdi
  unsigned __int8 v44; // al
  __int64 v45; // rax
  int v46; // edx
  __int64 v47; // rsi
  int v48; // edx
  unsigned int v49; // eax
  __int64 v50; // rcx
  int v51; // r8d
  char v54; // [rsp+18h] [rbp-68h]
  __m128i v56; // [rsp+20h] [rbp-60h] BYREF
  void *v57; // [rsp+30h] [rbp-50h] BYREF
  __m128i v58; // [rsp+38h] [rbp-48h]

  v56.m128i_i64[0] = a2;
  v56.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v56) )
    v56.m128i_i64[1] = 0;
  v8 = _mm_load_si128(&v56);
  v9 = (__int64)&v57;
  v57 = &unk_438A66A;
  v58 = v8;
  v10 = sub_25134D0(a1 + 136, (__int64 *)&v57);
  if ( !v10 || (v11 = v10[3]) == 0 )
  {
    v17 = *(_QWORD *)(a1 + 4376);
    if ( v17 )
    {
      v18 = *(_QWORD *)(v17 + 8);
      v19 = *(_DWORD *)(v17 + 24);
      if ( !v19 )
        return 0;
      v20 = v19 - 1;
      v21 = 1;
      v22 = (v19 - 1) & (((unsigned int)&unk_438A66A >> 9) ^ ((unsigned int)&unk_438A66A >> 4));
      v9 = *(_QWORD *)(v18 + 8LL * v22);
      if ( (_UNKNOWN *)v9 != &unk_438A66A )
      {
        while ( v9 != -4096 )
        {
          v22 = v20 & (v21 + v22);
          v9 = *(_QWORD *)(v18 + 8LL * v22);
          if ( (_UNKNOWN *)v9 == &unk_438A66A )
            goto LABEL_19;
          ++v21;
        }
        return 0;
      }
    }
LABEL_19:
    v23 = sub_25096F0(&v56);
    v24 = v23;
    if ( !v23 || !(unsigned __int8)sub_B2D610(v23, 20) && (v9 = 48, !(unsigned __int8)sub_B2D610(v24, 48)) )
    {
      if ( *(_DWORD *)(a1 + 3556) <= dword_4FEEF68[0] )
      {
        if ( (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) > 1 )
        {
          v25 = sub_250CBE0(v56.m128i_i64, v9);
          v26 = sub_2509800(&v56);
          if ( v26 > 7u || ((1LL << v26) & 0xA8) == 0 )
            goto LABEL_29;
          v27 = v56.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
          if ( (v56.m128i_i8[0] & 3) == 3 )
            v27 = *(_QWORD *)(v27 + 24);
          if ( **(_BYTE **)(v27 - 32) != 25 )
          {
LABEL_29:
            v54 = sub_250CC70(a1, v56.m128i_i64);
            if ( v54 )
            {
              if ( !v25 || *(_BYTE *)(a1 + 4296) || (unsigned __int8)sub_266EE70(*(_QWORD *)(a1 + 200), (__int64)v25) )
                goto LABEL_33;
              v43 = v56.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
              if ( (v56.m128i_i8[0] & 3) == 3 )
                v43 = *(_QWORD *)(v43 + 24);
              v44 = *(_BYTE *)v43;
              if ( *(_BYTE *)v43 )
              {
                if ( v44 == 22 )
                {
                  v43 = *(_QWORD *)(v43 + 24);
                }
                else if ( v44 <= 0x1Cu )
                {
                  v43 = 0;
                }
                else
                {
                  v45 = sub_B43CB0(v43);
                  v28 = *(_QWORD *)(a1 + 200);
                  v43 = v45;
                }
              }
              if ( !*(_DWORD *)(v28 + 40) )
                goto LABEL_33;
              v46 = *(_DWORD *)(v28 + 24);
              v47 = *(_QWORD *)(v28 + 8);
              if ( v46 )
              {
                v48 = v46 - 1;
                v49 = v48 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
                v50 = *(_QWORD *)(v47 + 8LL * v49);
                if ( v50 == v43 )
                {
LABEL_33:
                  v11 = sub_2566180(&v56, a1);
                  v57 = &unk_438A66A;
                  v58 = _mm_loadu_si128((const __m128i *)(v11 + 72));
                  *sub_2519B70(a1 + 136, (__int64)&v57) = v11;
                  if ( *(_DWORD *)(a1 + 3552) <= 1u )
                  {
                    v57 = (void *)(v11 & 0xFFFFFFFFFFFFFFFBLL);
                    sub_269CF50(a1 + 224, (unsigned __int64 *)&v57, v29, v30, v31, v32);
                    if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v11) )
                      goto LABEL_59;
                  }
                  v57 = (void *)v11;
                  v33 = sub_C99770(
                          "initialize",
                          10,
                          (void (__fastcall *)(__m128i **, __int64))sub_250A890,
                          (__int64)&v57);
                  v34 = *(_DWORD *)(a1 + 3556);
                  *(_DWORD *)(a1 + 3556) = v34 + 1;
                  v35 = *(void (**)())(*(_QWORD *)v11 + 24LL);
                  if ( v35 != nullsub_1516 )
                  {
                    ((void (__fastcall *)(__int64, __int64))v35)(v11, a1);
                    v34 = *(_DWORD *)(a1 + 3556) - 1;
                  }
                  *(_DWORD *)(a1 + 3556) = v34;
                  if ( v33 )
                    sub_C9AF60(v33);
                  if ( v54 )
                  {
                    if ( a7 )
                    {
                      v42 = *(_DWORD *)(a1 + 3552);
                      *(_DWORD *)(a1 + 3552) = 1;
                      sub_251C580(a1, v11);
                      *(_DWORD *)(a1 + 3552) = v42;
                    }
                    if ( a4 )
                    {
                      v36 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
                      v37 = (_BYTE *)(v36 == sub_2505F20 ? v11 + 88 : v36(v11));
                      v38 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v37 + 16LL);
                      if ( v38 == sub_2505E30 ? v37[9] : ((__int64 (*)(void))v38)() )
                        sub_250ED80(a1, v11, a4, a5);
                    }
                  }
                  else
                  {
LABEL_59:
                    v40 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
                    if ( v40 == sub_2505F20 )
                      v41 = v11 + 88;
                    else
                      v41 = v40(v11);
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v41 + 40LL))(v41);
                  }
                  return v11;
                }
                v51 = 1;
                while ( v50 != -4096 )
                {
                  v49 = v48 & (v51 + v49);
                  v50 = *(_QWORD *)(v47 + 8LL * v49);
                  if ( v43 == v50 )
                    goto LABEL_33;
                  ++v51;
                }
              }
            }
          }
        }
        v54 = 0;
        goto LABEL_33;
      }
    }
    return 0;
  }
  if ( a5 != 2
    && a4
    && ((v12 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL), v12 != sub_2505F20)
      ? (v13 = (_BYTE *)v12(v11))
      : (v13 = (_BYTE *)(v11 + 88)),
        (v14 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v13 + 16LL), v14 != sub_2505E30)
      ? (v15 = ((__int64 (*)(void))v14)())
      : (v15 = v13[9]),
        v15) )
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
