// Function: sub_269E5E0
// Address: 0x269e5e0
//
__int64 __fastcall sub_269E5E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
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
  __int64 v33; // rax
  __int64 v34; // r13
  __int64 (__fastcall *v35)(__int64); // rax
  _BYTE *v36; // rdi
  __int64 (__fastcall *v37)(__int64); // rax
  __int64 (__fastcall *v39)(__int64); // rax
  __int64 v40; // rdi
  int v41; // r13d
  unsigned __int64 v42; // rdi
  unsigned __int8 v43; // al
  __int64 v44; // rax
  int v45; // edx
  __int64 v46; // rsi
  int v47; // edx
  unsigned int v48; // eax
  __int64 v49; // rcx
  int v50; // r8d
  char v53; // [rsp+18h] [rbp-68h]
  __m128i v55; // [rsp+20h] [rbp-60h] BYREF
  void *v56; // [rsp+30h] [rbp-50h] BYREF
  __m128i v57; // [rsp+38h] [rbp-48h]

  v55.m128i_i64[0] = a2;
  v55.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v55) )
    v55.m128i_i64[1] = 0;
  v8 = _mm_load_si128(&v55);
  v9 = (__int64)&v56;
  v56 = &unk_438FC87;
  v57 = v8;
  v10 = sub_25134D0(a1 + 136, (__int64 *)&v56);
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
      v22 = (v19 - 1) & (((unsigned int)&unk_438FC87 >> 9) ^ ((unsigned int)&unk_438FC87 >> 4));
      v9 = *(_QWORD *)(v18 + 8LL * v22);
      if ( (_UNKNOWN *)v9 != &unk_438FC87 )
      {
        while ( v9 != -4096 )
        {
          v22 = v20 & (v21 + v22);
          v9 = *(_QWORD *)(v18 + 8LL * v22);
          if ( (_UNKNOWN *)v9 == &unk_438FC87 )
            goto LABEL_19;
          ++v21;
        }
        return 0;
      }
    }
LABEL_19:
    v23 = sub_25096F0(&v55);
    v24 = v23;
    if ( !v23 || !(unsigned __int8)sub_B2D610(v23, 20) && (v9 = 48, !(unsigned __int8)sub_B2D610(v24, 48)) )
    {
      if ( *(_DWORD *)(a1 + 3556) <= dword_4FEEF68[0] )
      {
        if ( (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) > 1 )
        {
          v25 = sub_250CBE0(v55.m128i_i64, v9);
          v26 = sub_2509800(&v55);
          if ( v26 > 7u || ((1LL << v26) & 0xA8) == 0 )
            goto LABEL_29;
          v27 = v55.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
          if ( (v55.m128i_i8[0] & 3) == 3 )
            v27 = *(_QWORD *)(v27 + 24);
          if ( **(_BYTE **)(v27 - 32) != 25 )
          {
LABEL_29:
            v53 = sub_250CC70(a1, v55.m128i_i64);
            if ( v53 )
            {
              if ( !v25 || *(_BYTE *)(a1 + 4296) || (unsigned __int8)sub_266EE70(*(_QWORD *)(a1 + 200), (__int64)v25) )
                goto LABEL_33;
              v42 = v55.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
              if ( (v55.m128i_i8[0] & 3) == 3 )
                v42 = *(_QWORD *)(v42 + 24);
              v43 = *(_BYTE *)v42;
              if ( *(_BYTE *)v42 )
              {
                if ( v43 == 22 )
                {
                  v42 = *(_QWORD *)(v42 + 24);
                }
                else if ( v43 <= 0x1Cu )
                {
                  v42 = 0;
                }
                else
                {
                  v44 = sub_B43CB0(v42);
                  v28 = *(_QWORD *)(a1 + 200);
                  v42 = v44;
                }
              }
              if ( !*(_DWORD *)(v28 + 40) )
                goto LABEL_33;
              v45 = *(_DWORD *)(v28 + 24);
              v46 = *(_QWORD *)(v28 + 8);
              if ( v45 )
              {
                v47 = v45 - 1;
                v48 = v47 & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
                v49 = *(_QWORD *)(v46 + 8LL * v48);
                if ( v49 == v42 )
                {
LABEL_33:
                  v11 = sub_267DD10(&v55, a1);
                  v56 = &unk_438FC87;
                  v57 = _mm_loadu_si128((const __m128i *)(v11 + 72));
                  *sub_2519B70(a1 + 136, (__int64)&v56) = v11;
                  if ( *(_DWORD *)(a1 + 3552) <= 1u )
                  {
                    v56 = (void *)(v11 & 0xFFFFFFFFFFFFFFFBLL);
                    sub_269CF50(a1 + 224, (unsigned __int64 *)&v56, v29, v30, v31, v32);
                    if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v11) )
                      goto LABEL_57;
                  }
                  v56 = (void *)v11;
                  v33 = sub_C99770(
                          "initialize",
                          10,
                          (void (__fastcall *)(__m128i **, __int64))sub_2675870,
                          (__int64)&v56);
                  ++*(_DWORD *)(a1 + 3556);
                  v34 = v33;
                  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v11 + 24LL))(v11, a1);
                  --*(_DWORD *)(a1 + 3556);
                  if ( v34 )
                    sub_C9AF60(v34);
                  if ( v53 )
                  {
                    if ( a7 )
                    {
                      v41 = *(_DWORD *)(a1 + 3552);
                      *(_DWORD *)(a1 + 3552) = 1;
                      sub_251C580(a1, v11);
                      *(_DWORD *)(a1 + 3552) = v41;
                    }
                    if ( a4 )
                    {
                      v35 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
                      v36 = (_BYTE *)(v35 == sub_2505F20 ? v11 + 88 : v35(v11));
                      v37 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v36 + 16LL);
                      if ( v37 == sub_2505E30 ? v36[9] : ((__int64 (*)(void))v37)() )
                        sub_250ED80(a1, v11, a4, a5);
                    }
                  }
                  else
                  {
LABEL_57:
                    v39 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 40LL);
                    if ( v39 == sub_2505F20 )
                      v40 = v11 + 88;
                    else
                      v40 = v39(v11);
                    (*(void (__fastcall **)(__int64))(*(_QWORD *)v40 + 40LL))(v40);
                  }
                  return v11;
                }
                v50 = 1;
                while ( v49 != -4096 )
                {
                  v48 = v47 & (v50 + v48);
                  v49 = *(_QWORD *)(v46 + 8LL * v48);
                  if ( v42 == v49 )
                    goto LABEL_33;
                  ++v50;
                }
              }
            }
          }
        }
        v53 = 0;
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
