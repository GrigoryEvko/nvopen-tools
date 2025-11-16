// Function: sub_251FA90
// Address: 0x251fa90
//
__int64 __fastcall sub_251FA90(__int64 a1, const __m128i *a2, __int64 a3, char a4)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 result; // rax
  int v9; // ecx
  int v10; // r8d
  void *v11; // rdi
  __m128i v12; // xmm1
  __int64 v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rax
  unsigned __int8 v16; // cl
  unsigned __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // r13
  int v25; // eax
  void (*v26)(); // rdx
  int v27; // r13d
  __int64 (__fastcall *v28)(__int64); // rax
  unsigned __int8 *v29; // rdi
  __int64 (*v30)(void); // rax
  __int64 v31; // [rsp+0h] [rbp-70h]
  unsigned __int8 *v32; // [rsp+0h] [rbp-70h]
  __int64 v33; // [rsp+8h] [rbp-68h] BYREF
  __m128i v34; // [rsp+10h] [rbp-60h] BYREF
  void *v35; // [rsp+20h] [rbp-50h] BYREF
  __m128i v36; // [rsp+28h] [rbp-48h]

  v33 = a3;
  if ( !a4 )
  {
    result = sub_A73170(&v33, 22);
    if ( (_BYTE)result )
      return result;
  }
  v6 = *(_QWORD *)(a1 + 4376);
  if ( !v6 )
    goto LABEL_5;
  v7 = *(_QWORD *)(v6 + 8);
  result = *(unsigned int *)(v6 + 24);
  if ( !(_DWORD)result )
    return result;
  v9 = result - 1;
  v10 = 1;
  result = ((_DWORD)result - 1) & (((unsigned int)&unk_438A672 >> 9) ^ ((unsigned int)&unk_438A672 >> 4));
  v11 = *(void **)(v7 + 8 * result);
  if ( v11 != &unk_438A672 )
  {
    while ( v11 != (void *)-4096LL )
    {
      result = v9 & (unsigned int)(v10 + result);
      v11 = *(void **)(v7 + 8LL * (unsigned int)result);
      if ( v11 == &unk_438A672 )
        goto LABEL_5;
      ++v10;
    }
  }
  else
  {
LABEL_5:
    result = sub_2553BE0(a1, a2, 22, 0);
    if ( !(_BYTE)result )
    {
      v34 = _mm_loadu_si128(a2);
      if ( !(unsigned __int8)sub_250E300(a1, &v34) )
        v34.m128i_i64[1] = 0;
      v12 = _mm_loadu_si128(&v34);
      v13 = (__int64)&v35;
      v35 = &unk_438A672;
      v36 = v12;
      result = (__int64)sub_25134D0(a1 + 136, (__int64 *)&v35);
      if ( !result || !*(_QWORD *)(result + 24) )
      {
        result = sub_250D180(v34.m128i_i64, (__int64)&v35);
        if ( *(_BYTE *)(result + 8) == 14 )
        {
          v14 = *(_QWORD *)(a1 + 4376);
          if ( !v14
            || (v13 = (__int64)&v35, v35 = &unk_438A672, (result = (__int64)sub_2517B80(v14, (__int64 *)&v35)) != 0) )
          {
            v15 = sub_25096F0(&v34);
            if ( !v15
              || (v31 = v15, result = sub_B2D610(v15, 20), !(_BYTE)result)
              && (v13 = 48, result = sub_B2D610(v31, 48), !(_BYTE)result) )
            {
              result = dword_4FEEF68[0];
              if ( *(_DWORD *)(a1 + 3556) <= dword_4FEEF68[0] )
              {
                result = (unsigned int)(*(_DWORD *)(a1 + 3552) - 2);
                if ( (unsigned int)result > 1 )
                {
                  v32 = sub_250CBE0(v34.m128i_i64, v13);
                  v16 = sub_2509800(&v34);
                  if ( v16 > 7u || ((1LL << v16) & 0xA8) == 0 )
                    goto LABEL_24;
                  v17 = v34.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL;
                  if ( (v34.m128i_i8[0] & 3) == 3 )
                    v17 = *(_QWORD *)(v17 + 24);
                  result = *(_QWORD *)(v17 - 32);
                  if ( *(_BYTE *)result != 25 )
                  {
LABEL_24:
                    if ( (v16 & 0xFD) == 4 )
                    {
                      result = (v32[32] & 0xFu) - 7;
                      if ( (unsigned int)result > 1 )
                        return result;
                      result = sub_250CC70(a1, v34.m128i_i64);
                      if ( !(_BYTE)result )
                        return result;
                    }
                    else
                    {
                      result = sub_250CC70(a1, v34.m128i_i64);
                      if ( !(_BYTE)result )
                        return result;
                      if ( !v32 )
                      {
LABEL_29:
                        v18 = sub_25628D0(&v34, a1);
                        v35 = &unk_438A672;
                        v19 = v18;
                        v36 = _mm_loadu_si128((const __m128i *)(v18 + 72));
                        *sub_2519B70(a1 + 136, (__int64)&v35) = v18;
                        if ( *(_DWORD *)(a1 + 3552) > 1u
                          || (v35 = (void *)(v19 & 0xFFFFFFFFFFFFFFFBLL),
                              sub_251B630(a1 + 224, (unsigned __int64 *)&v35, v20, v21, v22, v23),
                              *(_DWORD *)(a1 + 3552))
                          || (unsigned __int8)sub_250E880(a1, v19) )
                        {
                          v35 = (void *)v19;
                          v24 = sub_C99770(
                                  "initialize",
                                  10,
                                  (void (__fastcall *)(__m128i **, __int64))sub_250B5D0,
                                  (__int64)&v35);
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
                          v27 = *(_DWORD *)(a1 + 3552);
                          *(_DWORD *)(a1 + 3552) = 1;
                          result = sub_251C580(a1, v19);
                          *(_DWORD *)(a1 + 3552) = v27;
                        }
                        else
                        {
                          v28 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v19 + 40LL);
                          if ( v28 == sub_2505F20 )
                            v29 = (unsigned __int8 *)(v19 + 88);
                          else
                            v29 = (unsigned __int8 *)v28(v19);
                          v30 = *(__int64 (**)(void))(*(_QWORD *)v29 + 40LL);
                          if ( (char *)v30 == (char *)sub_2505E20 )
                          {
                            result = v29[8];
                            v29[9] = result;
                          }
                          else
                          {
                            return v30();
                          }
                        }
                        return result;
                      }
                    }
                    if ( !*(_BYTE *)(a1 + 4296) && !(unsigned __int8)sub_2506F10(*(_QWORD *)(a1 + 200), (__int64)v32) )
                    {
                      result = sub_2508DC0(a1, &v34);
                      if ( !(_BYTE)result )
                        return result;
                    }
                    goto LABEL_29;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}
