// Function: sub_251F740
// Address: 0x251f740
//
__int64 __fastcall sub_251F740(__int64 a1, const __m128i *a2, __int64 a3, char a4)
{
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 result; // rax
  int v9; // edx
  int v10; // edi
  void *v11; // rsi
  __m128i v12; // xmm1
  int v13; // ecx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r12
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r13
  int v23; // eax
  void (*v24)(); // rdx
  int v25; // r13d
  __int64 (__fastcall *v26)(__int64); // rax
  unsigned __int8 *v27; // rdi
  __int64 (*v28)(void); // rax
  __int64 v29; // [rsp+0h] [rbp-80h]
  __int64 v30; // [rsp+8h] [rbp-78h] BYREF
  char v31; // [rsp+1Fh] [rbp-61h] BYREF
  __m128i v32; // [rsp+20h] [rbp-60h] BYREF
  void *v33; // [rsp+30h] [rbp-50h] BYREF
  __m128i v34; // [rsp+38h] [rbp-48h]

  v30 = a3;
  if ( a4 || (result = sub_A73170(&v30, 43), !(_BYTE)result) )
  {
    v6 = *(_QWORD *)(a1 + 4376);
    if ( !v6 )
      goto LABEL_5;
    v7 = *(_QWORD *)(v6 + 8);
    result = *(unsigned int *)(v6 + 24);
    if ( !(_DWORD)result )
      return result;
    v9 = result - 1;
    v10 = 1;
    result = ((_DWORD)result - 1) & (((unsigned int)&unk_438A678 >> 9) ^ ((unsigned int)&unk_438A678 >> 4));
    v11 = *(void **)(v7 + 8 * result);
    if ( v11 != &unk_438A678 )
    {
      while ( v11 != (void *)-4096LL )
      {
        result = v9 & (unsigned int)(v10 + result);
        v11 = *(void **)(v7 + 8LL * (unsigned int)result);
        if ( v11 == &unk_438A678 )
          goto LABEL_5;
        ++v10;
      }
    }
    else
    {
LABEL_5:
      result = sub_255E680(a1, a2, 43, 0);
      if ( !(_BYTE)result )
      {
        v32 = _mm_loadu_si128(a2);
        if ( !(unsigned __int8)sub_250E300(a1, &v32) )
          v32.m128i_i64[1] = 0;
        v12 = _mm_loadu_si128(&v32);
        v33 = &unk_438A678;
        v34 = v12;
        result = (__int64)sub_25134D0(a1 + 136, (__int64 *)&v33);
        if ( !result || !*(_QWORD *)(result + 24) )
        {
          result = sub_250D180(v32.m128i_i64, (__int64)&v33);
          v13 = *(unsigned __int8 *)(result + 8);
          if ( (unsigned int)(v13 - 17) <= 1 )
          {
            result = **(_QWORD **)(result + 16);
            LOBYTE(v13) = *(_BYTE *)(result + 8);
          }
          if ( (_BYTE)v13 == 14 )
          {
            v14 = *(_QWORD *)(a1 + 4376);
            if ( !v14 || (v33 = &unk_438A678, (result = (__int64)sub_2517B80(v14, (__int64 *)&v33)) != 0) )
            {
              v15 = sub_25096F0(&v32);
              if ( !v15
                || (v29 = v15, result = sub_B2D610(v15, 20), !(_BYTE)result)
                && (result = sub_B2D610(v29, 48), !(_BYTE)result) )
              {
                result = sub_250CDD0(a1, v32.m128i_i64, &v31);
                if ( (_BYTE)result )
                {
                  v16 = sub_25626E0(&v32, a1);
                  v33 = &unk_438A678;
                  v17 = v16;
                  v34 = _mm_loadu_si128((const __m128i *)(v16 + 72));
                  *sub_2519B70(a1 + 136, (__int64)&v33) = v16;
                  if ( *(_DWORD *)(a1 + 3552) <= 1u )
                  {
                    v33 = (void *)(v17 & 0xFFFFFFFFFFFFFFFBLL);
                    sub_251B630(a1 + 224, (unsigned __int64 *)&v33, v18, v19, v20, v21);
                    if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v17) )
                      goto LABEL_35;
                  }
                  v33 = (void *)v17;
                  v22 = sub_C99770(
                          "initialize",
                          10,
                          (void (__fastcall *)(__m128i **, __int64))sub_250B4A0,
                          (__int64)&v33);
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
                  if ( v31 )
                  {
                    v25 = *(_DWORD *)(a1 + 3552);
                    *(_DWORD *)(a1 + 3552) = 1;
                    result = sub_251C580(a1, v17);
                    *(_DWORD *)(a1 + 3552) = v25;
                  }
                  else
                  {
LABEL_35:
                    v26 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v17 + 40LL);
                    if ( v26 == sub_2505F20 )
                      v27 = (unsigned __int8 *)(v17 + 88);
                    else
                      v27 = (unsigned __int8 *)v26(v17);
                    v28 = *(__int64 (**)(void))(*(_QWORD *)v27 + 40LL);
                    if ( (char *)v28 == (char *)sub_2505E20 )
                    {
                      result = v27[8];
                      v27[9] = result;
                    }
                    else
                    {
                      return v28();
                    }
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
