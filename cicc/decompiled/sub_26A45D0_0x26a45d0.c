// Function: sub_26A45D0
// Address: 0x26a45d0
//
__int64 __fastcall sub_26A45D0(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // r10d
  int v5; // r10d
  int v6; // r13d
  __int64 result; // rax
  __int64 v8; // r11
  unsigned int v9; // eax
  __int64 v10; // rax
  __int64 v11; // rsi
  int v12; // ecx
  int v13; // r8d
  void *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r13
  char v17; // r13
  char v18; // al
  __int64 v19; // rax
  __int64 v20; // r12
  __m128i v21; // xmm1
  __m128i *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 v28; // r14
  __int64 (__fastcall *v29)(__int64); // rax
  __int64 v30; // rdi
  __m128i v31; // [rsp+0h] [rbp-50h] BYREF
  void *v32; // [rsp+10h] [rbp-40h] BYREF
  __m128i v33; // [rsp+18h] [rbp-38h]

  v31.m128i_i64[0] = a2;
  v31.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v31) )
    v31.m128i_i64[1] = 0;
  v4 = *(_DWORD *)(a1 + 160);
  if ( !v4 )
    goto LABEL_10;
  v5 = v4 - 1;
  v6 = 1;
  for ( result = v5
               & ((unsigned int)((0xBF58476D1CE4E5B9LL
                                * (((unsigned __int64)(((unsigned int)&unk_438FC85 >> 9)
                                                     ^ ((unsigned int)&unk_438FC85 >> 4)) << 32)
                                 | ((unsigned __int32)v31.m128i_i32[2] >> 9)
                                 ^ ((unsigned __int32)v31.m128i_i32[2] >> 4)
                                 ^ (16
                                  * (((unsigned __int32)v31.m128i_i32[0] >> 9)
                                   ^ ((unsigned __int32)v31.m128i_i32[0] >> 4))))) >> 31)
                ^ (484763065
                 * (((unsigned __int32)v31.m128i_i32[2] >> 9)
                  ^ ((unsigned __int32)v31.m128i_i32[2] >> 4)
                  ^ (16 * (((unsigned __int32)v31.m128i_i32[0] >> 9) ^ ((unsigned __int32)v31.m128i_i32[0] >> 4))))));
        ;
        result = v5 & v9 )
  {
    v8 = *(_QWORD *)(a1 + 144) + 32LL * (unsigned int)result;
    if ( *(_UNKNOWN **)v8 == &unk_438FC85 && *(_OWORD *)(v8 + 8) == *(_OWORD *)&v31 )
      break;
    if ( *(_QWORD *)v8 == -4096 && *(_QWORD *)(v8 + 8) == qword_4FEE4D0 && *(_QWORD *)(v8 + 16) == qword_4FEE4D8 )
      goto LABEL_10;
    v9 = v6 + result;
    ++v6;
  }
  if ( !*(_QWORD *)(v8 + 24) )
  {
LABEL_10:
    v10 = *(_QWORD *)(a1 + 4376);
    if ( !v10 )
      goto LABEL_13;
    v11 = *(_QWORD *)(v10 + 8);
    result = *(unsigned int *)(v10 + 24);
    if ( !(_DWORD)result )
      return result;
    v12 = result - 1;
    v13 = 1;
    result = ((_DWORD)result - 1) & (((unsigned int)&unk_438FC85 >> 9) ^ ((unsigned int)&unk_438FC85 >> 4));
    v14 = *(void **)(v11 + 8 * result);
    if ( v14 == &unk_438FC85 )
    {
LABEL_13:
      v15 = sub_25096F0(&v31);
      v16 = v15;
      if ( v15 )
      {
        result = sub_B2D610(v15, 20);
        if ( (_BYTE)result )
          return result;
        result = sub_B2D610(v16, 48);
        if ( (_BYTE)result )
          return result;
      }
      result = dword_4FEEF68[0];
      if ( *(_DWORD *)(a1 + 3556) > dword_4FEEF68[0] )
        return result;
      v17 = sub_2673B80(a1, v31.m128i_i64);
      v18 = sub_2509800(&v31);
      if ( v18 == 3 )
      {
        v19 = sub_A777F0(0x80u, *(__int64 **)(a1 + 128));
        v20 = v19;
        if ( v19 )
        {
          v21 = _mm_loadu_si128(&v31);
          v22 = (__m128i *)(v19 + 56);
          v22[-3].m128i_i64[0] = 0;
          v22[-3].m128i_i64[1] = 0;
          v22[1] = v21;
          v22[-2].m128i_i64[0] = 0;
          v22[-2].m128i_i32[2] = 0;
          *(_QWORD *)(v20 + 40) = v22;
          *(_QWORD *)(v20 + 48) = 0x200000000LL;
          *(_QWORD *)v20 = off_4A20558;
          *(_QWORD *)(v20 + 88) = &unk_4A205E0;
          *(_WORD *)(v20 + 96) = 256;
          *(_BYTE *)(v20 + 112) = 0;
          v32 = &unk_438FC85;
          v33 = _mm_loadu_si128((const __m128i *)(v20 + 72));
          *sub_2519B70(a1 + 136, (__int64)&v32) = v20;
          if ( *(_DWORD *)(a1 + 3552) <= 1u )
          {
            v32 = (void *)(v20 & 0xFFFFFFFFFFFFFFFBLL);
            sub_269CF50(a1 + 224, (unsigned __int64 *)&v32, v23, v24, v25, v26);
            if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v20) )
              goto LABEL_23;
          }
          v32 = (void *)v20;
          v27 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_26751B0, (__int64)&v32);
          ++*(_DWORD *)(a1 + 3556);
          v28 = v27;
          result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v20 + 24LL))(v20, a1);
          --*(_DWORD *)(a1 + 3556);
          if ( v28 )
            result = (__int64)sub_C9AF60(v28);
          if ( !v17 )
          {
LABEL_23:
            v29 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v20 + 40LL);
            if ( v29 == sub_2505F20 )
              v30 = v20 + 88;
            else
              v30 = v29(v20);
            return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v30 + 40LL))(v30);
          }
          return result;
        }
LABEL_42:
        v32 = &unk_438FC85;
        BUG();
      }
      if ( v18 > 3 )
      {
        if ( (unsigned __int8)(v18 - 4) > 3u )
          goto LABEL_42;
      }
      else if ( (unsigned __int8)v18 > 2u )
      {
        goto LABEL_42;
      }
      BUG();
    }
    while ( v14 != (void *)-4096LL )
    {
      result = v12 & (unsigned int)(v13 + result);
      v14 = *(void **)(v11 + 8LL * (unsigned int)result);
      if ( v14 == &unk_438FC85 )
        goto LABEL_13;
      ++v13;
    }
  }
  return result;
}
