// Function: sub_3421E80
// Address: 0x3421e80
//
void (*__fastcall sub_3421E80(__int64 a1, __int64 a2))()
{
  void (*v3)(void); // rax
  const __m128i *v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rax
  __m128i v7; // xmm0
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rcx
  _QWORD *v12; // rax
  const __m128i *v13; // r14
  _QWORD *v14; // rcx
  unsigned __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rbx
  __int64 v20; // r15
  void (*result)(); // rax
  __m128i v22; // [rsp+0h] [rbp-130h] BYREF
  __int64 v23; // [rsp+10h] [rbp-120h]
  __int64 v24; // [rsp+18h] [rbp-118h]
  __int64 v25; // [rsp+20h] [rbp-110h]
  __int64 v26; // [rsp+28h] [rbp-108h]
  __m128i v27; // [rsp+30h] [rbp-100h]
  _QWORD *v28; // [rsp+48h] [rbp-E8h] BYREF
  __int64 (__fastcall **v29)(); // [rsp+50h] [rbp-E0h] BYREF
  __int64 v30; // [rsp+58h] [rbp-D8h]
  __int64 v31; // [rsp+60h] [rbp-D0h]
  _QWORD *v32; // [rsp+68h] [rbp-C8h]
  _QWORD v33[8]; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v34; // [rsp+B0h] [rbp-80h]
  int v35; // [rsp+B8h] [rbp-78h]
  __int64 v36; // [rsp+C0h] [rbp-70h]
  __int64 v37; // [rsp+C8h] [rbp-68h]
  __int64 v38; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v39; // [rsp+D8h] [rbp-58h]
  _QWORD *v40; // [rsp+E0h] [rbp-50h]
  __int64 v41; // [rsp+E8h] [rbp-48h]
  __int64 v42; // [rsp+F0h] [rbp-40h] BYREF

  v3 = *(void (**)(void))(*(_QWORD *)a1 + 32LL);
  if ( v3 != nullsub_1876 )
    v3();
  *(_DWORD *)(a1 + 920) = sub_33E2990(*(_QWORD *)(a1 + 64));
  v4 = *(const __m128i **)(a1 + 64);
  v5 = v4[24].m128i_i64[0];
  v22 = _mm_loadu_si128(v4 + 24);
  v6 = sub_33ECD10(1u);
  v35 = 0;
  v7 = _mm_load_si128(&v22);
  v33[6] = v6;
  v34 = 0x100000000LL;
  v37 = 0xFFFFFFFFLL;
  v27 = v7;
  v42 = 0;
  v33[7] = 0;
  v36 = 0;
  v41 = 0;
  v40 = v33;
  LODWORD(v39) = v7.m128i_i32[2];
  v38 = v7.m128i_i64[0];
  v8 = *(_QWORD *)(v5 + 56);
  memset(v33, 0, 24);
  v33[3] = 328;
  v33[4] = -65536;
  v42 = v8;
  if ( v8 )
    *(_QWORD *)(v8 + 24) = &v42;
  v41 = v5 + 56;
  *(_QWORD *)(v5 + 56) = &v38;
  v9 = *(_QWORD *)(a1 + 64);
  v33[5] = &v38;
  v10 = *(_QWORD *)(v9 + 384);
  LODWORD(v34) = 1;
  if ( !v10 )
    BUG();
  v11 = *(_QWORD *)(v9 + 768);
  v12 = *(_QWORD **)(v10 + 16);
  v31 = v9;
  v29 = off_4A36AD0;
  v30 = v11;
  *(_QWORD *)(v9 + 768) = &v29;
  v28 = v12;
  v32 = &v28;
LABEL_10:
  v13 = *(const __m128i **)(a1 + 64);
  v14 = (_QWORD *)v13[25].m128i_i64[1];
  while ( v12 != v14 )
  {
    v15 = *v12 & 0xFFFFFFFFFFFFFFF8LL;
    v28 = (_QWORD *)v15;
    v12 = (_QWORD *)v15;
    if ( !v15 )
      BUG();
    if ( *(_QWORD *)(v15 + 48) )
    {
      v16 = *(_QWORD *)(a1 + 808);
      a2 = v15 - 8;
      if ( !*(_BYTE *)(v16 + 537006) )
      {
        v17 = *(unsigned int *)(v15 + 16);
        if ( (int)v17 <= 239 )
        {
          if ( (int)v17 > 237 || (unsigned int)(v17 - 101) <= 0x2F )
          {
            if ( (unsigned int)(v17 - 135) <= 0xD && ((1LL << ((unsigned __int8)v17 + 121)) & 0x330F) != 0 )
              v18 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v15 + 32) + 40LL) + 48LL)
                                        + 16LL * *(unsigned int *)(*(_QWORD *)(v15 + 32) + 48LL));
            else
LABEL_17:
              v18 = **(unsigned __int16 **)(v15 + 40);
            if ( !(_WORD)v18 || *(_BYTE *)(v17 + 500 * v18 + v16 + 6414) == 2 )
              a2 = sub_3417A70(v13, a2);
          }
        }
        else if ( (unsigned int)(v17 - 242) <= 1 )
        {
          goto LABEL_17;
        }
      }
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
      v12 = v28;
      goto LABEL_10;
    }
  }
  v19 = v38;
  v20 = v39;
  if ( v38 )
  {
    nullsub_1875();
    v26 = v20;
    a2 = 0;
    v25 = v19;
    v13[24].m128i_i64[0] = v19;
    v13[24].m128i_i32[2] = v26;
    sub_33E2B60();
  }
  else
  {
    v24 = v39;
    v23 = 0;
    v13[24].m128i_i64[0] = 0;
    v13[24].m128i_i32[2] = v24;
  }
  *(_QWORD *)(v31 + 768) = v30;
  sub_33CF710((__int64)v33);
  result = *(void (**)())(*(_QWORD *)a1 + 40LL);
  if ( result != nullsub_1877 )
    return (void (*)())((__int64 (__fastcall *)(__int64, __int64))result)(a1, a2);
  return result;
}
