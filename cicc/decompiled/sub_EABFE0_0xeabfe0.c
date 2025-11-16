// Function: sub_EABFE0
// Address: 0xeabfe0
//
__int64 __fastcall sub_EABFE0(__int64 a1)
{
  __int64 v1; // r15
  __m128i *v3; // rbx
  __int64 v4; // rax
  unsigned __int64 v5; // r12
  __m128i v6; // xmm1
  bool v7; // cc
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 *v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // rdx
  void (__fastcall *v19)(__int64, __int64 *); // rax
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // r12
  __m128i v22; // xmm0
  __int64 v23; // rdi
  __int64 v24; // rax
  unsigned __int64 v25; // rsi
  __int64 v26; // rsi
  __int64 v27; // r12
  void (__fastcall *v28)(__int64, __int64 *); // rbx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v33; // [rsp+8h] [rbp-68h]
  __int64 v34; // [rsp+10h] [rbp-60h] BYREF
  __int64 v35; // [rsp+18h] [rbp-58h]
  __int64 v36; // [rsp+28h] [rbp-48h]
  unsigned int v37; // [rsp+30h] [rbp-40h]

  v1 = a1 + 40;
  v33 = a1 + 48;
  if ( **(_DWORD **)(a1 + 48) == 1 )
    goto LABEL_28;
LABEL_2:
  if ( *(_DWORD *)sub_ECD7B0(a1) == 9 )
    goto LABEL_29;
  while ( 1 )
  {
    v3 = *(__m128i **)(a1 + 48);
    v4 = *(unsigned int *)(a1 + 56);
    *(_BYTE *)(a1 + 155) = v3->m128i_i32[0] == 9;
    v5 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v4 - 40) >> 3);
    if ( (unsigned __int64)(40 * v4) > 0x28 )
    {
      do
      {
        v6 = _mm_loadu_si128(v3 + 3);
        v7 = v3[2].m128i_i32[0] <= 0x40u;
        v3->m128i_i32[0] = v3[2].m128i_i32[2];
        *(__m128i *)((char *)v3 + 8) = v6;
        if ( !v7 )
        {
          v8 = v3[1].m128i_i64[1];
          if ( v8 )
            j_j___libc_free_0_0(v8);
        }
        v9 = v3[4].m128i_i64[0];
        v3 = (__m128i *)((char *)v3 + 40);
        v3[-1].m128i_i64[0] = v9;
        LODWORD(v9) = v3[2].m128i_i32[0];
        v3[2].m128i_i32[0] = 0;
        v3[-1].m128i_i32[2] = v9;
        --v5;
      }
      while ( v5 );
      LODWORD(v4) = *(_DWORD *)(a1 + 56);
      v3 = *(__m128i **)(a1 + 48);
    }
    while ( 1 )
    {
      v10 = (unsigned int)(v4 - 1);
      *(_DWORD *)(a1 + 56) = v10;
      v11 = &v3->m128i_i64[5 * v10];
      if ( *((_DWORD *)v11 + 8) > 0x40u )
      {
        v12 = v11[3];
        if ( v12 )
          j_j___libc_free_0_0(v12);
      }
      if ( !*(_DWORD *)(a1 + 56) )
      {
        sub_1097F60(&v34, v1);
        sub_EAA0A0(v33, *(_QWORD *)(a1 + 48), (unsigned __int64)&v34, v13, v14, v15);
        if ( v37 > 0x40 )
        {
          if ( v36 )
            j_j___libc_free_0_0(v36);
        }
      }
      v3 = *(__m128i **)(a1 + 48);
      if ( v3->m128i_i32[0] != 7 )
        break;
      if ( *(_BYTE *)(*(_QWORD *)(a1 + 240) + 394LL) )
      {
        v16 = *(_QWORD *)(a1 + 232);
        v17 = v3->m128i_i64[1];
        v18 = v3[1].m128i_i64[0];
        v19 = *(void (__fastcall **)(__int64, __int64 *))(*(_QWORD *)v16 + 144LL);
        LOWORD(v37) = 261;
        v34 = v17;
        v35 = v18;
        v19(v16, &v34);
        v3 = *(__m128i **)(a1 + 48);
      }
      v20 = *(unsigned int *)(a1 + 56);
      *(_BYTE *)(a1 + 155) = v3->m128i_i32[0] == 9;
      LODWORD(v4) = v20;
      v20 *= 40LL;
      v21 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v20 - 40) >> 3);
      if ( v20 > 0x28 )
      {
        do
        {
          v22 = _mm_loadu_si128(v3 + 3);
          v7 = v3[2].m128i_i32[0] <= 0x40u;
          v3->m128i_i32[0] = v3[2].m128i_i32[2];
          *(__m128i *)((char *)v3 + 8) = v22;
          if ( !v7 )
          {
            v23 = v3[1].m128i_i64[1];
            if ( v23 )
              j_j___libc_free_0_0(v23);
          }
          v24 = v3[4].m128i_i64[0];
          v3 = (__m128i *)((char *)v3 + 40);
          v3[-1].m128i_i64[0] = v24;
          LODWORD(v24) = v3[2].m128i_i32[0];
          v3[2].m128i_i32[0] = 0;
          v3[-1].m128i_i32[2] = v24;
          --v21;
        }
        while ( v21 );
        LODWORD(v4) = *(_DWORD *)(a1 + 56);
        v3 = *(__m128i **)(a1 + 48);
      }
    }
    if ( v3->m128i_i32[0] )
      break;
    v25 = *(_QWORD *)(**(_QWORD **)(a1 + 248) + 24LL * (unsigned int)(*(_DWORD *)(a1 + 304) - 1) + 16);
    if ( !v25 )
      break;
    sub_EA24B0(a1, v25, 0);
    if ( **(_DWORD **)(a1 + 48) != 1 )
      goto LABEL_2;
LABEL_28:
    v26 = *(_QWORD *)(a1 + 104);
    LOWORD(v37) = 260;
    v34 = a1 + 112;
    sub_ECDA70(a1, v26, &v34, 0, 0);
    if ( *(_DWORD *)sub_ECD7B0(a1) == 9 )
    {
LABEL_29:
      if ( *(_QWORD *)(sub_ECD7B0(a1) + 16)
        && **(_BYTE **)(sub_ECD7B0(a1) + 8) != 10
        && **(_BYTE **)(sub_ECD7B0(a1) + 8) != 13
        && *(_BYTE *)(*(_QWORD *)(a1 + 240) + 394LL) )
      {
        v27 = *(_QWORD *)(a1 + 232);
        v28 = *(void (__fastcall **)(__int64, __int64 *))(*(_QWORD *)v27 + 144LL);
        v29 = sub_ECD7B0(a1);
        v30 = *(_QWORD *)(v29 + 8);
        v31 = *(_QWORD *)(v29 + 16);
        LOWORD(v37) = 261;
        v34 = v30;
        v35 = v31;
        v28(v27, &v34);
      }
    }
  }
  return *(_QWORD *)(a1 + 48);
}
