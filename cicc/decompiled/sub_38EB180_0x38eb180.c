// Function: sub_38EB180
// Address: 0x38eb180
//
__int64 __fastcall sub_38EB180(__int64 a1)
{
  __int64 v1; // r15
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned __int64 v5; // r12
  __m128i v6; // xmm1
  bool v7; // cc
  unsigned __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rcx
  void (__fastcall *v15)(__int64, _QWORD *); // rax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // r12
  __m128i v18; // xmm0
  unsigned __int64 v19; // rdi
  __int64 v20; // rax
  unsigned __int64 v21; // rsi
  __int64 v22; // rsi
  __int64 v23; // r12
  void (__fastcall *v24)(__int64, _QWORD *); // rbx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v29; // [rsp+8h] [rbp-78h]
  __int64 v30; // [rsp+10h] [rbp-70h] BYREF
  __int64 v31; // [rsp+18h] [rbp-68h]
  _QWORD v32[2]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v33; // [rsp+30h] [rbp-50h]
  unsigned __int64 v34; // [rsp+38h] [rbp-48h]
  unsigned int v35; // [rsp+40h] [rbp-40h]

  v1 = a1 + 144;
  v29 = a1 + 152;
  if ( **(_DWORD **)(a1 + 152) == 1 )
    goto LABEL_28;
LABEL_2:
  if ( *(_DWORD *)sub_3909460(a1) == 9 )
    goto LABEL_29;
  while ( 1 )
  {
    v3 = *(_QWORD *)(a1 + 152);
    v4 = *(unsigned int *)(a1 + 160);
    *(_BYTE *)(a1 + 258) = *(_DWORD *)v3 == 9;
    v5 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v4 - 40) >> 3);
    if ( (unsigned __int64)(40 * v4) > 0x28 )
    {
      do
      {
        v6 = _mm_loadu_si128((const __m128i *)(v3 + 48));
        v7 = *(_DWORD *)(v3 + 32) <= 0x40u;
        *(_DWORD *)v3 = *(_DWORD *)(v3 + 40);
        *(__m128i *)(v3 + 8) = v6;
        if ( !v7 )
        {
          v8 = *(_QWORD *)(v3 + 24);
          if ( v8 )
            j_j___libc_free_0_0(v8);
        }
        v9 = *(_QWORD *)(v3 + 64);
        v3 += 40;
        *(_QWORD *)(v3 - 16) = v9;
        LODWORD(v9) = *(_DWORD *)(v3 + 32);
        *(_DWORD *)(v3 + 32) = 0;
        *(_DWORD *)(v3 - 8) = v9;
        --v5;
      }
      while ( v5 );
      LODWORD(v4) = *(_DWORD *)(a1 + 160);
      v3 = *(_QWORD *)(a1 + 152);
    }
    while ( 1 )
    {
      v10 = (unsigned int)(v4 - 1);
      *(_DWORD *)(a1 + 160) = v10;
      v11 = v3 + 40 * v10;
      if ( *(_DWORD *)(v11 + 32) > 0x40u )
      {
        v12 = *(_QWORD *)(v11 + 24);
        if ( v12 )
          j_j___libc_free_0_0(v12);
      }
      if ( !*(_DWORD *)(a1 + 160) )
      {
        sub_392C2E0(v32, v1);
        sub_38E90E0(v29, *(_QWORD *)(a1 + 152), (unsigned __int64)v32);
        if ( v35 > 0x40 )
        {
          if ( v34 )
            j_j___libc_free_0_0(v34);
        }
      }
      v3 = *(_QWORD *)(a1 + 152);
      if ( *(_DWORD *)v3 != 7 )
        break;
      if ( *(_BYTE *)(*(_QWORD *)(a1 + 336) + 393LL) )
      {
        v13 = *(_QWORD *)(a1 + 328);
        v14 = *(_QWORD *)(v3 + 8);
        v15 = *(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v13 + 128LL);
        v31 = *(_QWORD *)(v3 + 16);
        v33 = 261;
        v30 = v14;
        v32[0] = &v30;
        v15(v13, v32);
        v3 = *(_QWORD *)(a1 + 152);
      }
      v16 = *(unsigned int *)(a1 + 160);
      *(_BYTE *)(a1 + 258) = *(_DWORD *)v3 == 9;
      LODWORD(v4) = v16;
      v16 *= 40LL;
      v17 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v16 - 40) >> 3);
      if ( v16 > 0x28 )
      {
        do
        {
          v18 = _mm_loadu_si128((const __m128i *)(v3 + 48));
          v7 = *(_DWORD *)(v3 + 32) <= 0x40u;
          *(_DWORD *)v3 = *(_DWORD *)(v3 + 40);
          *(__m128i *)(v3 + 8) = v18;
          if ( !v7 )
          {
            v19 = *(_QWORD *)(v3 + 24);
            if ( v19 )
              j_j___libc_free_0_0(v19);
          }
          v20 = *(_QWORD *)(v3 + 64);
          v3 += 40;
          *(_QWORD *)(v3 - 16) = v20;
          LODWORD(v20) = *(_DWORD *)(v3 + 32);
          *(_DWORD *)(v3 + 32) = 0;
          *(_DWORD *)(v3 - 8) = v20;
          --v17;
        }
        while ( v17 );
        LODWORD(v4) = *(_DWORD *)(a1 + 160);
        v3 = *(_QWORD *)(a1 + 152);
      }
    }
    if ( *(_DWORD *)v3 )
      break;
    v21 = *(_QWORD *)(**(_QWORD **)(a1 + 344) + 24LL * (unsigned int)(*(_DWORD *)(a1 + 376) - 1) + 16);
    if ( !v21 )
      break;
    sub_38E2E70(a1, v21, 0);
    if ( **(_DWORD **)(a1 + 152) != 1 )
      goto LABEL_2;
LABEL_28:
    v22 = *(_QWORD *)(a1 + 208);
    v33 = 260;
    v32[0] = a1 + 216;
    sub_3909790(a1, v22, v32, 0, 0);
    if ( *(_DWORD *)sub_3909460(a1) == 9 )
    {
LABEL_29:
      if ( *(_QWORD *)(sub_3909460(a1) + 16)
        && **(_BYTE **)(sub_3909460(a1) + 8) != 10
        && **(_BYTE **)(sub_3909460(a1) + 8) != 13
        && *(_BYTE *)(*(_QWORD *)(a1 + 336) + 393LL) )
      {
        v23 = *(_QWORD *)(a1 + 328);
        v24 = *(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v23 + 128LL);
        v25 = sub_3909460(a1);
        v26 = *(_QWORD *)(v25 + 16);
        v27 = *(_QWORD *)(v25 + 8);
        v33 = 261;
        v30 = v27;
        v31 = v26;
        v32[0] = &v30;
        v24(v23, v32);
      }
    }
  }
  return *(_QWORD *)(a1 + 152);
}
