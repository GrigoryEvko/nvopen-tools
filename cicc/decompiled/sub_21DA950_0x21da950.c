// Function: sub_21DA950
// Address: 0x21da950
//
__int64 __fastcall sub_21DA950(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r12d
  int v4; // r13d
  __int64 v5; // r15
  __int64 v6; // rax
  _QWORD *v7; // r12
  __int64 v8; // rax
  int v9; // esi
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 (*v15)(); // rax
  _QWORD *v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  const __m128i *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 *v28; // rdx
  __int64 *v29; // r13
  __int64 v30; // rdx
  unsigned __int64 v31; // rsi
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // r13
  __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v40; // rdi
  __int64 v41; // [rsp+0h] [rbp-E0h]
  size_t v42; // [rsp+8h] [rbp-D8h]
  __int64 v43; // [rsp+8h] [rbp-D8h]
  __int64 v44; // [rsp+10h] [rbp-D0h]
  bool v45; // [rsp+10h] [rbp-D0h]
  __int64 v46; // [rsp+18h] [rbp-C8h]
  int v47; // [rsp+18h] [rbp-C8h]
  __int32 v48; // [rsp+18h] [rbp-C8h]
  __int64 v49; // [rsp+20h] [rbp-C0h]
  size_t v50; // [rsp+20h] [rbp-C0h]
  __int64 v51; // [rsp+20h] [rbp-C0h]
  __int64 v52; // [rsp+28h] [rbp-B8h]
  __int64 v53; // [rsp+28h] [rbp-B8h]
  __int64 v54; // [rsp+28h] [rbp-B8h]
  __int64 v55; // [rsp+28h] [rbp-B8h]
  _QWORD *v56; // [rsp+30h] [rbp-B0h]
  _QWORD *v58; // [rsp+48h] [rbp-98h]
  __m128i v59; // [rsp+80h] [rbp-60h] BYREF
  __int64 v60; // [rsp+90h] [rbp-50h]
  __int64 v61; // [rsp+98h] [rbp-48h]
  __int64 v62; // [rsp+A0h] [rbp-40h]

  v2 = sub_1636880(a1, *(_QWORD *)a2);
  if ( (_BYTE)v2 )
    return 0;
  v3 = v2;
  v58 = *(_QWORD **)(a2 + 328);
  v56 = (_QWORD *)(a2 + 320);
  if ( v58 == (_QWORD *)(a2 + 320) )
    goto LABEL_40;
  do
  {
    v4 = v3;
    v5 = v58[4];
    if ( (_QWORD *)v5 == v58 + 3 )
      goto LABEL_39;
    while ( 1 )
    {
      if ( !v5 )
        BUG();
      v6 = v5;
      if ( (*(_BYTE *)v5 & 4) == 0 && (*(_BYTE *)(v5 + 46) & 8) != 0 )
      {
        do
          v6 = *(_QWORD *)(v6 + 8);
        while ( (*(_BYTE *)(v6 + 46) & 8) != 0 );
      }
      v7 = *(_QWORD **)(v6 + 8);
      if ( (**(_WORD **)(v5 + 16) & 0xFFFD) == 0x1300 )
      {
        v8 = *(_QWORD *)(v5 + 32);
        if ( *(_BYTE *)(v8 + 40) )
          goto LABEL_11;
        v9 = *(_DWORD *)(v8 + 48);
        if ( v9 >= 0 )
          goto LABEL_11;
        v10 = *(_QWORD *)(v5 + 24);
        v11 = sub_1E69D60(*(_QWORD *)(*(_QWORD *)(v10 + 56) + 40LL), v9);
        if ( v11 )
        {
          if ( v10 == *(_QWORD *)(v11 + 24) && (unsigned int)**(unsigned __int16 **)(v11 + 16) - 3073 <= 1 )
          {
            v12 = *(_QWORD *)(v11 + 32);
            if ( !*(_BYTE *)(v12 + 40) && *(_DWORD *)(v12 + 48) == 2 )
            {
              v13 = *(_QWORD *)(v5 + 24);
              v14 = *(_QWORD *)(v13 + 56);
              v15 = *(__int64 (**)())(**(_QWORD **)(v14 + 16) + 40LL);
              if ( v15 == sub_1D00B00 )
              {
                sub_1E69D60(*(_QWORD *)(v14 + 40), *(_DWORD *)(*(_QWORD *)(v5 + 32) + 48LL));
                BUG();
              }
              v52 = *(_QWORD *)(v14 + 40);
              v44 = v15();
              v46 = v52;
              v49 = sub_1E69D60(v52, *(_DWORD *)(*(_QWORD *)(v5 + 32) + 48LL));
              LODWORD(v52) = *(_DWORD *)(*(_QWORD *)(v5 + 32) + 8LL);
              v16 = sub_1E0B640(
                      v14,
                      *(_QWORD *)(v44 + 8) + ((unsigned __int64)**(unsigned __int16 **)(v49 + 16) << 6),
                      (__int64 *)(v5 + 64),
                      0);
              v59.m128i_i64[0] = 0x10000000;
              v59.m128i_i32[2] = v52;
              v53 = (__int64)v16;
              v60 = 0;
              v61 = 0;
              v62 = 0;
              sub_1E1A9C0((__int64)v16, v14, &v59);
              v59.m128i_i64[0] = 0;
              v59.m128i_i64[1] = 3;
              v60 = 0;
              v61 = 0;
              v62 = 0;
              sub_1E1A9C0(v53, v14, &v59);
              sub_1E1A9C0(v53, v14, (const __m128i *)(*(_QWORD *)(v49 + 32) + 80LL));
              sub_1DD5BA0((__int64 *)(v13 + 16), v53);
              v17 = *(_QWORD *)v53;
              v18 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v53 + 8) = v5;
              *(_QWORD *)v53 = v18 | v17 & 7;
              *(_QWORD *)(v18 + 8) = v53;
              *(_QWORD *)v5 = *(_QWORD *)v5 & 7LL | v53;
              if ( (unsigned __int8)sub_1E69E00(v46, *(_DWORD *)(*(_QWORD *)(v49 + 32) + 8LL)) )
                sub_1E162E0(v49);
              v4 = 1;
              sub_1E162E0(v5);
            }
          }
        }
      }
      if ( byte_4FD3E60 )
      {
        if ( **(_WORD **)(v5 + 16) == 190 )
        {
          v19 = *(const __m128i **)(v5 + 32);
          if ( !(v19->m128i_i8[0] | _mm_loadu_si128(v19).m128i_i8[3] & 0x10) )
          {
            v54 = *(_QWORD *)(a2 + 40);
            v20 = sub_1E69D60(v54, v19->m128i_i32[2]);
            v21 = v20;
            if ( v20 )
            {
              if ( **(_WORD **)(v20 + 16) == 3358 && *(_DWORD *)(v20 + 40) > 3u )
              {
                v22 = *(_QWORD *)(v20 + 32);
                v45 = *(_BYTE *)(v22 + 120) == 1
                   && *(_QWORD *)(v22 + 104) == 0
                   && *(_BYTE *)(v22 + 40) == 0
                   && *(_BYTE *)(v22 + 80) == 1;
                if ( v45 && *(_BYTE *)(v22 + 144) == 1 )
                {
                  v50 = v54;
                  v47 = *(_DWORD *)(v22 + 48);
                  v23 = sub_1E69D60(v54, v47);
                  v55 = v23;
                  if ( v23 )
                  {
                    if ( **(_WORD **)(v23 + 16) == 141 )
                    {
                      v42 = v50;
                      if ( *(_DWORD *)(v23 + 40) > 2u )
                      {
                        v51 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 40LL))(*(_QWORD *)(a2 + 16));
                        v48 = sub_1E6B9A0(
                                v42,
                                *(_QWORD *)(*(_QWORD *)(v42 + 24) + 16LL * (v47 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
                                (unsigned __int8 *)byte_3F871B3,
                                0,
                                v24,
                                v42);
                        v25 = sub_1E69D60(v42, *(_DWORD *)(*(_QWORD *)(v55 + 32) + 48LL));
                        if ( v25 )
                        {
                          if ( **(_WORD **)(v25 + 16) == 3642 )
                          {
                            v41 = v25;
                            v26 = sub_21DA8E0(a2, (__int64 *)(v25 + 64), *(_QWORD *)(v51 + 8) + 39808LL, v48);
                            v43 = v27;
                            sub_1E1A9C0(v27, v26, (const __m128i *)(*(_QWORD *)(v55 + 32) + 40LL));
                            v28 = *(__int64 **)(v41 + 24);
                            if ( v28 + 3 == (__int64 *)(v28[3] & 0xFFFFFFFFFFFFFFF8LL) )
                              v29 = (__int64 *)v28[4];
                            else
                              v29 = *(__int64 **)(v41 + 8);
                            sub_1DD5BA0(v28 + 2, v43);
                            v30 = *(_QWORD *)v43;
                            v31 = *v29 & 0xFFFFFFFFFFFFFFF8LL;
                            *(_QWORD *)(v43 + 8) = v29;
                            *(_QWORD *)v43 = v31 | v30 & 7;
                            *(_QWORD *)(v31 + 8) = v43;
                            *v29 = *v29 & 7 | v43;
                            v32 = sub_21DA8E0(
                                    a2,
                                    (__int64 *)(v21 + 64),
                                    *(_QWORD *)(v51 + 8) + 214976LL,
                                    *(_DWORD *)(*(_QWORD *)(v21 + 32) + 8LL));
                            v34 = v33;
                            sub_1E1A9C0(v33, v32, (const __m128i *)(*(_QWORD *)(v55 + 32) + 80LL));
                            v59.m128i_i64[0] = 0;
                            v59.m128i_i32[2] = v48;
                            v60 = 0;
                            v61 = 0;
                            v62 = 0;
                            sub_1E1A9C0(v34, v32, &v59);
                            v59.m128i_i64[0] = 1;
                            v60 = 0;
                            v61 = 1;
                            sub_1E1A9C0(v34, v32, &v59);
                            sub_1DD5BA0(v58 + 2, v34);
                            v35 = *(_QWORD *)v5;
                            v36 = *(_QWORD *)v34;
                            *(_QWORD *)(v34 + 8) = v5;
                            v35 &= 0xFFFFFFFFFFFFFFF8LL;
                            *(_QWORD *)v34 = v35 | v36 & 7;
                            *(_QWORD *)(v35 + 8) = v34;
                            *(_QWORD *)v5 = *(_QWORD *)v5 & 7LL | v34;
                            sub_1E162E0(v21);
                            v4 = v45;
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
      }
LABEL_11:
      if ( v7 == v58 + 3 )
        break;
      v5 = (__int64)v7;
    }
    v3 = v4;
LABEL_39:
    v58 = (_QWORD *)v58[1];
  }
  while ( v56 != v58 );
LABEL_40:
  v37 = *(_QWORD *)(a2 + 40);
  v38 = *(_QWORD *)(*(_QWORD *)(v37 + 272) + 16LL);
  if ( v38 )
  {
    if ( (*(_BYTE *)(v38 + 3) & 0x10) == 0 )
      return v3;
    while ( 1 )
    {
      v38 = *(_QWORD *)(v38 + 32);
      if ( !v38 )
        break;
      if ( (*(_BYTE *)(v38 + 3) & 0x10) == 0 )
        return v3;
    }
  }
  v40 = sub_1E69D60(v37, 2);
  if ( v40 )
    sub_1E162E0(v40);
  return v3;
}
