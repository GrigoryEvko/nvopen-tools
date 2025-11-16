// Function: sub_2437F60
// Address: 0x2437f60
//
void __fastcall sub_2437F60(__int64 a1, __int64 *a2, __int64 a3, __int64 *a4, __int64 a5)
{
  unsigned __int8 *v5; // r10
  unsigned __int8 v10; // al
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdx
  char v25; // al
  char v26; // al
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // rdx
  int v30; // eax
  __int64 v31; // rdi
  int v32; // edx
  __int64 v33; // rax
  __int64 v34; // rax
  const __m128i *v35; // r15
  __int64 v36; // r9
  __int64 v37; // rdx
  unsigned __int64 v38; // r8
  unsigned __int64 v39; // rax
  __m128i *v40; // rax
  const void *v41; // rsi
  __int64 v42; // [rsp+0h] [rbp-B0h]
  unsigned __int64 v43; // [rsp+0h] [rbp-B0h]
  __int64 v44; // [rsp+8h] [rbp-A8h]
  int v45; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v46; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v47; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v48; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v49; // [rsp+8h] [rbp-A8h]
  unsigned __int8 *v50; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v51; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v52; // [rsp+10h] [rbp-A0h]
  unsigned __int8 *v53; // [rsp+10h] [rbp-A0h]
  unsigned int v54; // [rsp+10h] [rbp-A0h]
  __int64 v55[2]; // [rsp+18h] [rbp-98h] BYREF
  char v56; // [rsp+2Ah] [rbp-86h] BYREF
  char v57; // [rsp+2Bh] [rbp-85h] BYREF
  unsigned int v58; // [rsp+2Ch] [rbp-84h] BYREF
  __int64 v59[16]; // [rsp+30h] [rbp-80h] BYREF

  v5 = (unsigned __int8 *)a3;
  v55[0] = a3;
  if ( (*(_BYTE *)(a3 + 7) & 0x20) != 0 )
  {
    if ( sub_B91C10(a3, 31) )
      return;
    v5 = (unsigned __int8 *)v55[0];
  }
  if ( *(unsigned __int8 **)(a1 + 512) == v5 )
    return;
  v10 = *v5;
  if ( *v5 == 61 )
  {
    if ( byte_4FE5608 )
    {
      v50 = v5;
      if ( !(unsigned __int8)sub_2434DB0(a1, a2, (unsigned __int64)v5, *((_QWORD *)v5 - 4)) )
      {
        _BitScanReverse64(&v11, 1LL << (*((_WORD *)v50 + 1) >> 1));
        v56 = 63 - (v11 ^ 0x3F);
        v12 = *((_QWORD *)v50 + 1);
        v57 = 0;
        v59[0] = v12;
        v58 = 0;
LABEL_10:
        sub_2437640(a5, v55, &v58, &v57, v59, &v56);
        return;
      }
    }
    return;
  }
  if ( v10 != 62 )
  {
    if ( v10 == 66 )
    {
      if ( !byte_4FE5448 )
        return;
      v52 = v5;
      if ( (unsigned __int8)sub_2434DB0(a1, a2, (unsigned __int64)v5, *((_QWORD *)v5 - 8)) )
        return;
      v15 = *((_QWORD *)v52 - 4);
      goto LABEL_19;
    }
    if ( v10 == 65 )
    {
      if ( !byte_4FE5448 )
        return;
      v53 = v5;
      if ( (unsigned __int8)sub_2434DB0(a1, a2, (unsigned __int64)v5, *((_QWORD *)v5 - 12)) )
        return;
      v15 = *((_QWORD *)v53 - 8);
LABEL_19:
      v16 = *(_QWORD *)(v15 + 8);
      v57 = 1;
      v59[0] = v16;
      v58 = 0;
      sub_2437400(a5, v55, &v58, &v57, v59);
      return;
    }
    if ( v10 != 85 )
      return;
    v54 = 0;
LABEL_26:
    v17 = 0;
    while ( 1 )
    {
      v18 = -32 - 32LL * v17;
      if ( (v5[7] & 0x80u) == 0 )
        goto LABEL_47;
      v42 = (__int64)v5;
      v19 = sub_BD2BC0((__int64)v5);
      v5 = (unsigned __int8 *)v42;
      v44 = v20 + v19;
      if ( *(char *)(v42 + 7) >= 0 )
        break;
      v21 = sub_BD2BC0(v42);
      v5 = (unsigned __int8 *)v42;
      if ( !(unsigned int)((v44 - v21) >> 4) )
        goto LABEL_47;
      if ( *(char *)(v42 + 7) >= 0 )
        goto LABEL_63;
      v45 = *(_DWORD *)(sub_BD2BC0(v42) + 8);
      if ( *(char *)(v42 + 7) >= 0 )
        BUG();
      v22 = sub_BD2BC0(v42);
      v5 = (unsigned __int8 *)v42;
      v24 = 32LL * (unsigned int)(*(_DWORD *)(v22 + v23 - 4) - v45);
LABEL_33:
      if ( v54 >= (unsigned int)((32LL * (*((_DWORD *)v5 + 1) & 0x7FFFFFF) + v18 - v24) >> 5) )
      {
        sub_F58670((__int64)v5, a4);
        return;
      }
      if ( byte_4FE5368 )
      {
        v46 = v5;
        v25 = sub_B49B80((__int64)v5, v54, 81);
        v5 = v46;
        if ( v25 )
        {
          v26 = sub_2434DB0(
                  a1,
                  a2,
                  v55[0],
                  *(_QWORD *)&v46[32 * (v54 - (unsigned __int64)(*((_DWORD *)v46 + 1) & 0x7FFFFFF))]);
          v5 = v46;
          if ( !v26 )
          {
            v27 = sub_A748A0((_QWORD *)v46 + 9, v54);
            v5 = v46;
            v28 = v27;
            if ( !v27 )
            {
              v33 = *((_QWORD *)v46 - 4);
              if ( v33 )
              {
                if ( !*(_BYTE *)v33 && *(_QWORD *)(v33 + 24) == *((_QWORD *)v46 + 10) )
                {
                  v59[0] = *(_QWORD *)(v33 + 120);
                  v34 = sub_A748A0(v59, v54);
                  v5 = v46;
                  v28 = v34;
                }
              }
            }
            v29 = *(unsigned int *)(a5 + 8);
            v30 = v29;
            if ( *(_DWORD *)(a5 + 12) <= (unsigned int)v29 )
            {
              v35 = (const __m128i *)v59;
              v49 = v5;
              sub_23DF780((__int64)v59, v55[0], v54, 0, v28, 256, 0, 0, 0);
              v37 = *(unsigned int *)(a5 + 8);
              v5 = v49;
              v38 = v37 + 1;
              v39 = *(_QWORD *)a5;
              if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
              {
                v41 = (const void *)(a5 + 16);
                if ( v39 > (unsigned __int64)v59 || (v43 = *(_QWORD *)a5, (unsigned __int64)v59 >= v39 + 72 * v37) )
                {
                  sub_C8D5F0(a5, v41, v38, 0x48u, v38, v36);
                  v37 = *(unsigned int *)(a5 + 8);
                  v39 = *(_QWORD *)a5;
                  v5 = v49;
                }
                else
                {
                  sub_C8D5F0(a5, v41, v38, 0x48u, v38, v36);
                  v37 = *(unsigned int *)(a5 + 8);
                  v5 = v49;
                  v39 = *(_QWORD *)a5;
                  v35 = (const __m128i *)((char *)v59 + *(_QWORD *)a5 - v43);
                }
              }
              v40 = (__m128i *)(v39 + 72 * v37);
              *v40 = _mm_loadu_si128(v35);
              v40[1] = _mm_loadu_si128(v35 + 1);
              v40[2] = _mm_loadu_si128(v35 + 2);
              v40[3] = _mm_loadu_si128(v35 + 3);
              v40[4].m128i_i64[0] = v35[4].m128i_i64[0];
              ++*(_DWORD *)(a5 + 8);
            }
            else
            {
              v31 = *(_QWORD *)a5 + 72 * v29;
              if ( v31 )
              {
                v47 = v5;
                sub_23DF780(v31, v55[0], v54, 0, v28, 256, 0, 0, 0);
                v30 = *(_DWORD *)(a5 + 8);
                v5 = v47;
              }
              *(_DWORD *)(a5 + 8) = v30 + 1;
            }
          }
        }
      }
      v32 = *v5;
      ++v54;
      switch ( v32 )
      {
        case '(':
          v48 = v5;
          v17 = sub_B491D0((__int64)v5);
          v5 = v48;
          break;
        case 'U':
          goto LABEL_26;
        case '"':
          v17 = 2;
          break;
        default:
          BUG();
      }
    }
    if ( (unsigned int)(v44 >> 4) )
LABEL_63:
      BUG();
LABEL_47:
    v24 = 0;
    goto LABEL_33;
  }
  if ( byte_4FE5528 )
  {
    v51 = v5;
    if ( !(unsigned __int8)sub_2434DB0(a1, a2, (unsigned __int64)v5, *((_QWORD *)v5 - 4)) )
    {
      _BitScanReverse64(&v13, 1LL << (*((_WORD *)v51 + 1) >> 1));
      v56 = 63 - (v13 ^ 0x3F);
      v14 = *(_QWORD *)(*((_QWORD *)v51 - 8) + 8LL);
      v57 = 1;
      v58 = 1;
      v59[0] = v14;
      goto LABEL_10;
    }
  }
}
