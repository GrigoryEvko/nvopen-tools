// Function: sub_1E451E0
// Address: 0x1e451e0
//
void __fastcall sub_1E451E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rsi
  unsigned int v14; // edi
  __int64 *v15; // rdx
  __int64 v16; // rsi
  int v17; // eax
  void (*v18)(); // rax
  __int64 v19; // rdi
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdi
  unsigned int v24; // esi
  __int64 *v25; // rcx
  __int64 v26; // r10
  __int64 v27; // rdx
  _QWORD *v28; // rcx
  _QWORD *k; // rdi
  _QWORD *v30; // rax
  _QWORD *v31; // rcx
  int j; // ecx
  int v33; // r11d
  const __m128i *v34; // rbx
  unsigned int v35; // edx
  const __m128i *i; // r12
  __int64 v37; // rcx
  __int64 v38; // rsi
  int v39; // edi
  __int64 v40; // rax
  int v41; // eax
  __m128i v42; // xmm0
  __int64 v43; // r12
  __int64 v44; // rbx
  int v45; // edx
  int v46; // r11d
  __int64 v47; // [rsp+8h] [rbp-C8h]
  __int64 v48; // [rsp+18h] [rbp-B8h]
  unsigned int v49; // [rsp+24h] [rbp-ACh]
  int v50; // [rsp+24h] [rbp-ACh]
  __int64 v51; // [rsp+30h] [rbp-A0h]
  unsigned int v52; // [rsp+38h] [rbp-98h]
  int v53; // [rsp+3Ch] [rbp-94h]
  unsigned __int64 v54; // [rsp+40h] [rbp-90h] BYREF
  __int64 v55; // [rsp+48h] [rbp-88h]
  _BYTE *v56; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v57; // [rsp+58h] [rbp-78h]
  unsigned int v58; // [rsp+5Ch] [rbp-74h]
  _BYTE v59[112]; // [rsp+60h] [rbp-70h] BYREF

  v56 = v59;
  v7 = *(_QWORD *)(a1 + 48);
  v8 = *(_QWORD *)(a1 + 32);
  v58 = 4;
  v48 = *(_QWORD *)(v8 + 16);
  v47 = *(_QWORD *)(a1 + 56);
  if ( v7 != v47 )
  {
    while ( 1 )
    {
      v57 = 0;
      v9 = *(_QWORD *)(v7 + 8);
      v53 = 0;
      v10 = *(_QWORD *)(v9 + 32);
      v52 = 0;
      v51 = v10 + 40LL * *(unsigned int *)(v9 + 40);
      if ( v10 != v51 )
        break;
LABEL_40:
      if ( byte_4FC6E40 )
      {
        v34 = *(const __m128i **)(v7 + 32);
        v35 = v57;
        for ( i = &v34[*(unsigned int *)(v7 + 40)]; i != v34; ++v57 )
        {
          while ( 1 )
          {
            v37 = *(_QWORD *)((v34->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 8);
            if ( (**(_WORD **)(v37 + 16) == 45 || !**(_WORD **)(v37 + 16))
              && (((unsigned __int8)v34->m128i_i64[0] ^ 6) & 6) == 0 )
            {
              if ( **(_WORD **)(*(_QWORD *)(v7 + 8) + 16LL) && **(_WORD **)(*(_QWORD *)(v7 + 8) + 16LL) != 45 )
                break;
              v38 = *(_QWORD *)(v37 + 32);
              if ( *(_DWORD *)(v38 + 8) != v52 )
              {
                v39 = *(_DWORD *)(v37 + 40);
                a5 = *(_QWORD *)(v37 + 24);
                if ( v39 == 1 )
                {
LABEL_85:
                  v41 = 0;
                }
                else
                {
                  v40 = 1;
                  while ( a5 != *(_QWORD *)(v38 + 40LL * (unsigned int)(v40 + 1) + 24) )
                  {
                    v40 = (unsigned int)(v40 + 2);
                    if ( v39 == (_DWORD)v40 )
                      goto LABEL_85;
                  }
                  v41 = *(_DWORD *)(v38 + 40 * v40 + 8);
                }
                if ( v41 != v53 )
                  break;
              }
            }
            if ( i == ++v34 )
              goto LABEL_80;
          }
          if ( v35 >= v58 )
          {
            sub_16CD150((__int64)&v56, v59, 0, 16, a5, a6);
            v35 = v57;
          }
          v42 = _mm_loadu_si128(v34++);
          *(__m128i *)&v56[16 * v35] = v42;
          v35 = v57 + 1;
        }
LABEL_80:
        if ( v35 )
        {
          v43 = 0;
          v44 = 16 * (v35 - 1 + 1LL);
          do
          {
            v43 += 16;
            sub_1F01C30(v7);
          }
          while ( v43 != v44 );
        }
      }
      v7 += 272;
      if ( v47 == v7 )
      {
        if ( v56 != v59 )
          _libc_free((unsigned __int64)v56);
        return;
      }
    }
    while ( 1 )
    {
      if ( *(_BYTE *)v10 )
        goto LABEL_16;
      LODWORD(a5) = *(_DWORD *)(v10 + 8);
      v19 = *(_QWORD *)(a1 + 40);
      if ( (*(_BYTE *)(v10 + 3) & 0x10) != 0 )
      {
        if ( (int)a5 < 0 )
          v20 = *(_QWORD *)(*(_QWORD *)(v19 + 24) + 16 * (a5 & 0x7FFFFFFF) + 8);
        else
          v20 = *(_QWORD *)(*(_QWORD *)(v19 + 272) + 8LL * (unsigned int)a5);
        if ( !v20 )
          goto LABEL_16;
        if ( (*(_BYTE *)(v20 + 3) & 0x10) == 0 )
        {
LABEL_23:
          v21 = *(_QWORD *)(v20 + 16);
LABEL_24:
          v22 = *(unsigned int *)(a1 + 976);
          if ( !(_DWORD)v22 )
            goto LABEL_33;
          LODWORD(a6) = v22 - 1;
          v23 = *(_QWORD *)(a1 + 960);
          v24 = (v22 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v25 = (__int64 *)(v23 + 16LL * v24);
          v26 = *v25;
          if ( v21 != *v25 )
          {
            for ( j = 1; ; j = v33 )
            {
              if ( v26 == -8 )
                goto LABEL_33;
              v33 = j + 1;
              v24 = a6 & (j + v24);
              v25 = (__int64 *)(v23 + 16LL * v24);
              v26 = *v25;
              if ( v21 == *v25 )
                break;
            }
          }
          if ( v25 != (__int64 *)(v23 + 16 * v22) )
          {
            v27 = v25[1];
            if ( v27 )
            {
              if ( **(_WORD **)(v21 + 16) == 45 || !**(_WORD **)(v21 + 16) )
              {
                if ( **(_WORD **)(v9 + 16) && **(_WORD **)(v9 + 16) != 45 )
                {
                  v55 = (unsigned int)a5 | 0x100000000LL;
                  v50 = a5;
                  v54 = v27 & 0xFFFFFFFFFFFFFFF9LL | 2;
                  sub_1F01A00(v7, &v54, 1);
                  v21 = *(_QWORD *)(v20 + 16);
                  LODWORD(a5) = v50;
                  goto LABEL_33;
                }
                v53 = a5;
                if ( *(_DWORD *)(v27 + 192) < *(_DWORD *)(v7 + 192) )
                {
                  v28 = *(_QWORD **)(v7 + 32);
                  for ( k = &v28[2 * *(unsigned int *)(v7 + 40)]; k != v28; v28 += 2 )
                  {
                    if ( v27 == (*v28 & 0xFFFFFFFFFFFFFFF8LL) )
                    {
                      v53 = a5;
                      goto LABEL_33;
                    }
                  }
                  v53 = a5;
                  v54 = v27 | 6;
                  v55 = 0;
                  sub_1F01A00(v7, &v54, 1);
                  v21 = *(_QWORD *)(v20 + 16);
                  LODWORD(a5) = v53;
                }
              }
            }
          }
LABEL_33:
          while ( 1 )
          {
            v20 = *(_QWORD *)(v20 + 32);
            if ( !v20 )
              goto LABEL_16;
            if ( (*(_BYTE *)(v20 + 3) & 0x10) == 0 && v21 != *(_QWORD *)(v20 + 16) )
            {
              v21 = *(_QWORD *)(v20 + 16);
              goto LABEL_24;
            }
          }
        }
        while ( 1 )
        {
          v20 = *(_QWORD *)(v20 + 32);
          if ( !v20 )
            break;
          if ( (*(_BYTE *)(v20 + 3) & 0x10) == 0 )
            goto LABEL_23;
        }
        v10 += 40;
        if ( v51 == v10 )
          goto LABEL_40;
      }
      else
      {
        v49 = *(_DWORD *)(v10 + 8);
        v11 = sub_1E69D60(v19);
        if ( v11 )
        {
          v12 = *(unsigned int *)(a1 + 976);
          if ( (_DWORD)v12 )
          {
            v13 = *(_QWORD *)(a1 + 960);
            LODWORD(a5) = v49;
            v14 = (v12 - 1) & (((unsigned int)v11 >> 4) ^ ((unsigned int)v11 >> 9));
            v15 = (__int64 *)(v13 + 16LL * v14);
            a6 = *v15;
            if ( v11 != *v15 )
            {
              v45 = 1;
              while ( a6 != -8 )
              {
                v46 = v45 + 1;
                v14 = (v12 - 1) & (v45 + v14);
                v15 = (__int64 *)(v13 + 16LL * v14);
                a6 = *v15;
                if ( v11 == *v15 )
                  goto LABEL_7;
                v45 = v46;
              }
              goto LABEL_16;
            }
LABEL_7:
            if ( v15 != (__int64 *)(v13 + 16 * v12) )
            {
              v16 = v15[1];
              if ( v16 )
              {
                v17 = **(unsigned __int16 **)(v11 + 16);
                if ( v17 == 45 || !v17 )
                {
                  if ( **(_WORD **)(v9 + 16) == 45 || !**(_WORD **)(v9 + 16) )
                  {
                    v52 = v49;
                    if ( *(_DWORD *)(v16 + 192) >= *(_DWORD *)(v7 + 192) )
                      goto LABEL_16;
                    v30 = *(_QWORD **)(v7 + 32);
                    v31 = &v30[2 * *(unsigned int *)(v7 + 40)];
                    if ( v30 != v31 )
                    {
                      while ( v16 != (*v30 & 0xFFFFFFFFFFFFFFF8LL) )
                      {
                        v30 += 2;
                        if ( v31 == v30 )
                          goto LABEL_90;
                      }
                      v52 = v49;
                      goto LABEL_16;
                    }
LABEL_90:
                    v52 = v49;
                    v54 = v16 | 6;
                    v55 = 0;
                  }
                  else
                  {
                    v55 = v49;
                    v54 = v16 & 0xFFFFFFFFFFFFFFF9LL;
                    v18 = *(void (**)())(*(_QWORD *)v48 + 208LL);
                    if ( v18 != nullsub_681 )
                      ((void (__fastcall *)(__int64, __int64, __int64, unsigned __int64 *))v18)(v48, v16, v7, &v54);
                  }
                  sub_1F01A00(v7, &v54, 1);
                }
              }
            }
          }
        }
LABEL_16:
        v10 += 40;
        if ( v51 == v10 )
          goto LABEL_40;
      }
    }
  }
}
