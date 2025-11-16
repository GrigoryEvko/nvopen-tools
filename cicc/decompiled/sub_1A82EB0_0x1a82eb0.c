// Function: sub_1A82EB0
// Address: 0x1a82eb0
//
char __fastcall sub_1A82EB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // r8
  unsigned int v9; // esi
  __int64 v10; // rdi
  int v11; // r10d
  __int64 *v12; // r9
  unsigned int v13; // ecx
  __int64 *v14; // rdx
  int v15; // eax
  int v16; // edx
  __m128i *v17; // rsi
  __int64 v18; // r15
  __int64 v19; // r14
  __int64 v20; // rsi
  unsigned int v21; // esi
  __int64 v22; // rdi
  int v23; // r10d
  __int64 *v24; // r9
  unsigned int v25; // ecx
  __int64 *v26; // rdx
  int v27; // eax
  int v28; // edx
  __m128i *v29; // rsi
  __m128i *v30; // rsi
  int v31; // eax
  int v32; // edx
  __int64 v33; // rsi
  int v34; // r8d
  __int64 *v35; // rdi
  unsigned int v36; // eax
  __int64 v37; // rcx
  int v38; // eax
  int v39; // eax
  __int64 v40; // rcx
  unsigned int v41; // r14d
  __int64 v42; // rdx
  int v43; // edi
  __int64 *v44; // rsi
  int v45; // edx
  int v46; // edx
  int v47; // esi
  __int64 *v48; // rcx
  unsigned int j; // eax
  __int64 v50; // rdi
  int v51; // edx
  int v52; // edx
  __int64 v53; // rdi
  int v54; // ecx
  unsigned int i; // r14d
  __int64 *v56; // rax
  __int64 v57; // rsi
  unsigned int v58; // eax
  unsigned int v59; // r14d
  __int64 v61; // [rsp+10h] [rbp-70h] BYREF
  __int64 v62; // [rsp+18h] [rbp-68h] BYREF
  __m128i v63; // [rsp+20h] [rbp-60h] BYREF
  char v64; // [rsp+40h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) == 5 )
  {
    LOBYTE(v7) = sub_1A82450(a2);
    if ( !(_BYTE)v7 )
      return v7;
    v21 = *(_DWORD *)(a4 + 24);
    if ( v21 )
    {
      v22 = *(_QWORD *)(a4 + 8);
      v23 = 1;
      v24 = 0;
      v25 = (v21 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v26 = (__int64 *)(v22 + 8LL * v25);
      v7 = *v26;
      if ( *v26 == a2 )
        return v7;
      while ( v7 != -8 )
      {
        if ( v7 == -16 && !v24 )
          v24 = v26;
        v25 = (v21 - 1) & (v23 + v25);
        v26 = (__int64 *)(v22 + 8LL * v25);
        v7 = *v26;
        if ( *v26 == a2 )
          return v7;
        ++v23;
      }
      v27 = *(_DWORD *)(a4 + 16);
      if ( !v24 )
        v24 = v26;
      ++*(_QWORD *)a4;
      v28 = v27 + 1;
      if ( 4 * (v27 + 1) < 3 * v21 )
      {
        LODWORD(v7) = v21 - *(_DWORD *)(a4 + 20) - v28;
        if ( (unsigned int)v7 > v21 >> 3 )
        {
LABEL_36:
          *(_DWORD *)(a4 + 16) = v28;
          if ( *v24 != -8 )
            --*(_DWORD *)(a4 + 20);
          *v24 = a2;
          v29 = *(__m128i **)(a3 + 8);
          v63.m128i_i64[0] = a2;
          v63.m128i_i8[8] = 0;
          if ( v29 == *(__m128i **)(a3 + 16) )
          {
            LOBYTE(v7) = sub_1A82AC0((const __m128i **)a3, v29, &v63);
          }
          else
          {
            if ( v29 )
            {
              *v29 = _mm_loadu_si128(&v63);
              v29 = *(__m128i **)(a3 + 8);
            }
            *(_QWORD *)(a3 + 8) = v29 + 1;
          }
          return v7;
        }
        sub_1353F00(a4, v21);
        v38 = *(_DWORD *)(a4 + 24);
        if ( v38 )
        {
          v39 = v38 - 1;
          v40 = *(_QWORD *)(a4 + 8);
          v41 = v39 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v24 = (__int64 *)(v40 + 8LL * v41);
          v42 = *v24;
          if ( *v24 != a2 )
          {
            v43 = 1;
            v44 = 0;
            while ( v42 != -8 )
            {
              if ( !v44 && v42 == -16 )
                v44 = v24;
              v41 = v39 & (v43 + v41);
              v24 = (__int64 *)(v40 + 8LL * v41);
              v42 = *v24;
              if ( *v24 == a2 )
                goto LABEL_51;
              ++v43;
            }
            LODWORD(v7) = *(_DWORD *)(a4 + 16);
            v28 = v7 + 1;
            if ( v44 )
              v24 = v44;
            goto LABEL_36;
          }
          goto LABEL_51;
        }
        goto LABEL_109;
      }
    }
    else
    {
      ++*(_QWORD *)a4;
    }
    sub_1353F00(a4, 2 * v21);
    v31 = *(_DWORD *)(a4 + 24);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = *(_QWORD *)(a4 + 8);
      v34 = 1;
      v35 = 0;
      v36 = (v31 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v24 = (__int64 *)(v33 + 8LL * v36);
      v37 = *v24;
      if ( *v24 != a2 )
      {
        while ( v37 != -8 )
        {
          if ( v37 == -16 && !v35 )
            v35 = v24;
          v36 = v32 & (v34 + v36);
          v24 = (__int64 *)(v33 + 8LL * v36);
          v37 = *v24;
          if ( *v24 == a2 )
            goto LABEL_51;
          ++v34;
        }
        LODWORD(v7) = *(_DWORD *)(a4 + 16);
        v28 = v7 + 1;
        if ( v35 )
          v24 = v35;
        goto LABEL_36;
      }
LABEL_51:
      LODWORD(v7) = *(_DWORD *)(a4 + 16);
      v28 = v7 + 1;
      goto LABEL_36;
    }
LABEL_109:
    ++*(_DWORD *)(a4 + 16);
    BUG();
  }
  LOBYTE(v7) = sub_1A82450(a2);
  if ( (_BYTE)v7 )
  {
    v7 = *(_QWORD *)a2;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
      v7 = **(_QWORD **)(v7 + 16);
    LODWORD(v7) = *(_DWORD *)(v7 + 8) >> 8;
    if ( *(_DWORD *)(v8 + 156) == (_DWORD)v7 )
    {
      v9 = *(_DWORD *)(a4 + 24);
      if ( v9 )
      {
        v10 = *(_QWORD *)(a4 + 8);
        v11 = 1;
        v12 = 0;
        v13 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v14 = (__int64 *)(v10 + 8LL * v13);
        v7 = *v14;
        if ( *v14 == a2 )
          return v7;
        while ( v7 != -8 )
        {
          if ( v7 == -16 && !v12 )
            v12 = v14;
          v13 = (v9 - 1) & (v11 + v13);
          v14 = (__int64 *)(v10 + 8LL * v13);
          v7 = *v14;
          if ( *v14 == a2 )
            return v7;
          ++v11;
        }
        v15 = *(_DWORD *)(a4 + 16);
        if ( !v12 )
          v12 = v14;
        ++*(_QWORD *)a4;
        v16 = v15 + 1;
        if ( 4 * (v15 + 1) < 3 * v9 )
        {
          if ( v9 - *(_DWORD *)(a4 + 20) - v16 > v9 >> 3 )
          {
LABEL_15:
            *(_DWORD *)(a4 + 16) = v16;
            if ( *v12 != -8 )
              --*(_DWORD *)(a4 + 20);
            *v12 = a2;
            v17 = *(__m128i **)(a3 + 8);
            v63.m128i_i64[0] = a2;
            v63.m128i_i8[8] = 0;
            if ( v17 == *(__m128i **)(a3 + 16) )
            {
              sub_1A82AC0((const __m128i **)a3, v17, &v63);
            }
            else
            {
              if ( v17 )
              {
                *v17 = _mm_loadu_si128(&v63);
                v17 = *(__m128i **)(a3 + 8);
              }
              *(_QWORD *)(a3 + 8) = v17 + 1;
            }
            LODWORD(v7) = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
            if ( (_DWORD)v7 )
            {
              v18 = 0;
              v19 = 24LL * (unsigned int)v7;
              do
              {
                if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
                  v7 = *(_QWORD *)(a2 - 8);
                else
                  v7 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
                v20 = *(_QWORD *)(v7 + v18);
                if ( *(_BYTE *)(v20 + 16) == 5 )
                {
                  v61 = *(_QWORD *)(v7 + v18);
                  LOBYTE(v7) = sub_1A82450(v61);
                  if ( (_BYTE)v7 )
                  {
                    v62 = v20;
                    LOBYTE(v7) = sub_1A82C40((__int64)&v63, a4, &v62);
                    if ( v64 )
                    {
                      v63.m128i_i8[0] = 0;
                      v30 = *(__m128i **)(a3 + 8);
                      if ( v30 == *(__m128i **)(a3 + 16) )
                      {
                        LOBYTE(v7) = sub_1A82930((const __m128i **)a3, v30, &v61, &v63);
                      }
                      else
                      {
                        if ( v30 )
                        {
                          v30->m128i_i64[0] = v61;
                          LOBYTE(v7) = v63.m128i_i8[0];
                          v30->m128i_i8[8] = v63.m128i_i8[0];
                          v30 = *(__m128i **)(a3 + 8);
                        }
                        *(_QWORD *)(a3 + 8) = v30 + 1;
                      }
                    }
                  }
                }
                v18 += 24;
              }
              while ( v19 != v18 );
            }
            return v7;
          }
          sub_1353F00(a4, v9);
          v51 = *(_DWORD *)(a4 + 24);
          if ( v51 )
          {
            v52 = v51 - 1;
            v54 = 1;
            v12 = 0;
            for ( i = v52 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)); ; i = v52 & v59 )
            {
              v53 = *(_QWORD *)(a4 + 8);
              v56 = (__int64 *)(v53 + 8LL * i);
              v57 = *v56;
              if ( *v56 == a2 )
              {
                v12 = (__int64 *)(v53 + 8LL * i);
                v16 = *(_DWORD *)(a4 + 16) + 1;
                goto LABEL_15;
              }
              if ( v57 == -8 )
                break;
              if ( v57 != -16 || v12 )
                v56 = v12;
              v59 = v54 + i;
              v12 = v56;
              ++v54;
            }
            v16 = *(_DWORD *)(a4 + 16) + 1;
            if ( !v12 )
              v12 = (__int64 *)(v53 + 8LL * i);
            goto LABEL_15;
          }
LABEL_108:
          ++*(_DWORD *)(a4 + 16);
          BUG();
        }
      }
      else
      {
        ++*(_QWORD *)a4;
      }
      sub_1353F00(a4, 2 * v9);
      v45 = *(_DWORD *)(a4 + 24);
      if ( v45 )
      {
        v46 = v45 - 1;
        v47 = 1;
        v48 = 0;
        for ( j = v46 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)); ; j = v46 & v58 )
        {
          v12 = (__int64 *)(*(_QWORD *)(a4 + 8) + 8LL * j);
          v50 = *v12;
          if ( *v12 == a2 )
          {
            v16 = *(_DWORD *)(a4 + 16) + 1;
            goto LABEL_15;
          }
          if ( v50 == -8 )
            break;
          if ( v50 != -16 || v48 )
            v12 = v48;
          v58 = v47 + j;
          v48 = v12;
          ++v47;
        }
        if ( v48 )
          v12 = v48;
        v16 = *(_DWORD *)(a4 + 16) + 1;
        goto LABEL_15;
      }
      goto LABEL_108;
    }
  }
  return v7;
}
