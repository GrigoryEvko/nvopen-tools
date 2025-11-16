// Function: sub_1C579A0
// Address: 0x1c579a0
//
void __fastcall sub_1C579A0(__m128i *src, const __m128i *a2, __int64 a3)
{
  const __m128i *v5; // r12
  unsigned int v7; // r9d
  __int64 v8; // r10
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r11
  unsigned int v12; // r11d
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r8
  const __m128i *v16; // r15
  __int64 v17; // r10
  __int64 v18; // r9
  __int64 v19; // r8
  __int64 v20; // rcx
  unsigned int v21; // esi
  _QWORD *v22; // r8
  __int64 *v23; // r15
  int v24; // ecx
  int v25; // ecx
  __int64 v26; // r10
  __int64 v27; // rsi
  unsigned int v28; // edx
  int v29; // eax
  __int64 *v30; // rdi
  __int64 v31; // r9
  __int64 v32; // rax
  int v33; // edx
  int v34; // edx
  __int64 v35; // r9
  unsigned int v36; // ecx
  int v37; // eax
  __int64 *v38; // rdi
  __int64 v39; // r8
  __int64 v40; // rax
  const __m128i *v41; // rdi
  int v42; // eax
  int v43; // eax
  int v44; // ecx
  int v45; // ecx
  __int64 v46; // r10
  __int64 v47; // rsi
  unsigned int v48; // edx
  __int64 v49; // r9
  __int64 *v50; // r11
  int v51; // r11d
  __int64 *v52; // r10
  __int64 v53; // [rsp-68h] [rbp-68h]
  __int64 v54; // [rsp-60h] [rbp-60h]
  __int64 v55; // [rsp-58h] [rbp-58h]
  __int64 v56; // [rsp-50h] [rbp-50h]
  _QWORD *v57; // [rsp-50h] [rbp-50h]
  int v58; // [rsp-50h] [rbp-50h]
  int v59; // [rsp-50h] [rbp-50h]
  _QWORD *v60; // [rsp-50h] [rbp-50h]
  int v61; // [rsp-50h] [rbp-50h]
  int v62; // [rsp-50h] [rbp-50h]
  __int64 *v63; // [rsp-40h] [rbp-40h] BYREF

  if ( src != a2 )
  {
    v5 = src + 2;
    if ( &src[2] != a2 )
    {
      while ( 1 )
      {
        v21 = *(_DWORD *)(a3 + 24);
        v22 = (_QWORD *)v5[1].m128i_i64[0];
        v23 = (__int64 *)src[1].m128i_i64[0];
        if ( v21 )
        {
          v7 = v21 - 1;
          v8 = *(_QWORD *)(a3 + 8);
          v9 = (v21 - 1) & (((unsigned int)*v22 >> 9) ^ ((unsigned int)*v22 >> 4));
          v10 = (__int64 *)(v8 + 16LL * v9);
          v11 = *v10;
          if ( *v22 == *v10 )
          {
LABEL_5:
            v12 = *((_DWORD *)v10 + 2);
            goto LABEL_6;
          }
          v59 = 1;
          v30 = 0;
          while ( v11 != -8 )
          {
            if ( v11 == -16 && !v30 )
              v30 = v10;
            v9 = v7 & (v59 + v9);
            v10 = (__int64 *)(v8 + 16LL * v9);
            v11 = *v10;
            if ( *v22 == *v10 )
              goto LABEL_5;
            ++v59;
          }
          if ( !v30 )
            v30 = v10;
          v43 = *(_DWORD *)(a3 + 16);
          ++*(_QWORD *)a3;
          v29 = v43 + 1;
          if ( 4 * v29 < 3 * v21 )
          {
            if ( v21 - *(_DWORD *)(a3 + 20) - v29 > v21 >> 3 )
              goto LABEL_15;
            v60 = v22;
            sub_1468630(a3, v21);
            v44 = *(_DWORD *)(a3 + 24);
            if ( !v44 )
            {
LABEL_80:
              ++*(_DWORD *)(a3 + 16);
              BUG();
            }
            v22 = v60;
            v45 = v44 - 1;
            v46 = *(_QWORD *)(a3 + 8);
            v47 = *v60;
            v48 = v45 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
            v29 = *(_DWORD *)(a3 + 16) + 1;
            v30 = (__int64 *)(v46 + 16LL * v48);
            v49 = *v30;
            if ( *v60 == *v30 )
              goto LABEL_15;
            v61 = 1;
            v50 = 0;
            while ( v49 != -8 )
            {
              if ( v49 == -16 && !v50 )
                v50 = v30;
              v48 = v45 & (v61 + v48);
              v30 = (__int64 *)(v46 + 16LL * v48);
              v49 = *v30;
              if ( v47 == *v30 )
                goto LABEL_15;
              ++v61;
            }
            goto LABEL_57;
          }
        }
        else
        {
          ++*(_QWORD *)a3;
        }
        v57 = v22;
        sub_1468630(a3, 2 * v21);
        v24 = *(_DWORD *)(a3 + 24);
        if ( !v24 )
          goto LABEL_80;
        v22 = v57;
        v25 = v24 - 1;
        v26 = *(_QWORD *)(a3 + 8);
        v27 = *v57;
        v28 = v25 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v29 = *(_DWORD *)(a3 + 16) + 1;
        v30 = (__int64 *)(v26 + 16LL * v28);
        v31 = *v30;
        if ( *v57 == *v30 )
          goto LABEL_15;
        v62 = 1;
        v50 = 0;
        while ( v31 != -8 )
        {
          if ( !v50 && v31 == -16 )
            v50 = v30;
          v28 = v25 & (v62 + v28);
          v30 = (__int64 *)(v26 + 16LL * v28);
          v31 = *v30;
          if ( v27 == *v30 )
            goto LABEL_15;
          ++v62;
        }
LABEL_57:
        if ( v50 )
          v30 = v50;
LABEL_15:
        *(_DWORD *)(a3 + 16) = v29;
        if ( *v30 != -8 )
          --*(_DWORD *)(a3 + 20);
        v32 = *v22;
        *((_DWORD *)v30 + 2) = 0;
        *v30 = v32;
        v21 = *(_DWORD *)(a3 + 24);
        if ( !v21 )
        {
          ++*(_QWORD *)a3;
          goto LABEL_19;
        }
        v8 = *(_QWORD *)(a3 + 8);
        v7 = v21 - 1;
        v12 = 0;
LABEL_6:
        v13 = v7 & (((unsigned int)*v23 >> 9) ^ ((unsigned int)*v23 >> 4));
        v14 = (__int64 *)(v8 + 16LL * v13);
        v15 = *v14;
        if ( *v23 != *v14 )
        {
          v58 = 1;
          v38 = 0;
          while ( v15 != -8 )
          {
            if ( v15 == -16 && !v38 )
              v38 = v14;
            v13 = v7 & (v58 + v13);
            v14 = (__int64 *)(v8 + 16LL * v13);
            v15 = *v14;
            if ( *v23 == *v14 )
              goto LABEL_7;
            ++v58;
          }
          if ( !v38 )
            v38 = v14;
          v42 = *(_DWORD *)(a3 + 16);
          ++*(_QWORD *)a3;
          v37 = v42 + 1;
          if ( 4 * v37 >= 3 * v21 )
          {
LABEL_19:
            sub_1468630(a3, 2 * v21);
            v33 = *(_DWORD *)(a3 + 24);
            if ( !v33 )
            {
              ++*(_DWORD *)(a3 + 16);
              BUG();
            }
            v34 = v33 - 1;
            v35 = *(_QWORD *)(a3 + 8);
            v36 = v34 & (((unsigned int)*v23 >> 9) ^ ((unsigned int)*v23 >> 4));
            v37 = *(_DWORD *)(a3 + 16) + 1;
            v38 = (__int64 *)(v35 + 16LL * v36);
            v39 = *v38;
            if ( *v38 != *v23 )
            {
              v51 = 1;
              v52 = 0;
              while ( v39 != -8 )
              {
                if ( !v52 && v39 == -16 )
                  v52 = v38;
                v36 = v34 & (v51 + v36);
                v38 = (__int64 *)(v35 + 16LL * v36);
                v39 = *v38;
                if ( *v23 == *v38 )
                  goto LABEL_21;
                ++v51;
              }
              if ( v52 )
                v38 = v52;
            }
          }
          else if ( v21 - (v37 + *(_DWORD *)(a3 + 20)) <= v21 >> 3 )
          {
            sub_1468630(a3, v21);
            sub_145FB10(a3, v23, &v63);
            v38 = v63;
            v37 = *(_DWORD *)(a3 + 16) + 1;
          }
LABEL_21:
          *(_DWORD *)(a3 + 16) = v37;
          if ( *v38 != -8 )
            --*(_DWORD *)(a3 + 20);
          v40 = *v23;
          *((_DWORD *)v38 + 2) = 0;
          v16 = v5 + 2;
          *v38 = v40;
          goto LABEL_24;
        }
LABEL_7:
        v16 = v5 + 2;
        if ( *((_DWORD *)v14 + 2) <= v12 )
        {
LABEL_24:
          v41 = v5;
          v5 = v16;
          sub_1C574B0(v41, a3);
          if ( a2 == v16 )
            return;
        }
        else
        {
          v17 = v5->m128i_i64[0];
          v18 = v5->m128i_i64[1];
          v19 = v5[1].m128i_i64[0];
          v20 = v5[1].m128i_i64[1];
          if ( src != v5 )
          {
            v53 = v5[1].m128i_i64[1];
            v54 = v5->m128i_i64[1];
            v55 = v5->m128i_i64[0];
            v56 = v5[1].m128i_i64[0];
            memmove(&src[2], src, (char *)v5 - (char *)src);
            v20 = v53;
            v18 = v54;
            v17 = v55;
            v19 = v56;
          }
          src->m128i_i64[0] = v17;
          v5 += 2;
          src->m128i_i64[1] = v18;
          src[1].m128i_i64[0] = v19;
          src[1].m128i_i64[1] = v20;
          if ( a2 == v16 )
            return;
        }
      }
    }
  }
}
