// Function: sub_2C97780
// Address: 0x2c97780
//
void __fastcall sub_2C97780(__m128i *src, const __m128i *a2, __int64 a3)
{
  const __m128i *v6; // rbx
  unsigned int v7; // r10d
  __int64 v8; // r9
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r11
  unsigned int v12; // r15d
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r11
  __int64 v16; // r9
  __int64 v17; // r8
  __int64 v18; // rcx
  __int64 v19; // r15
  unsigned int v20; // esi
  __int64 *v21; // r15
  _QWORD *v22; // r8
  int v23; // ecx
  int v24; // ecx
  __int64 v25; // r10
  unsigned int v26; // edx
  int v27; // eax
  __int64 *v28; // rdi
  __int64 v29; // r9
  __int64 v30; // rax
  int v31; // ecx
  int v32; // ecx
  __int64 v33; // r10
  __int64 v34; // rsi
  unsigned int v35; // edx
  int v36; // eax
  __int64 *v37; // rdi
  __int64 v38; // r9
  __int64 v39; // rax
  const __m128i *v40; // rdi
  int v41; // eax
  int v42; // edx
  int v43; // edx
  __int64 v44; // r10
  __int64 *v45; // r11
  int v46; // r15d
  __int64 v47; // rsi
  unsigned int v48; // ecx
  __int64 v49; // r9
  int v50; // eax
  int v51; // ecx
  int v52; // ecx
  __int64 v53; // r10
  unsigned int v54; // edx
  __int64 v55; // r9
  __int64 *v56; // r11
  int v57; // r15d
  __int64 v58; // [rsp-50h] [rbp-50h]
  __int64 v59; // [rsp-48h] [rbp-48h]
  __int64 v60; // [rsp-40h] [rbp-40h]
  _QWORD *v61; // [rsp-40h] [rbp-40h]
  _QWORD *v62; // [rsp-40h] [rbp-40h]
  int v63; // [rsp-40h] [rbp-40h]
  _QWORD *v64; // [rsp-40h] [rbp-40h]
  int v65; // [rsp-40h] [rbp-40h]
  _QWORD *v66; // [rsp-40h] [rbp-40h]
  int v67; // [rsp-40h] [rbp-40h]
  int v68; // [rsp-40h] [rbp-40h]

  if ( src != a2 )
  {
    v6 = src + 2;
    if ( &src[2] != a2 )
    {
      while ( 1 )
      {
        v20 = *(_DWORD *)(a3 + 24);
        v21 = (__int64 *)v6[1].m128i_i64[0];
        v22 = (_QWORD *)src[1].m128i_i64[0];
        if ( v20 )
        {
          v7 = v20 - 1;
          v8 = *(_QWORD *)(a3 + 8);
          v9 = (v20 - 1) & (((unsigned int)*v21 >> 9) ^ ((unsigned int)*v21 >> 4));
          v10 = (__int64 *)(v8 + 16LL * v9);
          v11 = *v10;
          if ( *v21 == *v10 )
          {
LABEL_5:
            v12 = *((_DWORD *)v10 + 2);
            goto LABEL_6;
          }
          v65 = 1;
          v28 = 0;
          while ( v11 != -4096 )
          {
            if ( !v28 && v11 == -8192 )
              v28 = v10;
            v9 = v7 & (v65 + v9);
            v10 = (__int64 *)(v8 + 16LL * v9);
            v11 = *v10;
            if ( *v21 == *v10 )
              goto LABEL_5;
            ++v65;
          }
          if ( !v28 )
            v28 = v10;
          v50 = *(_DWORD *)(a3 + 16);
          ++*(_QWORD *)a3;
          v27 = v50 + 1;
          if ( 4 * v27 < 3 * v20 )
          {
            if ( v20 - *(_DWORD *)(a3 + 20) - v27 > v20 >> 3 )
              goto LABEL_15;
            v66 = v22;
            sub_2C96F50(a3, v20);
            v51 = *(_DWORD *)(a3 + 24);
            if ( !v51 )
            {
LABEL_87:
              ++*(_DWORD *)(a3 + 16);
              BUG();
            }
            v52 = v51 - 1;
            v53 = *(_QWORD *)(a3 + 8);
            v22 = v66;
            v54 = v52 & (((unsigned int)*v21 >> 9) ^ ((unsigned int)*v21 >> 4));
            v27 = *(_DWORD *)(a3 + 16) + 1;
            v28 = (__int64 *)(v53 + 16LL * v54);
            v55 = *v28;
            if ( *v21 == *v28 )
              goto LABEL_15;
            v67 = 1;
            v56 = 0;
            while ( v55 != -4096 )
            {
              if ( v55 == -8192 && !v56 )
                v56 = v28;
              v54 = v52 & (v67 + v54);
              v28 = (__int64 *)(v53 + 16LL * v54);
              v55 = *v28;
              if ( *v21 == *v28 )
                goto LABEL_15;
              ++v67;
            }
            goto LABEL_64;
          }
        }
        else
        {
          ++*(_QWORD *)a3;
        }
        v61 = v22;
        sub_2C96F50(a3, 2 * v20);
        v23 = *(_DWORD *)(a3 + 24);
        if ( !v23 )
          goto LABEL_87;
        v24 = v23 - 1;
        v25 = *(_QWORD *)(a3 + 8);
        v22 = v61;
        v26 = v24 & (((unsigned int)*v21 >> 9) ^ ((unsigned int)*v21 >> 4));
        v27 = *(_DWORD *)(a3 + 16) + 1;
        v28 = (__int64 *)(v25 + 16LL * v26);
        v29 = *v28;
        if ( *v21 == *v28 )
          goto LABEL_15;
        v68 = 1;
        v56 = 0;
        while ( v29 != -4096 )
        {
          if ( !v56 && v29 == -8192 )
            v56 = v28;
          v26 = v24 & (v68 + v26);
          v28 = (__int64 *)(v25 + 16LL * v26);
          v29 = *v28;
          if ( *v21 == *v28 )
            goto LABEL_15;
          ++v68;
        }
LABEL_64:
        if ( v56 )
          v28 = v56;
LABEL_15:
        *(_DWORD *)(a3 + 16) = v27;
        if ( *v28 != -4096 )
          --*(_DWORD *)(a3 + 20);
        v30 = *v21;
        *((_DWORD *)v28 + 2) = 0;
        *v28 = v30;
        v20 = *(_DWORD *)(a3 + 24);
        if ( !v20 )
        {
          ++*(_QWORD *)a3;
          goto LABEL_19;
        }
        v8 = *(_QWORD *)(a3 + 8);
        v7 = v20 - 1;
        v12 = 0;
LABEL_6:
        v13 = v7 & (((unsigned int)*v22 >> 9) ^ ((unsigned int)*v22 >> 4));
        v14 = (__int64 *)(v8 + 16LL * v13);
        v15 = *v14;
        if ( *v22 != *v14 )
        {
          v63 = 1;
          v37 = 0;
          while ( v15 != -4096 )
          {
            if ( !v37 && v15 == -8192 )
              v37 = v14;
            v13 = v7 & (v63 + v13);
            v14 = (__int64 *)(v8 + 16LL * v13);
            v15 = *v14;
            if ( *v22 == *v14 )
              goto LABEL_7;
            ++v63;
          }
          if ( !v37 )
            v37 = v14;
          v41 = *(_DWORD *)(a3 + 16);
          ++*(_QWORD *)a3;
          v36 = v41 + 1;
          if ( 4 * v36 >= 3 * v20 )
          {
LABEL_19:
            v62 = v22;
            sub_2C96F50(a3, 2 * v20);
            v31 = *(_DWORD *)(a3 + 24);
            if ( !v31 )
              goto LABEL_88;
            v22 = v62;
            v32 = v31 - 1;
            v33 = *(_QWORD *)(a3 + 8);
            v34 = *v62;
            v35 = v32 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
            v36 = *(_DWORD *)(a3 + 16) + 1;
            v37 = (__int64 *)(v33 + 16LL * v35);
            v38 = *v37;
            if ( *v37 != *v62 )
            {
              v57 = 1;
              v45 = 0;
              while ( v38 != -4096 )
              {
                if ( !v45 && v38 == -8192 )
                  v45 = v37;
                v35 = v32 & (v57 + v35);
                v37 = (__int64 *)(v33 + 16LL * v35);
                v38 = *v37;
                if ( v34 == *v37 )
                  goto LABEL_21;
                ++v57;
              }
              goto LABEL_59;
            }
          }
          else if ( v20 - (v36 + *(_DWORD *)(a3 + 20)) <= v20 >> 3 )
          {
            v64 = v22;
            sub_2C96F50(a3, v20);
            v42 = *(_DWORD *)(a3 + 24);
            if ( !v42 )
            {
LABEL_88:
              ++*(_DWORD *)(a3 + 16);
              BUG();
            }
            v22 = v64;
            v43 = v42 - 1;
            v44 = *(_QWORD *)(a3 + 8);
            v45 = 0;
            v46 = 1;
            v47 = *v64;
            v48 = v43 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
            v36 = *(_DWORD *)(a3 + 16) + 1;
            v37 = (__int64 *)(v44 + 16LL * v48);
            v49 = *v37;
            if ( *v37 != *v64 )
            {
              while ( v49 != -4096 )
              {
                if ( !v45 && v49 == -8192 )
                  v45 = v37;
                v48 = v43 & (v46 + v48);
                v37 = (__int64 *)(v44 + 16LL * v48);
                v49 = *v37;
                if ( v47 == *v37 )
                  goto LABEL_21;
                ++v46;
              }
LABEL_59:
              if ( v45 )
                v37 = v45;
            }
          }
LABEL_21:
          *(_DWORD *)(a3 + 16) = v36;
          if ( *v37 != -4096 )
            --*(_DWORD *)(a3 + 20);
          v39 = *v22;
          *((_DWORD *)v37 + 2) = 0;
          *v37 = v39;
          goto LABEL_24;
        }
LABEL_7:
        if ( v12 >= *((_DWORD *)v14 + 2) )
        {
LABEL_24:
          v40 = v6;
          v6 += 2;
          sub_2C97270(v40, a3);
          if ( a2 == v6 )
            return;
        }
        else
        {
          v16 = v6->m128i_i64[0];
          v17 = v6->m128i_i64[1];
          v18 = v6[1].m128i_i64[0];
          v19 = v6[1].m128i_i64[1];
          if ( src != v6 )
          {
            v58 = v6->m128i_i64[1];
            v59 = v6->m128i_i64[0];
            v60 = v6[1].m128i_i64[0];
            memmove(&src[2], src, (char *)v6 - (char *)src);
            v17 = v58;
            v16 = v59;
            v18 = v60;
          }
          v6 += 2;
          src->m128i_i64[0] = v16;
          src->m128i_i64[1] = v17;
          src[1].m128i_i64[0] = v18;
          src[1].m128i_i64[1] = v19;
          if ( a2 == v6 )
            return;
        }
      }
    }
  }
}
