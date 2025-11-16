// Function: sub_2ACD0B0
// Address: 0x2acd0b0
//
void __fastcall sub_2ACD0B0(__m128i *a1, __m128i *a2, __int64 a3)
{
  __m128i *v5; // r12
  unsigned int v7; // esi
  unsigned int v8; // r8d
  __int64 v9; // rcx
  __int64 *v10; // r9
  int v11; // r11d
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r10
  unsigned int v15; // r15d
  int v16; // r11d
  __int64 *v17; // r9
  unsigned int v18; // edx
  __int64 *v19; // rax
  __int64 v20; // r10
  int v21; // r11d
  __int64 *v22; // r8
  unsigned int v23; // edx
  __int64 *v24; // rax
  __int64 v25; // r9
  unsigned int v26; // r11d
  int v27; // r15d
  __int64 *v28; // r8
  unsigned int v29; // edx
  __int64 *v30; // rax
  __int64 v31; // r9
  __m128i v32; // xmm0
  __int64 v33; // rdi
  __m128i *v34; // rsi
  __int64 v35; // r12
  __int64 v36; // rdx
  __m128i *v37; // rax
  __int64 v38; // rcx
  __int32 v39; // eax
  int v40; // eax
  int v41; // edx
  __int64 v42; // rax
  int v43; // eax
  __int64 v44; // rax
  int v45; // eax
  int v46; // eax
  int v47; // eax
  __int64 v48; // rax
  int v49; // eax
  int v50; // eax
  __int64 v51; // rax
  _OWORD v52[4]; // [rsp-48h] [rbp-48h] BYREF

  if ( a1 != a2 )
  {
    v5 = a1 + 1;
    if ( a2 != &a1[1] )
    {
      while ( 1 )
      {
        v7 = *(_DWORD *)(a3 + 24);
        if ( !v7 )
          break;
        v8 = v7 - 1;
        v9 = *(_QWORD *)(a3 + 8);
        v10 = 0;
        v11 = 1;
        v12 = (v7 - 1) & (((unsigned int)v5->m128i_i64[0] >> 9) ^ ((unsigned int)v5->m128i_i64[0] >> 4));
        v13 = (__int64 *)(v9 + 16LL * v12);
        v14 = *v13;
        if ( v5->m128i_i64[0] == *v13 )
        {
LABEL_5:
          v15 = *((_DWORD *)v13 + 2);
          goto LABEL_6;
        }
        while ( v14 != -4096 )
        {
          if ( v14 == -8192 && !v10 )
            v10 = v13;
          v12 = v8 & (v11 + v12);
          v13 = (__int64 *)(v9 + 16LL * v12);
          v14 = *v13;
          if ( v5->m128i_i64[0] == *v13 )
            goto LABEL_5;
          ++v11;
        }
        if ( !v10 )
          v10 = v13;
        v40 = *(_DWORD *)(a3 + 16);
        ++*(_QWORD *)a3;
        v41 = v40 + 1;
        *(_QWORD *)&v52[0] = v10;
        if ( 4 * (v40 + 1) >= 3 * v7 )
          goto LABEL_86;
        if ( v7 - *(_DWORD *)(a3 + 20) - v41 <= v7 >> 3 )
          goto LABEL_87;
LABEL_32:
        *(_DWORD *)(a3 + 16) = v41;
        if ( *v10 != -4096 )
          --*(_DWORD *)(a3 + 20);
        v42 = v5->m128i_i64[0];
        *((_DWORD *)v10 + 2) = 0;
        *v10 = v42;
        v7 = *(_DWORD *)(a3 + 24);
        v15 = v7;
        if ( !v7 )
        {
          ++*(_QWORD *)a3;
          v7 = 0;
          *(_QWORD *)&v52[0] = 0;
          goto LABEL_36;
        }
        v9 = *(_QWORD *)(a3 + 8);
        v8 = v7 - 1;
        v15 = 0;
LABEL_6:
        v16 = 1;
        v17 = 0;
        v18 = v8 & (((unsigned int)a1->m128i_i64[0] >> 9) ^ ((unsigned int)a1->m128i_i64[0] >> 4));
        v19 = (__int64 *)(v9 + 16LL * v18);
        v20 = *v19;
        if ( a1->m128i_i64[0] == *v19 )
        {
LABEL_7:
          if ( v15 == *((_DWORD *)v19 + 2) )
            goto LABEL_18;
          goto LABEL_8;
        }
        while ( v20 != -4096 )
        {
          if ( !v17 && v20 == -8192 )
            v17 = v19;
          v18 = v8 & (v16 + v18);
          v19 = (__int64 *)(v9 + 16LL * v18);
          v20 = *v19;
          if ( a1->m128i_i64[0] == *v19 )
            goto LABEL_7;
          ++v16;
        }
        if ( !v17 )
          v17 = v19;
        v46 = *(_DWORD *)(a3 + 16);
        ++*(_QWORD *)a3;
        v43 = v46 + 1;
        *(_QWORD *)&v52[0] = v17;
        if ( 4 * v43 < 3 * v7 )
        {
          if ( v7 - (v43 + *(_DWORD *)(a3 + 20)) > v7 >> 3 )
            goto LABEL_38;
          goto LABEL_37;
        }
LABEL_36:
        v7 *= 2;
LABEL_37:
        sub_2ACC850(a3, v7);
        sub_2AC1490(a3, a1->m128i_i64, v52);
        v17 = *(__int64 **)&v52[0];
        v43 = *(_DWORD *)(a3 + 16) + 1;
LABEL_38:
        *(_DWORD *)(a3 + 16) = v43;
        if ( *v17 != -4096 )
          --*(_DWORD *)(a3 + 20);
        v44 = a1->m128i_i64[0];
        *((_DWORD *)v17 + 2) = 0;
        *v17 = v44;
        if ( !v15 )
        {
LABEL_18:
          if ( (!v5->m128i_i8[12] || a1->m128i_i8[12]) && v5->m128i_i32[2] < (unsigned __int32)a1->m128i_i32[2] )
          {
LABEL_12:
            v32 = _mm_loadu_si128(v5);
            v33 = v5->m128i_i64[0];
            v34 = v5 + 1;
            v35 = (char *)v5 - (char *)a1;
            v52[0] = v32;
            v36 = v35 >> 4;
            if ( v35 > 0 )
            {
              v37 = v34;
              do
              {
                v38 = v37[-2].m128i_i64[0];
                --v37;
                v37->m128i_i64[0] = v38;
                v37->m128i_i32[2] = v37[-1].m128i_i32[2];
                v37->m128i_i8[12] = v37[-1].m128i_i8[12];
                --v36;
              }
              while ( v36 );
            }
            v39 = DWORD2(v52[0]);
            a1->m128i_i64[0] = v33;
            a1->m128i_i32[2] = v39;
            a1->m128i_i8[12] = BYTE12(v52[0]);
            goto LABEL_16;
          }
          goto LABEL_21;
        }
        v7 = *(_DWORD *)(a3 + 24);
        if ( !v7 )
        {
          ++*(_QWORD *)a3;
          *(_QWORD *)&v52[0] = 0;
          goto LABEL_43;
        }
        v9 = *(_QWORD *)(a3 + 8);
LABEL_8:
        v21 = 1;
        v22 = 0;
        v23 = (v7 - 1) & (((unsigned int)v5->m128i_i64[0] >> 9) ^ ((unsigned int)v5->m128i_i64[0] >> 4));
        v24 = (__int64 *)(v9 + 16LL * v23);
        v25 = *v24;
        if ( *v24 != v5->m128i_i64[0] )
        {
          while ( v25 != -4096 )
          {
            if ( v25 == -8192 && !v22 )
              v22 = v24;
            v23 = (v7 - 1) & (v21 + v23);
            v24 = (__int64 *)(v9 + 16LL * v23);
            v25 = *v24;
            if ( v5->m128i_i64[0] == *v24 )
              goto LABEL_9;
            ++v21;
          }
          if ( !v22 )
            v22 = v24;
          v47 = *(_DWORD *)(a3 + 16);
          ++*(_QWORD *)a3;
          v45 = v47 + 1;
          *(_QWORD *)&v52[0] = v22;
          if ( 4 * v45 >= 3 * v7 )
          {
LABEL_43:
            v7 *= 2;
          }
          else if ( v7 - (v45 + *(_DWORD *)(a3 + 20)) > v7 >> 3 )
          {
            goto LABEL_66;
          }
          sub_2ACC850(a3, v7);
          sub_2AC1490(a3, v5->m128i_i64, v52);
          v22 = *(__int64 **)&v52[0];
          v45 = *(_DWORD *)(a3 + 16) + 1;
LABEL_66:
          *(_DWORD *)(a3 + 16) = v45;
          if ( *v22 != -4096 )
            --*(_DWORD *)(a3 + 20);
          v48 = v5->m128i_i64[0];
          *((_DWORD *)v22 + 2) = 0;
          *v22 = v48;
          v7 = *(_DWORD *)(a3 + 24);
          if ( !v7 )
          {
            ++*(_QWORD *)a3;
            *(_QWORD *)&v52[0] = 0;
            goto LABEL_70;
          }
          v9 = *(_QWORD *)(a3 + 8);
          v26 = 0;
          goto LABEL_10;
        }
LABEL_9:
        v26 = *((_DWORD *)v24 + 2);
LABEL_10:
        v27 = 1;
        v28 = 0;
        v29 = (v7 - 1) & (((unsigned int)a1->m128i_i64[0] >> 9) ^ ((unsigned int)a1->m128i_i64[0] >> 4));
        v30 = (__int64 *)(v9 + 16LL * v29);
        v31 = *v30;
        if ( *v30 != a1->m128i_i64[0] )
        {
          while ( v31 != -4096 )
          {
            if ( v31 == -8192 && !v28 )
              v28 = v30;
            v29 = (v7 - 1) & (v27 + v29);
            v30 = (__int64 *)(v9 + 16LL * v29);
            v31 = *v30;
            if ( a1->m128i_i64[0] == *v30 )
              goto LABEL_11;
            ++v27;
          }
          if ( !v28 )
            v28 = v30;
          v50 = *(_DWORD *)(a3 + 16);
          ++*(_QWORD *)a3;
          v49 = v50 + 1;
          *(_QWORD *)&v52[0] = v28;
          if ( 4 * v49 >= 3 * v7 )
          {
LABEL_70:
            v7 *= 2;
          }
          else if ( v7 - (v49 + *(_DWORD *)(a3 + 20)) > v7 >> 3 )
          {
            goto LABEL_82;
          }
          sub_2ACC850(a3, v7);
          sub_2AC1490(a3, a1->m128i_i64, v52);
          v28 = *(__int64 **)&v52[0];
          v49 = *(_DWORD *)(a3 + 16) + 1;
LABEL_82:
          *(_DWORD *)(a3 + 16) = v49;
          if ( *v28 != -4096 )
            --*(_DWORD *)(a3 + 20);
          v51 = a1->m128i_i64[0];
          *((_DWORD *)v28 + 2) = 0;
          *v28 = v51;
          goto LABEL_21;
        }
LABEL_11:
        if ( v26 < *((_DWORD *)v30 + 2) )
          goto LABEL_12;
LABEL_21:
        sub_2ACCB40(v5, a3);
        v34 = v5 + 1;
LABEL_16:
        v5 = v34;
        if ( a2 == v34 )
          return;
      }
      ++*(_QWORD *)a3;
      *(_QWORD *)&v52[0] = 0;
LABEL_86:
      v7 *= 2;
LABEL_87:
      sub_2ACC850(a3, v7);
      sub_2AC1490(a3, v5->m128i_i64, v52);
      v10 = *(__int64 **)&v52[0];
      v41 = *(_DWORD *)(a3 + 16) + 1;
      goto LABEL_32;
    }
  }
}
