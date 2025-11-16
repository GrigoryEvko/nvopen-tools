// Function: sub_24FEA80
// Address: 0x24fea80
//
__int64 __fastcall sub_24FEA80(__int64 a1, __int64 a2, __int64 *a3, const __m128i *a4)
{
  __int64 result; // rax
  __int64 v8; // rdi
  char v9; // cl
  __int64 v10; // r9
  __int64 v11; // rsi
  __int64 v12; // r15
  int v13; // r14d
  __int64 v14; // r10
  __int64 *v15; // rdx
  __int64 v16; // r11
  __int64 v17; // r15
  unsigned int v18; // edx
  __int64 *v19; // r8
  int v20; // esi
  unsigned int v21; // edi
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rdi
  int v26; // edx
  unsigned int v27; // esi
  __int64 v28; // r9
  __int64 v29; // rdi
  int v30; // ecx
  unsigned int v31; // esi
  __int64 v32; // r9
  int v33; // r11d
  __int64 *v34; // r10
  int v35; // edx
  int v36; // ecx
  int v37; // r11d
  int v38; // [rsp+8h] [rbp-38h]
  __int64 v39; // [rsp+8h] [rbp-38h]
  __int64 v40; // [rsp+8h] [rbp-38h]

  result = a1;
  v8 = *(_QWORD *)a2;
  v9 = *(_BYTE *)(a2 + 8) & 1;
  if ( v9 )
  {
    v10 = a2 + 16;
    v11 = *a3;
    v12 = 128;
    v13 = 3;
    v14 = *a3 & 3;
    v15 = (__int64 *)(v10 + 32LL * (*(_DWORD *)a3 & 3));
    v16 = *v15;
    if ( v11 == *v15 )
      goto LABEL_3;
  }
  else
  {
    v17 = *(unsigned int *)(a2 + 24);
    if ( !(_DWORD)v17 )
    {
      v18 = *(_DWORD *)(a2 + 8);
      v19 = 0;
      *(_QWORD *)a2 = v8 + 1;
      v20 = (v18 >> 1) + 1;
      goto LABEL_8;
    }
    v10 = *(_QWORD *)(a2 + 16);
    v11 = *a3;
    v13 = v17 - 1;
    LODWORD(v14) = (v17 - 1) & (37 * *a3);
    v15 = (__int64 *)(v10 + 32LL * (unsigned int)v14);
    v16 = *v15;
    if ( v11 == *v15 )
      goto LABEL_6;
  }
  v38 = 1;
  v19 = 0;
  while ( 1 )
  {
    if ( v16 == 0x7FFFFFFFFFFFFFFFLL )
    {
      if ( !v19 )
        v19 = v15;
      v18 = *(_DWORD *)(a2 + 8);
      *(_QWORD *)a2 = v8 + 1;
      v20 = (v18 >> 1) + 1;
      if ( v9 )
      {
        v21 = 12;
        LODWORD(v17) = 4;
LABEL_9:
        if ( 4 * v20 < v21 )
        {
          if ( (int)v17 - *(_DWORD *)(a2 + 12) - v20 > (unsigned int)v17 >> 3 )
          {
LABEL_11:
            *(_DWORD *)(a2 + 8) = (2 * (v18 >> 1) + 2) | v18 & 1;
            if ( *v19 != 0x7FFFFFFFFFFFFFFFLL )
              --*(_DWORD *)(a2 + 12);
            *v19 = *a3;
            *(__m128i *)(v19 + 1) = _mm_loadu_si128(a4);
            v19[3] = a4[1].m128i_i64[0];
            if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
            {
              v22 = a2 + 16;
              v23 = 128;
            }
            else
            {
              v22 = *(_QWORD *)(a2 + 16);
              v23 = 32LL * *(unsigned int *)(a2 + 24);
            }
            v24 = *(_QWORD *)a2;
            *(_QWORD *)result = a2;
            *(_QWORD *)(result + 16) = v19;
            *(_QWORD *)(result + 8) = v24;
            *(_QWORD *)(result + 24) = v23 + v22;
            *(_BYTE *)(result + 32) = 1;
            return result;
          }
          v40 = result;
          sub_24FE5C0(a2, v17);
          result = v40;
          if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
          {
            v29 = a2 + 16;
            v30 = 3;
            goto LABEL_29;
          }
          v36 = *(_DWORD *)(a2 + 24);
          v29 = *(_QWORD *)(a2 + 16);
          if ( v36 )
          {
            v30 = v36 - 1;
LABEL_29:
            v31 = v30 & (37 * *a3);
            v19 = (__int64 *)(v29 + 32LL * v31);
            v32 = *v19;
            if ( *v19 != *a3 )
            {
              v33 = 1;
              v34 = 0;
              while ( v32 != 0x7FFFFFFFFFFFFFFFLL )
              {
                if ( v32 == 0x7FFFFFFFFFFFFFFELL && !v34 )
                  v34 = v19;
                v31 = v30 & (v33 + v31);
                v19 = (__int64 *)(v29 + 32LL * v31);
                v32 = *v19;
                if ( *a3 == *v19 )
                  goto LABEL_26;
                ++v33;
              }
LABEL_32:
              if ( v34 )
                v19 = v34;
              goto LABEL_26;
            }
            goto LABEL_26;
          }
LABEL_59:
          *(_DWORD *)(a2 + 8) = (2 * (*(_DWORD *)(a2 + 8) >> 1) + 2) | *(_DWORD *)(a2 + 8) & 1;
          BUG();
        }
        v39 = result;
        sub_24FE5C0(a2, 2 * v17);
        result = v39;
        if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
        {
          v25 = a2 + 16;
          v26 = 3;
        }
        else
        {
          v35 = *(_DWORD *)(a2 + 24);
          v25 = *(_QWORD *)(a2 + 16);
          if ( !v35 )
            goto LABEL_59;
          v26 = v35 - 1;
        }
        v27 = v26 & (37 * *a3);
        v19 = (__int64 *)(v25 + 32LL * v27);
        v28 = *v19;
        if ( *a3 != *v19 )
        {
          v37 = 1;
          v34 = 0;
          while ( v28 != 0x7FFFFFFFFFFFFFFFLL )
          {
            if ( !v34 && v28 == 0x7FFFFFFFFFFFFFFELL )
              v34 = v19;
            v27 = v26 & (v37 + v27);
            v19 = (__int64 *)(v25 + 32LL * v27);
            v28 = *v19;
            if ( *a3 == *v19 )
              goto LABEL_26;
            ++v37;
          }
          goto LABEL_32;
        }
LABEL_26:
        v18 = *(_DWORD *)(a2 + 8);
        goto LABEL_11;
      }
      LODWORD(v17) = *(_DWORD *)(a2 + 24);
LABEL_8:
      v21 = 3 * v17;
      goto LABEL_9;
    }
    if ( v16 == 0x7FFFFFFFFFFFFFFELL && !v19 )
      v19 = v15;
    LODWORD(v14) = v13 & (v38 + v14);
    v15 = (__int64 *)(v10 + 32LL * (unsigned int)v14);
    v16 = *v15;
    if ( *v15 == v11 )
      break;
    ++v38;
  }
  if ( v9 )
  {
    v12 = 128;
  }
  else
  {
    v17 = *(unsigned int *)(a2 + 24);
LABEL_6:
    v12 = 32 * v17;
  }
LABEL_3:
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 8) = v8;
  *(_QWORD *)(result + 16) = v15;
  *(_QWORD *)(result + 24) = v10 + v12;
  *(_BYTE *)(result + 32) = 0;
  return result;
}
