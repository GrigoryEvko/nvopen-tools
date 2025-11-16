// Function: sub_EBC400
// Address: 0xebc400
//
__int64 __fastcall sub_EBC400(__int64 a1, __int64 *a2, char a3)
{
  int *v5; // rcx
  int v6; // edx
  int v7; // ebx
  char v9; // al
  int v10; // eax
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdx
  __m128i v15; // xmm1
  unsigned int v16; // eax
  __int64 v17; // rax
  __int64 v18; // r15
  int v19; // edx
  unsigned __int64 v20; // r13
  __m128i v21; // xmm0
  bool v22; // cc
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  const char *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rax
  unsigned int v37; // ecx
  __int64 v38; // rax
  __int64 v39; // rsi
  __m128i v40; // xmm3
  unsigned int v41; // edx
  __int64 v42; // rdx
  __int64 v43; // r13
  int v44; // ecx
  unsigned __int64 v45; // rsi
  unsigned __int64 v46; // rdx
  __m128i v47; // xmm2
  __int64 v48; // rdi
  __int64 v49; // rcx
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rdi
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  unsigned __int64 v56; // [rsp+0h] [rbp-80h]
  unsigned __int8 v57; // [rsp+Fh] [rbp-71h]
  int v58; // [rsp+1Ch] [rbp-64h] BYREF
  __int64 v59; // [rsp+20h] [rbp-60h] BYREF
  __int64 v60; // [rsp+28h] [rbp-58h]
  __int64 v61; // [rsp+38h] [rbp-48h]
  unsigned int v62; // [rsp+40h] [rbp-40h]

  v5 = *(int **)(a1 + 48);
  if ( a3 )
  {
    v57 = 0;
    if ( *v5 != 9 )
    {
      v32 = sub_EABDC0(a1);
      v58 = 3;
      v33 = a2[1];
      v59 = v32;
      v60 = v34;
      if ( v33 == a2[2] )
      {
        sub_EA8AC0(a2, v33, &v58, &v59);
      }
      else
      {
        if ( v33 )
        {
          v35 = v59;
          v36 = v60;
          *(_DWORD *)v33 = 3;
          *(_DWORD *)(v33 + 32) = 64;
          *(_QWORD *)(v33 + 8) = v35;
          *(_QWORD *)(v33 + 16) = v36;
          *(_QWORD *)(v33 + 24) = 0;
          v33 = a2[1];
        }
        a2[1] = v33 + 40;
      }
      return 0;
    }
  }
  else
  {
    *(_BYTE *)(a1 + 152) = *(_BYTE *)(a1 + 868);
    v6 = *v5;
    v57 = v6 == 0 || v6 == 28;
    if ( v57 )
    {
LABEL_34:
      BYTE1(v62) = 1;
      v31 = "unexpected token in macro instantiation";
    }
    else
    {
      v7 = 0;
      while ( 1 )
      {
        if ( !v7 )
        {
          while ( v6 != 26 )
          {
            v9 = sub_ECE2A0(a1, 11);
            if ( *(_BYTE *)(a1 + 868)
              || (v37 = **(_DWORD **)(a1 + 48), v37 > 0x2D)
              || ((1LL << v37) & 0x3F9FF300F000LL) == 0 )
            {
              if ( v9 )
                goto LABEL_6;
              v5 = *(int **)(a1 + 48);
              goto LABEL_11;
            }
            v38 = sub_ECD7B0(a1);
            v39 = a2[1];
            if ( v39 == a2[2] )
            {
              sub_EA8D20(a2, v39, v38);
            }
            else
            {
              if ( v39 )
              {
                v40 = _mm_loadu_si128((const __m128i *)(v38 + 8));
                *(_DWORD *)v39 = *(_DWORD *)v38;
                *(__m128i *)(v39 + 8) = v40;
                v41 = *(_DWORD *)(v38 + 32);
                *(_DWORD *)(v39 + 32) = v41;
                if ( v41 > 0x40 )
                  sub_C43780(v39 + 24, (const void **)(v38 + 24));
                else
                  *(_QWORD *)(v39 + 24) = *(_QWORD *)(v38 + 24);
                v39 = a2[1];
              }
              a2[1] = v39 + 40;
            }
            v42 = *(unsigned int *)(a1 + 56);
            v43 = *(_QWORD *)(a1 + 48);
            v44 = *(_DWORD *)(a1 + 56);
            *(_BYTE *)(a1 + 155) = *(_DWORD *)v43 == 9;
            v45 = 40 * v42;
            v46 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v42 - 40) >> 3);
            if ( v45 > 0x28 )
            {
              do
              {
                v47 = _mm_loadu_si128((const __m128i *)(v43 + 48));
                v22 = *(_DWORD *)(v43 + 32) <= 0x40u;
                *(_DWORD *)v43 = *(_DWORD *)(v43 + 40);
                *(__m128i *)(v43 + 8) = v47;
                if ( !v22 )
                {
                  v48 = *(_QWORD *)(v43 + 24);
                  if ( v48 )
                  {
                    v56 = v46;
                    j_j___libc_free_0_0(v48);
                    v46 = v56;
                  }
                }
                v49 = *(_QWORD *)(v43 + 64);
                v43 += 40;
                *(_QWORD *)(v43 - 16) = v49;
                LODWORD(v49) = *(_DWORD *)(v43 + 32);
                *(_DWORD *)(v43 + 32) = 0;
                *(_DWORD *)(v43 - 8) = v49;
                --v46;
              }
              while ( v46 );
              v44 = *(_DWORD *)(a1 + 56);
              v43 = *(_QWORD *)(a1 + 48);
            }
            v50 = (unsigned int)(v44 - 1);
            *(_DWORD *)(a1 + 56) = v50;
            v51 = v43 + 40 * v50;
            if ( *(_DWORD *)(v51 + 32) > 0x40u )
            {
              v52 = *(_QWORD *)(v51 + 24);
              if ( v52 )
                j_j___libc_free_0_0(v52);
            }
            if ( !*(_DWORD *)(a1 + 56) )
            {
              sub_1097F60(&v59, a1 + 40);
              sub_EAA0A0(a1 + 48, *(_QWORD *)(a1 + 48), (unsigned __int64)&v59, v53, v54, v55);
              if ( v62 > 0x40 )
              {
                if ( v61 )
                  j_j___libc_free_0_0(v61);
              }
            }
            sub_ECE2A0(a1, 11);
            v6 = **(_DWORD **)(a1 + 48);
            if ( !v6 || v6 == 28 )
              goto LABEL_34;
          }
          goto LABEL_6;
        }
LABEL_11:
        v10 = *v5;
        if ( *v5 == 9 )
          break;
        if ( v10 == 17 )
        {
          ++v7;
        }
        else
        {
          v11 = v7;
          v7 = (v7 == 0) + v7 - 1;
          if ( v10 != 18 )
            v7 = v11;
        }
        v12 = sub_ECD7B0(a1);
        v13 = a2[1];
        v14 = v12;
        if ( v13 == a2[2] )
        {
          sub_EA8D20(a2, v13, v12);
        }
        else
        {
          if ( v13 )
          {
            v15 = _mm_loadu_si128((const __m128i *)(v12 + 8));
            *(_DWORD *)v13 = *(_DWORD *)v12;
            *(__m128i *)(v13 + 8) = v15;
            v16 = *(_DWORD *)(v12 + 32);
            *(_DWORD *)(v13 + 32) = v16;
            if ( v16 > 0x40 )
              sub_C43780(v13 + 24, (const void **)(v14 + 24));
            else
              *(_QWORD *)(v13 + 24) = *(_QWORD *)(v14 + 24);
            v13 = a2[1];
          }
          a2[1] = v13 + 40;
        }
        v17 = *(unsigned int *)(a1 + 56);
        v18 = *(_QWORD *)(a1 + 48);
        v19 = *(_DWORD *)(a1 + 56);
        *(_BYTE *)(a1 + 155) = *(_DWORD *)v18 == 9;
        v20 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v17 - 40) >> 3);
        if ( (unsigned __int64)(40 * v17) > 0x28 )
        {
          do
          {
            v21 = _mm_loadu_si128((const __m128i *)(v18 + 48));
            v22 = *(_DWORD *)(v18 + 32) <= 0x40u;
            *(_DWORD *)v18 = *(_DWORD *)(v18 + 40);
            *(__m128i *)(v18 + 8) = v21;
            if ( !v22 )
            {
              v23 = *(_QWORD *)(v18 + 24);
              if ( v23 )
                j_j___libc_free_0_0(v23);
            }
            v24 = *(_QWORD *)(v18 + 64);
            v18 += 40;
            *(_QWORD *)(v18 - 16) = v24;
            LODWORD(v24) = *(_DWORD *)(v18 + 32);
            *(_DWORD *)(v18 + 32) = 0;
            *(_DWORD *)(v18 - 8) = v24;
            --v20;
          }
          while ( v20 );
          v19 = *(_DWORD *)(a1 + 56);
          v18 = *(_QWORD *)(a1 + 48);
        }
        v25 = (unsigned int)(v19 - 1);
        *(_DWORD *)(a1 + 56) = v25;
        v26 = v18 + 40 * v25;
        if ( *(_DWORD *)(v26 + 32) > 0x40u )
        {
          v27 = *(_QWORD *)(v26 + 24);
          if ( v27 )
            j_j___libc_free_0_0(v27);
        }
        if ( !*(_DWORD *)(a1 + 56) )
        {
          sub_1097F60(&v59, a1 + 40);
          sub_EAA0A0(a1 + 48, *(_QWORD *)(a1 + 48), (unsigned __int64)&v59, v28, v29, v30);
          if ( v62 > 0x40 )
          {
            if ( v61 )
              j_j___libc_free_0_0(v61);
          }
        }
        v5 = *(int **)(a1 + 48);
        v6 = *v5;
        if ( *v5 == 28 || !v6 )
          goto LABEL_34;
      }
      if ( !v7 )
        goto LABEL_6;
      BYTE1(v62) = 1;
      v31 = "unbalanced parentheses in macro argument";
    }
    v59 = (__int64)v31;
    LOBYTE(v62) = 3;
    v57 = sub_ECE0E0(a1, &v59, 0, 0);
LABEL_6:
    *(_BYTE *)(a1 + 152) = 1;
  }
  return v57;
}
