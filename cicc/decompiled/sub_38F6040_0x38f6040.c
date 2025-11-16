// Function: sub_38F6040
// Address: 0x38f6040
//
__int64 __fastcall sub_38F6040(__int64 a1, unsigned __int64 *a2, char a3, __int64 a4, __int64 a5, __int64 a6)
{
  _DWORD *v8; // rsi
  int v9; // ecx
  _DWORD *v10; // rdx
  int v11; // ebx
  int v13; // eax
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdx
  __m128i v18; // xmm1
  unsigned int v19; // eax
  _DWORD *v20; // rdi
  __int64 v21; // rax
  _DWORD *v22; // r13
  int v23; // ecx
  unsigned __int64 v24; // r14
  __m128i v25; // xmm0
  bool v26; // cc
  unsigned __int64 v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // rcx
  _DWORD *v30; // rax
  unsigned __int64 v31; // rdi
  const char *v32; // rax
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rsi
  __m128i v40; // xmm5
  unsigned int v41; // edx
  __int64 v42; // r14
  __int64 v43; // rcx
  unsigned __int64 v44; // rdx
  __m128i v45; // xmm3
  unsigned __int64 v46; // rdi
  __int64 v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // rax
  unsigned __int64 v50; // rdi
  __int64 v51; // rax
  int v52; // ecx
  unsigned __int64 v53; // r14
  __m128i v54; // xmm2
  unsigned __int64 v55; // rdi
  __int64 v56; // rcx
  __int64 v57; // rax
  _DWORD *v58; // rax
  unsigned __int64 v59; // rdi
  unsigned int v60; // ecx
  __int64 v61; // rax
  unsigned __int64 v62; // rcx
  unsigned __int64 v63; // rax
  __m128i v64; // xmm4
  unsigned __int64 v65; // rdi
  __int64 v66; // rcx
  __int64 v67; // rax
  _DWORD *v68; // rax
  unsigned __int64 v69; // rdi
  unsigned __int64 v70; // [rsp+8h] [rbp-98h]
  __int64 v71; // [rsp+10h] [rbp-90h]
  __int64 v72; // [rsp+18h] [rbp-88h]
  unsigned __int64 v73; // [rsp+20h] [rbp-80h]
  _DWORD *v74; // [rsp+20h] [rbp-80h]
  _DWORD *v75; // [rsp+20h] [rbp-80h]
  unsigned __int8 v76; // [rsp+2Fh] [rbp-71h]
  int v77; // [rsp+3Ch] [rbp-64h] BYREF
  __int64 v78; // [rsp+40h] [rbp-60h] BYREF
  __int64 v79; // [rsp+48h] [rbp-58h]
  char v80; // [rsp+50h] [rbp-50h]
  char v81; // [rsp+51h] [rbp-4Fh]
  unsigned __int64 v82; // [rsp+58h] [rbp-48h]
  unsigned int v83; // [rsp+60h] [rbp-40h]

  v8 = *(_DWORD **)(a1 + 152);
  if ( a3 )
  {
    v76 = 0;
    if ( *v8 != 9 )
    {
      v33 = sub_38EAF10(a1);
      v77 = 3;
      v34 = a2[1];
      v78 = v33;
      v79 = v35;
      if ( v34 == a2[2] )
      {
        sub_38E9360(a2, v34, &v77, &v78);
      }
      else
      {
        if ( v34 )
        {
          v36 = v78;
          v37 = v79;
          *(_DWORD *)v34 = 3;
          *(_DWORD *)(v34 + 32) = 64;
          *(_QWORD *)(v34 + 8) = v36;
          *(_QWORD *)(v34 + 16) = v37;
          *(_QWORD *)(v34 + 24) = 0;
          v34 = a2[1];
        }
        a2[1] = v34 + 40;
      }
      return 0;
    }
  }
  else
  {
    *(_BYTE *)(a1 + 256) = *(_BYTE *)(a1 + 844);
    v9 = *v8;
    v76 = v9 == 27 || v9 == 0;
    if ( v76 )
    {
LABEL_33:
      v81 = 1;
      v32 = "unexpected token in macro instantiation";
    }
    else
    {
      v10 = v8;
      v11 = 0;
      while ( 1 )
      {
        if ( !v11 )
        {
          if ( v9 != 25 )
          {
            while ( 1 )
            {
              if ( v9 == 11 )
              {
                v51 = *(unsigned int *)(a1 + 160);
                *(_BYTE *)(a1 + 258) = 0;
                v52 = v51;
                v53 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v51 - 40) >> 3);
                if ( (unsigned __int64)(40 * v51) > 0x28 )
                {
                  do
                  {
                    v54 = _mm_loadu_si128((const __m128i *)v10 + 3);
                    v26 = v10[8] <= 0x40u;
                    *v10 = v10[10];
                    *(__m128i *)(v10 + 2) = v54;
                    if ( !v26 )
                    {
                      v55 = *((_QWORD *)v10 + 3);
                      if ( v55 )
                      {
                        v74 = v10;
                        j_j___libc_free_0_0(v55);
                        v10 = v74;
                      }
                    }
                    v56 = *((_QWORD *)v10 + 8);
                    v10 += 10;
                    *((_QWORD *)v10 - 2) = v56;
                    LODWORD(v56) = v10[8];
                    v10[8] = 0;
                    *(v10 - 2) = v56;
                    --v53;
                  }
                  while ( v53 );
                  v52 = *(_DWORD *)(a1 + 160);
                  v8 = *(_DWORD **)(a1 + 152);
                }
                v57 = (unsigned int)(v52 - 1);
                *(_DWORD *)(a1 + 160) = v57;
                v58 = &v8[10 * v57];
                if ( v58[8] > 0x40u )
                {
                  v59 = *((_QWORD *)v58 + 3);
                  if ( v59 )
                    j_j___libc_free_0_0(v59);
                }
                if ( !*(_DWORD *)(a1 + 160) )
                {
                  sub_392C2E0(&v78, a1 + 144);
                  sub_38E90E0(a1 + 152, *(_QWORD *)(a1 + 152), (unsigned __int64)&v78);
                  if ( v83 > 0x40 )
                  {
                    if ( v82 )
                      j_j___libc_free_0_0(v82);
                  }
                }
                if ( *(_BYTE *)(a1 + 844) )
                  goto LABEL_6;
                v60 = **(_DWORD **)(a1 + 152);
                if ( v60 > 0x2C || ((1LL << v60) & 0x1FCFF980F000LL) == 0 )
                  goto LABEL_6;
              }
              else if ( *(_BYTE *)(a1 + 844) || *v8 > 0x2Cu || ((1LL << *v8) & 0x1FCFF980F000LL) == 0 )
              {
                goto LABEL_10;
              }
              v38 = sub_3909460(a1);
              v39 = a2[1];
              if ( v39 == a2[2] )
              {
                sub_38E95C0(a2, v39, v38);
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
                    sub_16A4FD0(v39 + 24, (const void **)(v38 + 24));
                  else
                    *(_QWORD *)(v39 + 24) = *(_QWORD *)(v38 + 24);
                  v39 = a2[1];
                }
                a2[1] = v39 + 40;
              }
              v42 = *(_QWORD *)(a1 + 152);
              v72 = a1 + 144;
              v43 = *(unsigned int *)(a1 + 160);
              v71 = a1 + 152;
              *(_BYTE *)(a1 + 258) = *(_DWORD *)v42 == 9;
              v44 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v43 - 40) >> 3);
              if ( (unsigned __int64)(40 * v43) > 0x28 )
              {
                do
                {
                  v45 = _mm_loadu_si128((const __m128i *)(v42 + 48));
                  v26 = *(_DWORD *)(v42 + 32) <= 0x40u;
                  *(_DWORD *)v42 = *(_DWORD *)(v42 + 40);
                  *(__m128i *)(v42 + 8) = v45;
                  if ( !v26 )
                  {
                    v46 = *(_QWORD *)(v42 + 24);
                    if ( v46 )
                    {
                      v73 = v44;
                      j_j___libc_free_0_0(v46);
                      v44 = v73;
                    }
                  }
                  v47 = *(_QWORD *)(v42 + 64);
                  v42 += 40;
                  *(_QWORD *)(v42 - 16) = v47;
                  LODWORD(v47) = *(_DWORD *)(v42 + 32);
                  *(_DWORD *)(v42 + 32) = 0;
                  *(_DWORD *)(v42 - 8) = v47;
                  --v44;
                }
                while ( v44 );
                LODWORD(v43) = *(_DWORD *)(a1 + 160);
                v42 = *(_QWORD *)(a1 + 152);
              }
              v48 = (unsigned int)(v43 - 1);
              *(_DWORD *)(a1 + 160) = v48;
              v49 = v42 + 40 * v48;
              if ( *(_DWORD *)(v49 + 32) > 0x40u )
              {
                v50 = *(_QWORD *)(v49 + 24);
                if ( v50 )
                  j_j___libc_free_0_0(v50);
              }
              if ( !*(_DWORD *)(a1 + 160) )
              {
                sub_392C2E0(&v78, v72);
                sub_38E90E0(v71, *(_QWORD *)(a1 + 152), (unsigned __int64)&v78);
                if ( v83 > 0x40 )
                {
                  if ( v82 )
                    j_j___libc_free_0_0(v82);
                }
              }
              v8 = *(_DWORD **)(a1 + 152);
              v9 = *v8;
              v10 = v8;
              if ( *v8 == 11 )
              {
                v61 = *(unsigned int *)(a1 + 160);
                *(_BYTE *)(a1 + 258) = 0;
                v62 = 40 * v61;
                v63 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v61 - 40) >> 3);
                if ( v62 > 0x28 )
                {
                  do
                  {
                    v64 = _mm_loadu_si128((const __m128i *)v10 + 3);
                    v26 = v10[8] <= 0x40u;
                    *v10 = v10[10];
                    *(__m128i *)(v10 + 2) = v64;
                    if ( !v26 )
                    {
                      v65 = *((_QWORD *)v10 + 3);
                      if ( v65 )
                      {
                        v70 = v63;
                        v75 = v10;
                        j_j___libc_free_0_0(v65);
                        v63 = v70;
                        v10 = v75;
                      }
                    }
                    v66 = *((_QWORD *)v10 + 8);
                    v10 += 10;
                    *((_QWORD *)v10 - 2) = v66;
                    LODWORD(v66) = v10[8];
                    v10[8] = 0;
                    *(v10 - 2) = v66;
                    --v63;
                  }
                  while ( v63 );
                  v8 = *(_DWORD **)(a1 + 152);
                }
                v67 = (unsigned int)(*(_DWORD *)(a1 + 160) - 1);
                *(_DWORD *)(a1 + 160) = v67;
                v68 = &v8[10 * v67];
                if ( v68[8] > 0x40u )
                {
                  v69 = *((_QWORD *)v68 + 3);
                  if ( v69 )
                    j_j___libc_free_0_0(v69);
                }
                if ( !*(_DWORD *)(a1 + 160) )
                {
                  sub_392C2E0(&v78, v72);
                  sub_38E90E0(v71, *(_QWORD *)(a1 + 152), (unsigned __int64)&v78);
                  if ( v83 > 0x40 )
                  {
                    if ( v82 )
                      j_j___libc_free_0_0(v82);
                  }
                }
                v8 = *(_DWORD **)(a1 + 152);
                v9 = *v8;
                v10 = v8;
                if ( *v8 == 27 || !v9 )
                  goto LABEL_33;
              }
              else if ( !v9 || v9 == 27 )
              {
                goto LABEL_33;
              }
              if ( v9 == 25 )
                goto LABEL_6;
            }
          }
          goto LABEL_6;
        }
LABEL_10:
        v13 = *v8;
        if ( *v8 == 9 )
          break;
        if ( v13 == 17 )
        {
          ++v11;
        }
        else
        {
          v14 = v11;
          v11 = (v11 == 0) + v11 - 1;
          if ( v13 != 18 )
            v11 = v14;
        }
        v15 = sub_3909460(a1);
        v16 = a2[1];
        v17 = v15;
        if ( v16 == a2[2] )
        {
          sub_38E95C0(a2, v16, v15);
        }
        else
        {
          if ( v16 )
          {
            v18 = _mm_loadu_si128((const __m128i *)(v15 + 8));
            *(_DWORD *)v16 = *(_DWORD *)v15;
            *(__m128i *)(v16 + 8) = v18;
            v19 = *(_DWORD *)(v15 + 32);
            *(_DWORD *)(v16 + 32) = v19;
            if ( v19 > 0x40 )
              sub_16A4FD0(v16 + 24, (const void **)(v17 + 24));
            else
              *(_QWORD *)(v16 + 24) = *(_QWORD *)(v17 + 24);
            v16 = a2[1];
          }
          a2[1] = v16 + 40;
        }
        v20 = *(_DWORD **)(a1 + 152);
        v21 = *(unsigned int *)(a1 + 160);
        v22 = v20 + 10;
        v23 = *(_DWORD *)(a1 + 160);
        *(_BYTE *)(a1 + 258) = *v20 == 9;
        v24 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v21 - 40) >> 3);
        if ( (unsigned __int64)(40 * v21) > 0x28 )
        {
          do
          {
            v25 = _mm_loadu_si128((const __m128i *)(v22 + 2));
            v26 = *(v22 - 2) <= 0x40u;
            *(v22 - 10) = *v22;
            *((__m128i *)v22 - 2) = v25;
            if ( !v26 )
            {
              v27 = *((_QWORD *)v22 - 2);
              if ( v27 )
                j_j___libc_free_0_0(v27);
            }
            v28 = *((_QWORD *)v22 + 3);
            v22 += 10;
            *((_QWORD *)v22 - 7) = v28;
            LODWORD(v28) = *(v22 - 2);
            *(v22 - 2) = 0;
            *(v22 - 12) = v28;
            --v24;
          }
          while ( v24 );
          v23 = *(_DWORD *)(a1 + 160);
          v20 = *(_DWORD **)(a1 + 152);
        }
        v29 = (unsigned int)(v23 - 1);
        *(_DWORD *)(a1 + 160) = v29;
        v30 = &v20[10 * v29];
        if ( v30[8] > 0x40u )
        {
          v31 = *((_QWORD *)v30 + 3);
          if ( v31 )
            j_j___libc_free_0_0(v31);
        }
        if ( !*(_DWORD *)(a1 + 160) )
        {
          sub_392C2E0(&v78, a1 + 144);
          sub_38E90E0(a1 + 152, *(_QWORD *)(a1 + 152), (unsigned __int64)&v78);
          if ( v83 > 0x40 )
          {
            if ( v82 )
              j_j___libc_free_0_0(v82);
          }
        }
        v8 = *(_DWORD **)(a1 + 152);
        v9 = *v8;
        v10 = v8;
        if ( *v8 == 27 || !v9 )
          goto LABEL_33;
      }
      if ( !v11 )
        goto LABEL_6;
      v81 = 1;
      v32 = "unbalanced parentheses in macro argument";
    }
    v78 = (__int64)v32;
    v80 = 3;
    v76 = sub_3909CF0(a1, &v78, 0, 0, a5, a6);
LABEL_6:
    *(_BYTE *)(a1 + 256) = 1;
  }
  return v76;
}
