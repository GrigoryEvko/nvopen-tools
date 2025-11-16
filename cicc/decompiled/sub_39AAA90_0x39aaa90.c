// Function: sub_39AAA90
// Address: 0x39aaa90
//
void __fastcall sub_39AAA90(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v5; // r8
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned int v9; // r9d
  char v10; // cl
  char v11; // r15
  __int64 v12; // r11
  _QWORD *v13; // r10
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // r13
  unsigned int v17; // edi
  __int64 *v18; // r12
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdx
  __m128i v23; // xmm6
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rdx
  int v27; // esi
  __int64 v28; // rdx
  __int64 v29; // rdi
  __m128i v30; // xmm2
  __int64 v31; // rdx
  __int64 v32; // rax
  __m128i *v33; // rax
  __int16 v34; // dx
  bool v35; // al
  char v36; // al
  __int64 v37; // rax
  int v38; // edx
  int v39; // edx
  __int64 v40; // rdi
  int v41; // esi
  __int64 *v42; // rax
  __int64 v43; // rcx
  unsigned int v44; // r13d
  unsigned __int64 v45; // rdx
  __int64 v46; // r12
  __int64 v47; // rax
  __int64 v48; // rcx
  __m128i v49; // xmm4
  int v50; // edx
  int v51; // eax
  int v52; // r12d
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // [rsp+8h] [rbp-B8h]
  _QWORD *v57; // [rsp+10h] [rbp-B0h]
  _QWORD *v58; // [rsp+10h] [rbp-B0h]
  _QWORD *v59; // [rsp+10h] [rbp-B0h]
  _QWORD *v60; // [rsp+10h] [rbp-B0h]
  char v61; // [rsp+18h] [rbp-A8h]
  char v62; // [rsp+18h] [rbp-A8h]
  _QWORD *v63; // [rsp+18h] [rbp-A8h]
  __int64 v64; // [rsp+18h] [rbp-A8h]
  __int64 v65; // [rsp+20h] [rbp-A0h]
  __int64 v66; // [rsp+20h] [rbp-A0h]
  __int64 v67; // [rsp+20h] [rbp-A0h]
  int v68; // [rsp+20h] [rbp-A0h]
  __int64 v69; // [rsp+20h] [rbp-A0h]
  __int64 v70; // [rsp+28h] [rbp-98h]
  int v73; // [rsp+44h] [rbp-7Ch]
  __int64 v75; // [rsp+48h] [rbp-78h]
  __int64 v76; // [rsp+50h] [rbp-70h] BYREF
  unsigned __int64 v77; // [rsp+58h] [rbp-68h]
  __int64 v78; // [rsp+60h] [rbp-60h]
  unsigned int v79; // [rsp+68h] [rbp-58h]
  _QWORD v80[10]; // [rsp+70h] [rbp-50h] BYREF

  v76 = 0;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  sub_39AA770(a1, a3, (__int64)&v76);
  v6 = *(_QWORD *)(a1 + 8);
  v7 = *(_QWORD *)(v6 + 240);
  v8 = *(_QWORD *)(v6 + 264);
  v56 = v8 + 320;
  v73 = *(_DWORD *)(v7 + 348);
  v70 = *(_QWORD *)(v8 + 328);
  if ( v70 == v8 + 320 )
    goto LABEL_31;
  v9 = a3;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = (_QWORD *)a3;
  do
  {
    v14 = *(_QWORD *)(v70 + 32);
    v75 = v70 + 24;
    if ( v70 + 24 != v14 )
    {
      while ( 1 )
      {
        v15 = *(_QWORD *)(v14 + 16);
        if ( *(_WORD *)v15 != 3 )
        {
          v34 = *(_WORD *)(v14 + 46);
          if ( (v34 & 4) != 0 || (v34 & 8) == 0 )
          {
            v35 = (*(_QWORD *)(v15 + 8) & 0x10LL) != 0;
          }
          else
          {
            v57 = v13;
            v61 = v10;
            v65 = v12;
            v35 = sub_1E15D00(v14, 0x10u, 1);
            v12 = v65;
            v10 = v61;
            v13 = v57;
          }
          if ( v35 )
          {
            v58 = v13;
            v62 = v10;
            v66 = v12;
            v36 = sub_39AA510(v14);
            v12 = v66;
            v10 = v62;
            v13 = v58;
            v11 |= v36 ^ 1;
          }
          goto LABEL_23;
        }
        v16 = *(_QWORD *)(*(_QWORD *)(v14 + 32) + 24LL);
        if ( v16 == v12 )
          v11 = 0;
        if ( v79 )
        {
          LODWORD(v5) = v79 - 1;
          v9 = ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4);
          v17 = (v79 - 1) & v9;
          v18 = (__int64 *)(v77 + 16LL * v17);
          v19 = *v18;
          if ( v16 != *v18 )
          {
            v52 = 1;
            while ( v19 != -8 )
            {
              v17 = v5 & (v52 + v17);
              v68 = v52 + 1;
              v18 = (__int64 *)(v77 + 16LL * v17);
              v19 = *v18;
              if ( v16 == *v18 )
                goto LABEL_11;
              v52 = v68;
            }
            goto LABEL_23;
          }
LABEL_11:
          if ( v18 != (__int64 *)(v77 + 16LL * v79) )
            break;
        }
LABEL_23:
        if ( (*(_BYTE *)v14 & 4) != 0 )
        {
          v14 = *(_QWORD *)(v14 + 8);
          if ( v75 == v14 )
            goto LABEL_25;
        }
        else
        {
          while ( (*(_BYTE *)(v14 + 46) & 8) != 0 )
            v14 = *(_QWORD *)(v14 + 8);
          v14 = *(_QWORD *)(v14 + 8);
          if ( v75 == v14 )
            goto LABEL_25;
        }
      }
      v20 = *(_QWORD *)(*v13 + 8LL * *((unsigned int *)v18 + 2));
      if ( v11 )
      {
        v21 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 240LL);
        if ( (*(_DWORD *)(v21 + 348) & 0xFFFFFFFD) == 1
          || *(_DWORD *)(v21 + 348) == 4 && (v50 = *(_DWORD *)(v21 + 352)) != 0 && v50 != 6 )
        {
          v80[0] = v12;
          v22 = *(unsigned int *)(a2 + 8);
          v80[1] = v16;
          *(_OWORD *)&v80[2] = 0;
          if ( (unsigned int)v22 >= *(_DWORD *)(a2 + 12) )
          {
            v59 = v13;
            v69 = v20;
            sub_16CD150(a2, (const void *)(a2 + 16), 0, 32, v5, v9);
            v22 = *(unsigned int *)(a2 + 8);
            v13 = v59;
            v9 = ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4);
            v20 = v69;
          }
          v23 = _mm_loadu_si128((const __m128i *)&v80[2]);
          v10 = 0;
          v24 = *(_QWORD *)a2 + 32 * v22;
          *(__m128i *)v24 = _mm_loadu_si128((const __m128i *)v80);
          *(__m128i *)(v24 + 16) = v23;
          ++*(_DWORD *)(a2 + 8);
        }
      }
      v12 = *(_QWORD *)(*(_QWORD *)(v20 + 32) + 8LL * *((unsigned int *)v18 + 3));
      if ( !*(_QWORD *)(v20 + 88) )
      {
        v10 = 0;
        goto LABEL_23;
      }
      v25 = *((unsigned int *)v18 + 2);
      v80[0] = v16;
      v80[1] = v12;
      v26 = *a4;
      v80[2] = v20;
      v27 = *(_DWORD *)(v26 + 4 * v25);
      LODWORD(v80[3]) = v27;
      v10 &= v73 != 2;
      if ( v10 )
      {
        v5 = *(_QWORD *)a2;
        v28 = *(unsigned int *)(a2 + 8);
        v29 = *(_QWORD *)a2 + 32 * v28 - 32;
        if ( *(_QWORD *)(v29 + 16) == v20 && v27 == *(_DWORD *)(v29 + 24) )
        {
          *(_QWORD *)(v29 + 8) = v12;
          goto LABEL_23;
        }
        if ( *(_DWORD *)(a2 + 12) > (unsigned int)v28 )
          goto LABEL_21;
        goto LABEL_55;
      }
      if ( v73 != 2 )
      {
        v28 = *(unsigned int *)(a2 + 8);
        if ( *(_DWORD *)(a2 + 12) > (unsigned int)v28 )
        {
LABEL_21:
          v30 = _mm_loadu_si128((const __m128i *)&v80[2]);
          v31 = *(_QWORD *)a2 + 32 * v28;
          *(__m128i *)v31 = _mm_loadu_si128((const __m128i *)v80);
          *(__m128i *)(v31 + 16) = v30;
          ++*(_DWORD *)(a2 + 8);
LABEL_22:
          v10 = 1;
          goto LABEL_23;
        }
LABEL_55:
        v63 = v13;
        v67 = v12;
        sub_16CD150(a2, (const void *)(a2 + 16), 0, 32, v5, v9);
        v28 = *(unsigned int *)(a2 + 8);
        v13 = v63;
        v12 = v67;
        goto LABEL_21;
      }
      v37 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 264LL);
      v38 = *(_DWORD *)(v37 + 488);
      if ( v38 )
      {
        v39 = v38 - 1;
        v40 = *(_QWORD *)(v37 + 472);
        v41 = v39 & v9;
        v42 = (__int64 *)(v40 + 16LL * (v39 & v9));
        v43 = *v42;
        if ( v16 == *v42 )
        {
LABEL_44:
          v44 = *((_DWORD *)v42 + 2);
          v45 = *(unsigned int *)(a2 + 8);
          v46 = 32LL * (v44 - 1);
          if ( (unsigned int)v45 < v44 )
          {
            v47 = v44;
            if ( v44 >= v45 )
            {
              if ( v44 > v45 )
              {
                if ( v44 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
                {
                  v60 = v13;
                  v64 = v12;
                  sub_16CD150(a2, (const void *)(a2 + 16), v44, 32, v5, v9);
                  v45 = *(unsigned int *)(a2 + 8);
                  v13 = v60;
                  v12 = v64;
                  v47 = v44;
                }
                v48 = *(_QWORD *)a2;
                v53 = *(_QWORD *)a2 + 32 * v45;
                v54 = *(_QWORD *)a2 + 32 * v47;
                if ( v53 != v54 )
                {
                  do
                  {
                    if ( v53 )
                    {
                      *(_QWORD *)v53 = 0;
                      *(_QWORD *)(v53 + 8) = 0;
                      *(_QWORD *)(v53 + 16) = 0;
                      *(_DWORD *)(v53 + 24) = 0;
                    }
                    v53 += 32;
                  }
                  while ( v54 != v53 );
                  v48 = *(_QWORD *)a2;
                }
                *(_DWORD *)(a2 + 8) = v44;
                goto LABEL_48;
              }
            }
            else
            {
              *(_DWORD *)(a2 + 8) = v44;
            }
          }
          v48 = *(_QWORD *)a2;
LABEL_48:
          v49 = _mm_loadu_si128((const __m128i *)&v80[2]);
          *(__m128i *)(v48 + v46) = _mm_loadu_si128((const __m128i *)v80);
          *(__m128i *)(v48 + v46 + 16) = v49;
          goto LABEL_22;
        }
        v51 = 1;
        while ( v43 != -8 )
        {
          LODWORD(v5) = v51 + 1;
          v55 = v39 & (unsigned int)(v41 + v51);
          v41 = v55;
          v42 = (__int64 *)(v40 + 16 * v55);
          v43 = *v42;
          if ( v16 == *v42 )
            goto LABEL_44;
          v51 = v5;
        }
      }
      v46 = 0x1FFFFFFFE0LL;
      v48 = *(_QWORD *)a2;
      goto LABEL_48;
    }
LABEL_25:
    v70 = *(_QWORD *)(v70 + 8);
  }
  while ( v56 != v70 );
  if ( v73 != 2 && v11 )
  {
    v32 = *(unsigned int *)(a2 + 8);
    v80[0] = v12;
    memset(&v80[1], 0, 24);
    if ( (unsigned int)v32 >= *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, (const void *)(a2 + 16), 0, 32, v5, v9);
      v32 = *(unsigned int *)(a2 + 8);
    }
    v33 = (__m128i *)(*(_QWORD *)a2 + 32 * v32);
    *v33 = _mm_loadu_si128((const __m128i *)v80);
    v33[1] = _mm_loadu_si128((const __m128i *)&v80[2]);
    ++*(_DWORD *)(a2 + 8);
  }
LABEL_31:
  j___libc_free_0(v77);
}
