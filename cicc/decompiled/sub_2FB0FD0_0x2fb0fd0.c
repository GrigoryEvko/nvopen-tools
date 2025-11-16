// Function: sub_2FB0FD0
// Address: 0x2fb0fd0
//
char __fastcall sub_2FB0FD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  int v8; // edx
  int v9; // ecx
  unsigned __int64 v10; // rcx
  unsigned int v11; // eax
  int v12; // edx
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rbx
  void *v16; // rdi
  __int64 *v17; // rbx
  __int64 v18; // r14
  __int64 *v19; // r8
  __int64 v20; // rdx
  __int64 *v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // r12
  __int64 v24; // r10
  unsigned int v25; // ecx
  unsigned __int64 v26; // rcx
  unsigned int v27; // edx
  char v28; // r11
  __int64 v29; // rax
  unsigned int v30; // edx
  unsigned int v31; // esi
  __int64 v32; // r14
  __int64 v33; // rdi
  __int64 v34; // r11
  unsigned __int64 v35; // r15
  unsigned int v36; // ecx
  unsigned int v37; // eax
  __int64 v38; // rax
  unsigned __int64 v39; // rsi
  unsigned __int64 v40; // rcx
  unsigned __int64 v41; // rdx
  const __m128i *v42; // r9
  __m128i *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rax
  int v47; // ecx
  __int64 v48; // rdx
  unsigned __int64 v49; // rcx
  const __m128i *v50; // r14
  unsigned __int64 v51; // r9
  __m128i *v52; // rdx
  __int64 v53; // rcx
  _QWORD *v54; // r9
  __int64 v55; // rdi
  __int64 *v56; // rsi
  const void *v57; // rsi
  const void *v58; // rsi
  char *v59; // r15
  __int64 v60; // r12
  char v61; // bl
  __int64 v63; // [rsp+8h] [rbp-A8h]
  __int64 v64; // [rsp+8h] [rbp-A8h]
  int v65; // [rsp+14h] [rbp-9Ch]
  int v66; // [rsp+14h] [rbp-9Ch]
  __int64 *v67; // [rsp+18h] [rbp-98h]
  __int64 *v68; // [rsp+18h] [rbp-98h]
  char *v69; // [rsp+28h] [rbp-88h]
  __int64 *v70; // [rsp+28h] [rbp-88h]
  __int64 *v71; // [rsp+30h] [rbp-80h]
  __int64 v72; // [rsp+30h] [rbp-80h]
  __int64 v73; // [rsp+30h] [rbp-80h]
  __int64 v74; // [rsp+38h] [rbp-78h]
  __int64 *v75; // [rsp+48h] [rbp-68h]
  __int64 v76; // [rsp+50h] [rbp-60h] BYREF
  __int64 v77; // [rsp+58h] [rbp-58h]
  __int64 v78; // [rsp+60h] [rbp-50h]
  __int64 v79; // [rsp+68h] [rbp-48h]
  __int16 v80; // [rsp+70h] [rbp-40h]

  v7 = (__int64)(*(_QWORD *)(*(_QWORD *)a1 + 104LL) - *(_QWORD *)(*(_QWORD *)a1 + 96LL)) >> 3;
  LOBYTE(v8) = v7;
  v9 = *(_DWORD *)(a1 + 688) & 0x3F;
  if ( v9 )
    *(_QWORD *)(*(_QWORD *)(a1 + 624) + 8LL * *(unsigned int *)(a1 + 632) - 8) &= ~(-1LL << v9);
  *(_DWORD *)(a1 + 688) = v7;
  v10 = *(unsigned int *)(a1 + 632);
  v11 = (unsigned int)(v7 + 63) >> 6;
  if ( v11 != v10 )
  {
    if ( v11 >= v10 )
    {
      v15 = v11 - v10;
      if ( v11 > (unsigned __int64)*(unsigned int *)(a1 + 636) )
      {
        sub_C8D5F0(a1 + 624, (const void *)(a1 + 640), v11, 8u, v11, a6);
        v10 = *(unsigned int *)(a1 + 632);
      }
      v16 = (void *)(*(_QWORD *)(a1 + 624) + 8 * v10);
      if ( 8 * v15 )
      {
        memset(v16, 0, 8 * v15);
        LODWORD(v10) = *(_DWORD *)(a1 + 632);
      }
      v8 = *(_DWORD *)(a1 + 688);
      *(_DWORD *)(a1 + 632) = v15 + v10;
    }
    else
    {
      *(_DWORD *)(a1 + 632) = v11;
    }
  }
  v12 = v8 & 0x3F;
  if ( v12 )
    *(_QWORD *)(*(_QWORD *)(a1 + 624) + 8LL * *(unsigned int *)(a1 + 632) - 8) &= ~(-1LL << v12);
  *(_DWORD *)(a1 + 616) = 0;
  v13 = *(_QWORD *)(a1 + 40);
  *(_DWORD *)(a1 + 696) = 0;
  v14 = *(unsigned int *)(v13 + 8);
  if ( (_DWORD)v14 )
  {
    v17 = *(__int64 **)v13;
    v71 = *(__int64 **)(a1 + 200);
    v74 = *(_QWORD *)v13 + 24 * v14;
    v75 = &v71[*(unsigned int *)(a1 + 208)];
    v18 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL);
    v14 = sub_2F65810(v18, **(_QWORD **)v13);
    v19 = v71;
    while ( 1 )
    {
      v20 = *(unsigned int *)(v14 + 24);
      v77 = 0;
      v78 = 0;
      v21 = (__int64 *)(*(_QWORD *)(v18 + 152) + 16 * v20);
      v79 = 0;
      v22 = *v21;
      v23 = v21[1];
      v76 = v14;
      if ( v19 == v75
        || (v24 = (v23 >> 1) & 3,
            v25 = *(_DWORD *)((v23 & 0xFFFFFFFFFFFFFFF8LL) + 24) | v24,
            (*(_DWORD *)((*v19 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v19 >> 1) & 3) >= v25) )
      {
        ++*(_DWORD *)(a1 + 696);
        *(_QWORD *)(*(_QWORD *)(a1 + 624) + 8LL * (*(_DWORD *)(v14 + 24) >> 6)) |= 1LL << *(_DWORD *)(v14 + 24);
      }
      else
      {
        v77 = *v19;
        do
          ++v19;
        while ( v75 != v19 && v25 > (*(_DWORD *)((*v19 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v19 >> 1) & 3) );
        v78 = *(v19 - 1);
        v30 = *(_DWORD *)((*v17 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v17 >> 1) & 3;
        v31 = *(_DWORD *)((v22 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v22 >> 1) & 3;
        LOBYTE(v80) = v30 <= v31;
        if ( v30 > v31 )
          v79 = v77;
        v32 = v17[1];
        HIBYTE(v80) = 1;
        v33 = a1 + 280;
        if ( v25 > (*(_DWORD *)((v32 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v32 >> 1) & 3) )
        {
          v34 = v23;
          v72 = v14;
          v35 = v23 & 0xFFFFFFFFFFFFFFF8LL;
          do
          {
            v17 += 3;
            if ( (__int64 *)v74 == v17
              || (v36 = v24 | *(_DWORD *)(v35 + 24),
                  v37 = *(_DWORD *)((*v17 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v17 >> 1) & 3,
                  v37 >= v36) )
            {
              HIBYTE(v80) = 0;
              v14 = v72;
              v23 = v34;
              v78 = v32;
              goto LABEL_45;
            }
            if ( v37 > (*(_DWORD *)((v32 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v32 >> 1) & 3) )
            {
              v38 = *(unsigned int *)(a1 + 288);
              v39 = *(unsigned int *)(a1 + 292);
              HIBYTE(v80) = 0;
              ++*(_DWORD *)(a1 + 616);
              v40 = *(_QWORD *)(a1 + 280);
              v41 = v38 + 1;
              v42 = (const __m128i *)&v76;
              if ( v38 + 1 > v39 )
              {
                v57 = (const void *)(a1 + 296);
                if ( v40 > (unsigned __int64)&v76 )
                {
                  v64 = v34;
                  v66 = v24;
                  v68 = v19;
                  sub_C8D5F0(v33, v57, v41, 0x28u, (__int64)v19, (__int64)&v76);
                  v40 = *(_QWORD *)(a1 + 280);
                  v38 = *(unsigned int *)(a1 + 288);
                  v42 = (const __m128i *)&v76;
                  v19 = v68;
                  LODWORD(v24) = v66;
                  v34 = v64;
                }
                else
                {
                  v63 = v34;
                  v65 = v24;
                  v67 = v19;
                  if ( (unsigned __int64)&v76 < v40 + 40 * v38 )
                  {
                    v69 = (char *)&v76 - v40;
                    sub_C8D5F0(v33, v57, v41, 0x28u, (__int64)v19, (__int64)&v76);
                    v40 = *(_QWORD *)(a1 + 280);
                    v34 = v63;
                    LODWORD(v24) = v65;
                    v42 = (const __m128i *)&v69[v40];
                    v19 = v67;
                    v38 = *(unsigned int *)(a1 + 288);
                  }
                  else
                  {
                    sub_C8D5F0(v33, v57, v41, 0x28u, (__int64)v19, (__int64)&v76);
                    v40 = *(_QWORD *)(a1 + 280);
                    v38 = *(unsigned int *)(a1 + 288);
                    v19 = v67;
                    LODWORD(v24) = v65;
                    v34 = v63;
                    v42 = (const __m128i *)&v76;
                  }
                }
              }
              v43 = (__m128i *)(v40 + 40 * v38);
              *v43 = _mm_loadu_si128(v42);
              v43[1] = _mm_loadu_si128(v42 + 1);
              v44 = v42[2].m128i_i64[0];
              v80 = 256;
              v43[2].m128i_i64[0] = v44;
              v45 = *(_QWORD *)(a1 + 280);
              v46 = (unsigned int)(*(_DWORD *)(a1 + 288) + 1);
              *(_DWORD *)(a1 + 288) = v46;
              *(_QWORD *)(v45 + 40 * v46 - 24) = v32;
              v47 = *(_DWORD *)(v35 + 24);
              v79 = *v17;
              v77 = v79;
              v36 = v24 | v47;
            }
            if ( (v79 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
              v79 = *v17;
            v32 = v17[1];
          }
          while ( (*(_DWORD *)((v32 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v32 >> 1) & 3) < v36 );
          v14 = v72;
          v23 = v34;
        }
LABEL_45:
        v48 = *(unsigned int *)(a1 + 288);
        v49 = *(_QWORD *)(a1 + 280);
        v50 = (const __m128i *)&v76;
        v51 = v48 + 1;
        if ( v48 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 292) )
        {
          v73 = v14;
          v58 = (const void *)(a1 + 296);
          v70 = v19;
          if ( v49 > (unsigned __int64)&v76 || (unsigned __int64)&v76 >= v49 + 40 * v48 )
          {
            sub_C8D5F0(v33, v58, v51, 0x28u, (__int64)v19, v51);
            v49 = *(_QWORD *)(a1 + 280);
            v50 = (const __m128i *)&v76;
            v48 = *(unsigned int *)(a1 + 288);
            v19 = v70;
            v14 = v73;
          }
          else
          {
            v59 = (char *)&v76 - v49;
            sub_C8D5F0(v33, v58, v51, 0x28u, (__int64)v19, v51);
            v49 = *(_QWORD *)(a1 + 280);
            v14 = v73;
            v48 = *(unsigned int *)(a1 + 288);
            v19 = v70;
            v50 = (const __m128i *)&v59[v49];
          }
        }
        v52 = (__m128i *)(v49 + 40 * v48);
        *v52 = _mm_loadu_si128(v50);
        v52[1] = _mm_loadu_si128(v50 + 1);
        v52[2].m128i_i64[0] = v50[2].m128i_i64[0];
        ++*(_DWORD *)(a1 + 288);
        if ( v17 == (__int64 *)v74 )
        {
LABEL_23:
          v28 = byte_5025CC8;
          if ( byte_5025CC8 )
          {
            if ( *(_DWORD *)(a1 + 288) == 2 )
            {
              v60 = *(_QWORD *)(a1 + 280);
              LOBYTE(v14) = sub_2FB03C0(a1, v60);
              v61 = v14;
              if ( !(_BYTE)v14 )
              {
                LOBYTE(v14) = sub_2FB03C0(a1, v60 + 40);
                if ( !(_BYTE)v14 )
                  v28 = v61;
              }
            }
            else
            {
              v28 = 0;
            }
          }
          *(_BYTE *)(a1 + 700) = v28;
          return v14;
        }
      }
      if ( v17[1] == v23 )
      {
        v17 += 3;
        if ( (__int64 *)v74 == v17 )
          goto LABEL_23;
      }
      v18 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL);
      v26 = *v17 & 0xFFFFFFFFFFFFFFF8LL;
      v27 = *(_DWORD *)(v26 + 24) | (*v17 >> 1) & 3;
      if ( v27 >= (*(_DWORD *)((v23 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v23 >> 1) & 3) )
      {
        v29 = *(_QWORD *)(v26 + 16);
        if ( v29 )
        {
          v14 = *(_QWORD *)(v29 + 24);
        }
        else
        {
          v53 = *(unsigned int *)(v18 + 304);
          v54 = *(_QWORD **)(v18 + 296);
          if ( *(_DWORD *)(v18 + 304) )
          {
            do
            {
              while ( 1 )
              {
                v55 = v53 >> 1;
                v56 = &v54[2 * (v53 >> 1)];
                if ( v27 < (*(_DWORD *)((*v56 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v56 >> 1) & 3) )
                  break;
                v54 = v56 + 2;
                v53 = v53 - v55 - 1;
                if ( v53 <= 0 )
                  goto LABEL_54;
              }
              v53 >>= 1;
            }
            while ( v55 > 0 );
          }
LABEL_54:
          v14 = *(v54 - 1);
        }
      }
      else
      {
        v14 = *(_QWORD *)(v14 + 8);
      }
    }
  }
  return v14;
}
