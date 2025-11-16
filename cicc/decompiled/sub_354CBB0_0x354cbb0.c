// Function: sub_354CBB0
// Address: 0x354cbb0
//
void __fastcall sub_354CBB0(__int64 a1)
{
  __int64 v1; // r10
  __int64 v2; // rbx
  __int64 v3; // r14
  __int64 v4; // r15
  unsigned __int64 v5; // rax
  int v6; // esi
  __int64 v7; // rdi
  int v8; // esi
  unsigned int v9; // ecx
  __int64 *v10; // rdx
  __int64 v11; // r9
  __int64 v12; // rdi
  unsigned __int64 v13; // rax
  __int64 v14; // r9
  int v15; // esi
  __int64 v16; // rdi
  int v17; // esi
  __int64 v18; // rcx
  __int64 *v19; // rdx
  __int64 v20; // r11
  __int64 v21; // r13
  __int64 v22; // rdx
  __int64 v23; // r8
  const __m128i *v24; // r12
  __int64 m128i_i64; // r9
  __m128i v26; // xmm0
  __int64 v27; // rdx
  __m128i *v28; // r12
  __m128i *v29; // rbx
  __m128i *v30; // rsi
  __int64 v31; // r8
  const __m128i *v32; // r12
  __int64 v33; // rcx
  __int64 v34; // r8
  __m128i v35; // xmm0
  __m128i *v36; // rcx
  __m128i *v37; // r12
  __m128i *v38; // rbx
  __m128i *v39; // rsi
  __int64 v40; // rcx
  unsigned __int64 v41; // r8
  unsigned __int64 v42; // r9
  unsigned int v43; // esi
  int v44; // r13d
  __int64 v45; // r12
  __int32 v46; // r11d
  __int64 v47; // r8
  __int64 *v48; // rcx
  unsigned int v49; // edx
  __int64 *v50; // rax
  __int64 v51; // rdi
  _QWORD *v52; // rax
  __m128i *v53; // rdi
  int v54; // edx
  int v55; // r8d
  int v56; // edx
  int v57; // edx
  __int64 v58; // rax
  int v59; // eax
  __int64 v60; // [rsp+0h] [rbp-120h]
  __int64 v61; // [rsp+8h] [rbp-118h]
  __int64 v62; // [rsp+8h] [rbp-118h]
  __m128i v63; // [rsp+20h] [rbp-100h] BYREF
  __int64 v64; // [rsp+30h] [rbp-F0h]
  __int64 *v65; // [rsp+38h] [rbp-E8h]
  unsigned int *v66; // [rsp+40h] [rbp-E0h]
  unsigned int *v67; // [rsp+48h] [rbp-D8h]
  unsigned int *v68; // [rsp+50h] [rbp-D0h]
  __int64 v69; // [rsp+58h] [rbp-C8h]
  unsigned int v70; // [rsp+6Ch] [rbp-B4h] BYREF
  int v71; // [rsp+70h] [rbp-B0h] BYREF
  int v72; // [rsp+74h] [rbp-ACh] BYREF
  __int64 v73; // [rsp+78h] [rbp-A8h] BYREF
  __int64 v74; // [rsp+80h] [rbp-A0h] BYREF
  __int64 *v75; // [rsp+88h] [rbp-98h] BYREF
  unsigned __int64 v76; // [rsp+90h] [rbp-90h] BYREF
  int v77; // [rsp+98h] [rbp-88h]
  int v78; // [rsp+9Ch] [rbp-84h]
  __m128i *v79; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v80; // [rsp+A8h] [rbp-78h]
  _BYTE v81[112]; // [rsp+B0h] [rbp-70h] BYREF

  v1 = *(_QWORD *)(a1 + 48);
  v69 = *(_QWORD *)(a1 + 56);
  if ( v1 != v69 )
  {
    v2 = a1;
    v3 = v1;
    v65 = &v73;
    v4 = a1 + 3528;
    v66 = (unsigned int *)&v72;
    v67 = (unsigned int *)&v71;
    v68 = &v70;
    while ( 1 )
    {
      v70 = 0;
      v71 = 0;
      v64 = v3;
      v72 = 0;
      v73 = 0;
      if ( (unsigned __int8)sub_3543F70(v2, *(_QWORD *)v3, v68, v67, v66, v65) )
      {
        v5 = sub_2EBEE90(*(_QWORD *)(v2 + 40), *(_DWORD *)(*(_QWORD *)(*(_QWORD *)v3 + 32LL) + 40LL * v70 + 8));
        if ( v5 )
        {
          v6 = *(_DWORD *)(v2 + 960);
          v7 = *(_QWORD *)(v2 + 944);
          if ( v6 )
          {
            v8 = v6 - 1;
            v9 = v8 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
            v10 = (__int64 *)(v7 + 16LL * v9);
            v11 = *v10;
            if ( v5 != *v10 )
            {
              v54 = 1;
              while ( v11 != -4096 )
              {
                v55 = v54 + 1;
                v9 = v8 & (v54 + v9);
                v10 = (__int64 *)(v7 + 16LL * v9);
                v11 = *v10;
                if ( v5 == *v10 )
                  goto LABEL_7;
                v54 = v55;
              }
              goto LABEL_13;
            }
LABEL_7:
            if ( v10[1] )
            {
              v12 = *(_QWORD *)(v2 + 40);
              v63.m128i_i64[0] = v10[1];
              v13 = sub_2EBEE90(v12, v72);
              if ( v13 )
              {
                v15 = *(_DWORD *)(v2 + 960);
                v16 = *(_QWORD *)(v2 + 944);
                if ( v15 )
                {
                  v17 = v15 - 1;
                  v18 = v17 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
                  v19 = (__int64 *)(v16 + 16 * v18);
                  v20 = *v19;
                  if ( v13 != *v19 )
                  {
                    v56 = 1;
                    while ( v20 != -4096 )
                    {
                      v14 = (unsigned int)(v56 + 1);
                      v18 = v17 & (unsigned int)(v56 + v18);
                      v19 = (__int64 *)(v16 + 16LL * (unsigned int)v18);
                      v20 = *v19;
                      if ( v13 == *v19 )
                        goto LABEL_11;
                      v56 = v14;
                    }
                    goto LABEL_13;
                  }
LABEL_11:
                  v21 = v19[1];
                  if ( v21 )
                  {
                    if ( !(unsigned __int8)sub_2F90B20(v4, v3, v19[1], v18, v63.m128i_i64[0], v14) )
                      break;
                  }
                }
              }
            }
          }
        }
      }
LABEL_13:
      v3 += 256;
      if ( v69 == v3 )
        return;
    }
    v22 = 0;
    v23 = v63.m128i_i64[0];
    v79 = (__m128i *)v81;
    v80 = 0x400000000LL;
    v24 = *(const __m128i **)(v3 + 40);
    m128i_i64 = (__int64)v24[*(unsigned int *)(v3 + 48)].m128i_i64;
    if ( v24 != (const __m128i *)m128i_i64 )
    {
      do
      {
        while ( v23 != (v24->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) )
        {
          if ( (const __m128i *)m128i_i64 == ++v24 )
            goto LABEL_22;
        }
        v26 = _mm_loadu_si128(v24);
        if ( v22 + 1 > (unsigned __int64)HIDWORD(v80) )
        {
          v60 = v23;
          v61 = m128i_i64;
          v63 = v26;
          sub_C8D5F0((__int64)&v79, v81, v22 + 1, 0x10u, v23, m128i_i64);
          v22 = (unsigned int)v80;
          v23 = v60;
          m128i_i64 = v61;
          v26 = _mm_load_si128(&v63);
        }
        ++v24;
        v79[v22] = v26;
        v22 = (unsigned int)(v80 + 1);
        LODWORD(v80) = v80 + 1;
      }
      while ( (const __m128i *)m128i_i64 != v24 );
LABEL_22:
      v27 = v22;
      if ( &v79[v27] != v79 )
      {
        v63.m128i_i64[0] = v2;
        v28 = &v79[v27];
        v29 = v79;
        do
        {
          nullsub_1666();
          v30 = v29++;
          sub_2F8F420(v3, v30);
        }
        while ( v28 != v29 );
        v2 = v63.m128i_i64[0];
      }
    }
    v31 = *(unsigned int *)(v21 + 48);
    v32 = *(const __m128i **)(v21 + 40);
    LODWORD(v80) = 0;
    v33 = 0;
    v34 = (__int64)v32[v31].m128i_i64;
    if ( (const __m128i *)v34 != v32 )
    {
      do
      {
        while ( v3 != (v32->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) || ((v32->m128i_i8[0] ^ 6) & 6) != 0 )
        {
          if ( (const __m128i *)v34 == ++v32 )
            goto LABEL_34;
        }
        v35 = _mm_loadu_si128(v32);
        if ( v33 + 1 > (unsigned __int64)HIDWORD(v80) )
        {
          v62 = v34;
          v63 = v35;
          sub_C8D5F0((__int64)&v79, v81, v33 + 1, 0x10u, v34, m128i_i64);
          v33 = (unsigned int)v80;
          v34 = v62;
          v35 = _mm_load_si128(&v63);
        }
        ++v32;
        v79[v33] = v35;
        v33 = (unsigned int)(v80 + 1);
        LODWORD(v80) = v80 + 1;
      }
      while ( (const __m128i *)v34 != v32 );
LABEL_34:
      v36 = &v79[v33];
      if ( v36 != v79 )
      {
        v63.m128i_i64[0] = v2;
        v37 = v36;
        v38 = v79;
        do
        {
          nullsub_1666();
          v39 = v38++;
          sub_2F8F420(v21, v39);
        }
        while ( v37 != v38 );
        v2 = v63.m128i_i64[0];
      }
    }
    v78 = 0;
    v76 = v3 & 0xFFFFFFFFFFFFFFF9LL | 2;
    v77 = v72;
    sub_2F90A20(v4, v21, v3);
    sub_2F8F1B0(v21, (__int64)&v76, 1u, v40, v41, v42);
    v43 = *(_DWORD *)(v2 + 4040);
    v74 = v3;
    v44 = v72;
    v45 = v73;
    if ( v43 )
    {
      v46 = 1;
      v47 = *(_QWORD *)(v2 + 4024);
      v48 = 0;
      v49 = (v43 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v50 = (__int64 *)(v47 + 24LL * v49);
      v51 = *v50;
      if ( v3 == *v50 )
      {
LABEL_40:
        v52 = v50 + 1;
LABEL_41:
        *(_DWORD *)v52 = v44;
        v53 = v79;
        v52[1] = v45;
        if ( v53 != (__m128i *)v81 )
          _libc_free((unsigned __int64)v53);
        goto LABEL_13;
      }
      while ( v51 != -4096 )
      {
        if ( !v48 && v51 == -8192 )
          v48 = v50;
        v49 = (v43 - 1) & (v46 + v49);
        v63.m128i_i32[0] = v46 + 1;
        v50 = (__int64 *)(v47 + 24LL * v49);
        v51 = *v50;
        if ( v3 == *v50 )
          goto LABEL_40;
        v46 = v63.m128i_i32[0];
      }
      if ( !v48 )
        v48 = v50;
      v59 = *(_DWORD *)(v2 + 4032);
      ++*(_QWORD *)(v2 + 4016);
      v57 = v59 + 1;
      v75 = v48;
      if ( 4 * (v59 + 1) < 3 * v43 )
      {
        if ( v43 - *(_DWORD *)(v2 + 4036) - v57 > v43 >> 3 )
          goto LABEL_54;
        goto LABEL_53;
      }
    }
    else
    {
      ++*(_QWORD *)(v2 + 4016);
      v75 = 0;
    }
    v43 *= 2;
LABEL_53:
    v64 = v2 + 4016;
    sub_354C9B0(v2 + 4016, v43);
    sub_3547220(v64, &v74, &v75);
    v48 = v75;
    v64 = v74;
    v57 = *(_DWORD *)(v2 + 4032) + 1;
LABEL_54:
    *(_DWORD *)(v2 + 4032) = v57;
    if ( *v48 != -4096 )
      --*(_DWORD *)(v2 + 4036);
    v58 = v64;
    *((_DWORD *)v48 + 2) = 0;
    v48[2] = 0;
    *v48 = v58;
    v52 = v48 + 1;
    goto LABEL_41;
  }
}
