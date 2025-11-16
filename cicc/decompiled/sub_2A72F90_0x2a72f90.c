// Function: sub_2A72F90
// Address: 0x2a72f90
//
void __fastcall sub_2A72F90(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // r12
  char v9; // al
  __int64 *v10; // rax
  int v11; // r14d
  __int32 v12; // r12d
  unsigned int v13; // esi
  __m128i v14; // xmm0
  __int64 v15; // rdx
  int v16; // r15d
  __int64 v17; // r11
  __int64 *v18; // rdi
  unsigned int i; // eax
  __int64 *v20; // r8
  __int64 v21; // r9
  __int64 v22; // r12
  unsigned int v23; // esi
  __int64 v24; // r9
  int v25; // r8d
  __int64 v26; // rdi
  __int64 *v27; // rcx
  __int64 v28; // rdx
  __int64 *v29; // rax
  __int64 v30; // r10
  int v31; // ecx
  __int64 v32; // rcx
  int v33; // eax
  __int64 v34; // rdi
  __int64 v35; // rdi
  __int64 v36; // rax
  __int64 v37; // r9
  __m128i *v38; // r15
  __m128i *v39; // rdx
  __int32 v40; // edx
  unsigned __int64 v41; // rdi
  unsigned int v42; // eax
  int v43; // edx
  __int64 v44; // rax
  int v45; // edx
  _QWORD *v46; // rax
  int v47; // eax
  int v48; // eax
  __int64 v49; // rax
  __int64 v50; // rcx
  __int64 v51; // r12
  _QWORD *v52; // rax
  unsigned __int64 v53; // r13
  _QWORD *v54; // r14
  unsigned __int64 v55; // rax
  unsigned __int64 v56; // r12
  unsigned __int64 v57; // r13
  unsigned __int64 v58; // r12
  int v59; // r12d
  __int32 v60; // [rsp+10h] [rbp-E0h]
  __int64 v61; // [rsp+18h] [rbp-D8h]
  _QWORD *v62; // [rsp+18h] [rbp-D8h]
  __int64 *v63; // [rsp+28h] [rbp-C8h] BYREF
  __m128i v64; // [rsp+30h] [rbp-C0h] BYREF
  __m128i v65; // [rsp+50h] [rbp-A0h] BYREF
  int v66; // [rsp+60h] [rbp-90h]
  __m128i v67; // [rsp+80h] [rbp-70h] BYREF
  __int64 v68; // [rsp+90h] [rbp-60h] BYREF
  unsigned __int64 v69; // [rsp+98h] [rbp-58h]
  unsigned int v70; // [rsp+A0h] [rbp-50h]
  unsigned __int64 v71; // [rsp+A8h] [rbp-48h]
  unsigned int v72; // [rsp+B0h] [rbp-40h]

  v6 = a2;
  v8 = **(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL);
  v9 = *(_BYTE *)(v8 + 8);
  if ( v9 != 15 )
  {
    if ( v9 == 7 )
      return;
    v67 = (__m128i)(unsigned __int64)a2;
    v22 = a1 + 232;
    v64.m128i_i64[0] = a2;
    v23 = *(_DWORD *)(a1 + 256);
    v65.m128i_i64[0] = 0;
    v64.m128i_i32[2] = 0;
    if ( v23 )
    {
      v24 = v23 - 1;
      v25 = 1;
      v26 = *(_QWORD *)(a1 + 240);
      v27 = 0;
      LODWORD(v28) = v24 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v29 = (__int64 *)(v26 + 16LL * (unsigned int)v28);
      v30 = *v29;
      if ( v6 == *v29 )
      {
LABEL_25:
        sub_22C0090(&v67.m128i_u8[8]);
        sub_22C0090((unsigned __int8 *)&v65);
        return;
      }
      while ( v30 != -4096 )
      {
        if ( v30 != -8192 || v27 )
          v29 = v27;
        v28 = (unsigned int)v24 & ((_DWORD)v28 + v25);
        v30 = *(_QWORD *)(v26 + 16 * v28);
        if ( v6 == v30 )
          goto LABEL_25;
        ++v25;
        v27 = v29;
        v29 = (__int64 *)(v26 + 16 * v28);
      }
      if ( !v27 )
        v27 = v29;
      v48 = *(_DWORD *)(a1 + 248);
      ++*(_QWORD *)(a1 + 232);
      v43 = v48 + 1;
      v63 = v27;
      if ( 4 * (v48 + 1) < 3 * v23 )
      {
        if ( v23 - *(_DWORD *)(a1 + 252) - v43 > v23 >> 3 )
          goto LABEL_56;
        goto LABEL_55;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 232);
      v63 = 0;
    }
    v23 *= 2;
LABEL_55:
    sub_9E07A0(v22, v23);
    sub_264B350(v22, v64.m128i_i64, &v63);
    v6 = v64.m128i_i64[0];
    v27 = v63;
    v43 = *(_DWORD *)(a1 + 248) + 1;
LABEL_56:
    *(_DWORD *)(a1 + 248) = v43;
    if ( *v27 != -4096 )
      --*(_DWORD *)(a1 + 252);
    *v27 = v6;
    *((_DWORD *)v27 + 2) = v64.m128i_i32[2];
    *((_DWORD *)v27 + 2) = *(_DWORD *)(a1 + 272);
    v44 = *(unsigned int *)(a1 + 272);
    v45 = v44;
    if ( *(_DWORD *)(a1 + 276) <= (unsigned int)v44 )
    {
      v49 = sub_C8D7D0(a1 + 264, a1 + 280, 0, 0x30u, (unsigned __int64 *)&v64, v24);
      v50 = *(unsigned int *)(a1 + 272);
      v62 = (_QWORD *)v49;
      v51 = 48 * v50;
      v52 = (_QWORD *)(48 * v50 + v49);
      if ( v52 )
      {
        *v52 = v67.m128i_i64[0];
        sub_22C0650((__int64)(v52 + 1), &v67.m128i_u8[8]);
        v50 = *(unsigned int *)(a1 + 272);
        v51 = 48 * v50;
      }
      v53 = *(_QWORD *)(a1 + 264);
      v54 = v62;
      v55 = v53 + v51;
      v56 = v53;
      if ( v53 != v55 )
      {
        v57 = v55;
        do
        {
          if ( v54 )
          {
            *v54 = *(_QWORD *)v56;
            sub_22C0650((__int64)(v54 + 1), (unsigned __int8 *)(v56 + 8));
          }
          v56 += 48LL;
          v54 += 6;
        }
        while ( v57 != v56 );
        v50 = *(unsigned int *)(a1 + 272);
        v53 = *(_QWORD *)(a1 + 264);
      }
      v58 = v53 + 48 * v50;
      if ( v53 != v58 )
      {
        do
        {
          v58 -= 48LL;
          sub_22C0090((unsigned __int8 *)(v58 + 8));
        }
        while ( v53 != v58 );
        v53 = *(_QWORD *)(a1 + 264);
      }
      v59 = v64.m128i_i32[0];
      if ( a1 + 280 != v53 )
        _libc_free(v53);
      ++*(_DWORD *)(a1 + 272);
      *(_DWORD *)(a1 + 276) = v59;
      *(_QWORD *)(a1 + 264) = v62;
    }
    else
    {
      v46 = (_QWORD *)(*(_QWORD *)(a1 + 264) + 48 * v44);
      if ( v46 )
      {
        *v46 = v67.m128i_i64[0];
        sub_22C0650((__int64)(v46 + 1), &v67.m128i_u8[8]);
        v45 = *(_DWORD *)(a1 + 272);
      }
      *(_DWORD *)(a1 + 272) = v45 + 1;
    }
    goto LABEL_25;
  }
  if ( !*(_BYTE *)(a1 + 388) )
    goto LABEL_8;
  v10 = *(__int64 **)(a1 + 368);
  a4 = *(unsigned int *)(a1 + 380);
  a3 = &v10[a4];
  if ( v10 != a3 )
  {
    while ( a2 != *v10 )
    {
      if ( a3 == ++v10 )
        goto LABEL_26;
    }
    goto LABEL_9;
  }
LABEL_26:
  if ( (unsigned int)a4 < *(_DWORD *)(a1 + 376) )
  {
    *(_DWORD *)(a1 + 380) = a4 + 1;
    *a3 = a2;
    ++*(_QWORD *)(a1 + 360);
  }
  else
  {
LABEL_8:
    sub_C8CC70(a1 + 360, a2, (__int64)a3, a4, a5, a6);
  }
LABEL_9:
  v11 = *(_DWORD *)(v8 + 12);
  v12 = 0;
  v61 = a1 + 280;
  if ( v11 )
  {
    do
    {
      v67.m128i_i64[0] = v6;
      v13 = *(_DWORD *)(a1 + 304);
      v67.m128i_i32[2] = v12;
      v14 = _mm_loadu_si128(&v67);
      v68 = 0;
      v66 = 0;
      v64 = v14;
      v65 = v14;
      if ( v13 )
      {
        v15 = v65.m128i_i64[0];
        v16 = 1;
        v17 = *(_QWORD *)(a1 + 288);
        v18 = 0;
        for ( i = (v13 - 1)
                & (((0xBF58476D1CE4E5B9LL
                   * ((unsigned int)(37 * v65.m128i_i32[2])
                    | ((unsigned __int64)(((unsigned __int32)v65.m128i_i32[0] >> 9)
                                        ^ ((unsigned __int32)v65.m128i_i32[0] >> 4)) << 32))) >> 31)
                 ^ (756364221 * v65.m128i_i32[2])); ; i = (v13 - 1) & v42 )
        {
          v20 = (__int64 *)(v17 + 24LL * i);
          v21 = *v20;
          if ( *v20 == v65.m128i_i64[0] && *((_DWORD *)v20 + 2) == v65.m128i_i32[2] )
            break;
          if ( v21 == -4096 )
          {
            if ( *((_DWORD *)v20 + 2) == -1 )
            {
              v47 = *(_DWORD *)(a1 + 296);
              if ( !v18 )
                v18 = v20;
              ++*(_QWORD *)(a1 + 280);
              v31 = v47 + 1;
              v63 = v18;
              if ( 4 * (v47 + 1) >= 3 * v13 )
                goto LABEL_29;
              if ( v13 - *(_DWORD *)(a1 + 300) - v31 > v13 >> 3 )
                goto LABEL_31;
              goto LABEL_30;
            }
          }
          else if ( v21 == -8192 && *((_DWORD *)v20 + 2) == -2 && !v18 )
          {
            v18 = (__int64 *)(v17 + 24LL * i);
          }
          v42 = v16 + i;
          ++v16;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 280);
        v63 = 0;
LABEL_29:
        v13 *= 2;
LABEL_30:
        sub_2A6E970(v61, v13);
        sub_2A68270(v61, v65.m128i_i64, &v63);
        v15 = v65.m128i_i64[0];
        v18 = v63;
        v31 = *(_DWORD *)(a1 + 296) + 1;
LABEL_31:
        *(_DWORD *)(a1 + 296) = v31;
        if ( *v18 != -4096 || *((_DWORD *)v18 + 2) != -1 )
          --*(_DWORD *)(a1 + 300);
        *v18 = v15;
        *((_DWORD *)v18 + 2) = v65.m128i_i32[2];
        *((_DWORD *)v18 + 4) = v66;
        *((_DWORD *)v18 + 4) = *(_DWORD *)(a1 + 320);
        v32 = *(unsigned int *)(a1 + 320);
        v33 = v32;
        if ( *(_DWORD *)(a1 + 324) <= (unsigned int)v32 )
        {
          v36 = sub_C8D7D0(a1 + 312, a1 + 328, 0, 0x38u, (unsigned __int64 *)&v65, a1 + 312);
          v37 = a1 + 312;
          v38 = (__m128i *)v36;
          v39 = (__m128i *)(v36 + 56LL * *(unsigned int *)(a1 + 320));
          if ( v39 )
          {
            *v39 = _mm_loadu_si128(&v67);
            sub_22C0650((__int64)v39[1].m128i_i64, (unsigned __int8 *)&v68);
            v37 = a1 + 312;
          }
          sub_2A69D20(v37, v38);
          v40 = v65.m128i_i32[0];
          v41 = *(_QWORD *)(a1 + 312);
          if ( a1 + 328 != v41 )
          {
            v60 = v65.m128i_i32[0];
            _libc_free(v41);
            v40 = v60;
          }
          ++*(_DWORD *)(a1 + 320);
          *(_QWORD *)(a1 + 312) = v38;
          *(_DWORD *)(a1 + 324) = v40;
        }
        else
        {
          v34 = *(_QWORD *)(a1 + 312) + 56 * v32;
          if ( v34 )
          {
            v35 = v34 + 16;
            *(__m128i *)(v35 - 16) = _mm_loadu_si128(&v67);
            sub_22C0650(v35, (unsigned __int8 *)&v68);
            v33 = *(_DWORD *)(a1 + 320);
          }
          *(_DWORD *)(a1 + 320) = v33 + 1;
        }
        if ( (unsigned int)(unsigned __int8)v68 - 4 <= 1 )
        {
          if ( v72 > 0x40 && v71 )
            j_j___libc_free_0_0(v71);
          if ( v70 > 0x40 && v69 )
            j_j___libc_free_0_0(v69);
        }
      }
      ++v12;
    }
    while ( v12 != v11 );
  }
}
