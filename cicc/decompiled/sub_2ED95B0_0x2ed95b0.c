// Function: sub_2ED95B0
// Address: 0x2ed95b0
//
__m128i *__fastcall sub_2ED95B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned __int8 v4; // dl
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 v9; // r15
  int v10; // ebx
  __int64 v11; // rcx
  bool v12; // al
  int v13; // ebx
  int v14; // eax
  int v15; // r8d
  unsigned int i; // eax
  __int64 v17; // rdx
  unsigned int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // r15
  unsigned __int64 v22; // r8
  unsigned __int64 v23; // r11
  int v24; // r14d
  char v25; // dl
  __int64 v26; // rdi
  int v27; // esi
  unsigned int v28; // ecx
  __int64 v29; // r13
  int v30; // r10d
  __int64 v31; // rdx
  unsigned __int64 *v32; // rax
  unsigned __int64 v33; // r14
  __int64 v34; // rax
  __m128i *result; // rax
  unsigned int v36; // esi
  unsigned int v37; // ecx
  int *v38; // rax
  int v39; // edi
  unsigned int v40; // r13d
  int v41; // esi
  int v42; // edx
  unsigned int v43; // esi
  __int64 v44; // rax
  __int64 v45; // rdx
  unsigned __int64 v46; // rax
  __int64 v47; // rdx
  int v48; // ecx
  __int64 v49; // rdi
  int v50; // ecx
  unsigned int v51; // edx
  int v52; // esi
  __int64 v53; // rdi
  int v54; // edx
  unsigned int v55; // ecx
  int v56; // esi
  int v57; // r13d
  int *v58; // r10
  int v59; // edx
  int v60; // r13d
  unsigned __int64 v61; // [rsp+8h] [rbp-98h]
  unsigned __int64 v62; // [rsp+10h] [rbp-90h]
  unsigned __int64 v63; // [rsp+10h] [rbp-90h]
  unsigned __int64 v64; // [rsp+10h] [rbp-90h]
  unsigned __int64 v65; // [rsp+10h] [rbp-90h]
  unsigned __int64 v66; // [rsp+10h] [rbp-90h]
  __int64 v67; // [rsp+18h] [rbp-88h]
  unsigned __int64 v68; // [rsp+18h] [rbp-88h]
  unsigned __int64 v69; // [rsp+18h] [rbp-88h]
  unsigned __int64 v70; // [rsp+18h] [rbp-88h]
  unsigned __int64 v71; // [rsp+18h] [rbp-88h]
  unsigned __int64 v72; // [rsp+18h] [rbp-88h]
  __int32 v73; // [rsp+2Ch] [rbp-74h] BYREF
  __m128i *v74; // [rsp+30h] [rbp-70h] BYREF
  __m128i *v75; // [rsp+38h] [rbp-68h] BYREF
  __m128i v76; // [rsp+40h] [rbp-60h] BYREF
  __m128i v77; // [rsp+50h] [rbp-50h] BYREF
  __int64 v78; // [rsp+60h] [rbp-40h]

  v3 = sub_B10CD0(a2 + 56);
  v4 = *(_BYTE *)(v3 - 16);
  if ( (v4 & 2) != 0 )
  {
    if ( *(_DWORD *)(v3 - 24) != 2 )
    {
LABEL_3:
      v5 = 0;
      goto LABEL_4;
    }
    v19 = *(_QWORD *)(v3 - 32);
  }
  else
  {
    if ( ((*(_WORD *)(v3 - 16) >> 6) & 0xF) != 2 )
      goto LABEL_3;
    v19 = v3 - 16 - 8LL * ((v4 >> 2) & 0xF);
  }
  v5 = *(_QWORD *)(v19 + 8);
LABEL_4:
  v6 = sub_2E891C0(a2);
  v76.m128i_i64[0] = sub_2E89170(a2);
  if ( v6 )
    sub_AF47B0((__int64)&v76.m128i_i64[1], *(unsigned __int64 **)(v6 + 16), *(unsigned __int64 **)(v6 + 24));
  else
    v77.m128i_i8[8] = 0;
  v8 = *(unsigned int *)(a1 + 1144);
  v9 = *(_QWORD *)(a1 + 1128);
  v78 = v5;
  v10 = v8;
  v11 = v9 + 40 * v8;
  v12 = 0;
  if ( v10 )
  {
    v73 = 0;
    if ( v77.m128i_i8[8] )
      v73 = v77.m128i_u16[0] | (v76.m128i_i32[2] << 16);
    v75 = (__m128i *)v5;
    v67 = v11;
    v13 = v10 - 1;
    v74 = (__m128i *)v76.m128i_i64[0];
    v14 = sub_F11290((__int64 *)&v74, &v73, (__int64 *)&v75);
    v15 = 1;
    for ( i = v13 & v14; ; i = v13 & v18 )
    {
      v17 = v9 + 40LL * i;
      if ( *(_QWORD *)v17 == v76.m128i_i64[0] )
      {
        v7 = v77.m128i_u8[8];
        if ( v77.m128i_i8[8] == *(_BYTE *)(v17 + 24) )
        {
          if ( v77.m128i_i8[8] )
          {
            v7 = *(_QWORD *)(v17 + 8);
            if ( v76.m128i_i64[1] != v7 )
              goto LABEL_12;
            v7 = *(_QWORD *)(v17 + 16);
            if ( v77.m128i_i64[0] != v7 )
              goto LABEL_12;
          }
          v7 = *(_QWORD *)(v17 + 32);
          if ( v78 == v7 )
          {
            v12 = v67 != v17;
            break;
          }
        }
      }
      if ( !*(_QWORD *)v17 && !*(_BYTE *)(v17 + 24) && !*(_QWORD *)(v17 + 32) )
      {
        v12 = v67 != *(_QWORD *)(a1 + 1128) + 40LL * *(unsigned int *)(a1 + 1144);
        break;
      }
LABEL_12:
      v18 = v15 + i;
      ++v15;
    }
  }
  v20 = *(_QWORD *)(a2 + 32);
  v21 = v20 + 40;
  if ( *(_WORD *)(a2 + 68) != 14 )
  {
    v21 = v20 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
    v20 += 80;
  }
  if ( v21 != v20 )
  {
    v22 = (4LL * v12) | a2 & 0xFFFFFFFFFFFFFFFBLL;
    v23 = (4LL * v12) | a2 & 0xFFFFFFFFFFFFFFF9LL;
    while ( 1 )
    {
      if ( *(_BYTE *)v20 )
        goto LABEL_23;
      v24 = *(_DWORD *)(v20 + 8);
      if ( v24 >= 0 )
        goto LABEL_23;
      v25 = *(_BYTE *)(a1 + 1048) & 1;
      if ( v25 )
      {
        v26 = a1 + 1056;
        v27 = 3;
      }
      else
      {
        v36 = *(_DWORD *)(a1 + 1064);
        v26 = *(_QWORD *)(a1 + 1056);
        if ( !v36 )
        {
          v37 = *(_DWORD *)(a1 + 1048);
          v38 = 0;
          ++*(_QWORD *)(a1 + 1040);
          v39 = (v37 >> 1) + 1;
          goto LABEL_40;
        }
        v27 = v36 - 1;
      }
      v28 = v27 & (37 * v24);
      v29 = v26 + 16LL * v28;
      v30 = *(_DWORD *)v29;
      if ( v24 != *(_DWORD *)v29 )
        break;
LABEL_29:
      v31 = *(_QWORD *)(v29 + 8);
      v32 = (unsigned __int64 *)(v29 + 8);
      v33 = v31 & 0xFFFFFFFFFFFFFFFCLL;
      if ( (v31 & 0xFFFFFFFFFFFFFFFCLL) != 0 )
      {
        if ( (v31 & 2) == 0 )
        {
          v62 = v23;
          v68 = v22;
          v44 = sub_22077B0(0x30u);
          v22 = v68;
          v23 = v62;
          v7 = 0x400000000LL;
          if ( v44 )
          {
            *(_QWORD *)(v44 + 8) = 0x400000000LL;
            *(_QWORD *)v44 = v44 + 16;
          }
          v45 = v44;
          v46 = v44 & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)(v29 + 8) = v45 | 2;
          v47 = *(unsigned int *)(v46 + 8);
          if ( v47 + 1 > (unsigned __int64)*(unsigned int *)(v46 + 12) )
          {
            v61 = v62;
            v66 = v68;
            v72 = v46;
            sub_C8D5F0(v46, (const void *)(v46 + 16), v47 + 1, 8u, v22, 0x400000000LL);
            v46 = v72;
            v23 = v61;
            v22 = v66;
            v47 = *(unsigned int *)(v72 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v46 + 8 * v47) = v33;
          ++*(_DWORD *)(v46 + 8);
          v33 = *(_QWORD *)(v29 + 8) & 0xFFFFFFFFFFFFFFFCLL;
        }
        v34 = *(unsigned int *)(v33 + 8);
        if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(v33 + 12) )
        {
          v63 = v23;
          v69 = v22;
          sub_C8D5F0(v33, (const void *)(v33 + 16), v34 + 1, 8u, v22, v7);
          v34 = *(unsigned int *)(v33 + 8);
          v23 = v63;
          v22 = v69;
        }
        v20 += 40;
        *(_QWORD *)(*(_QWORD *)v33 + 8 * v34) = v22;
        ++*(_DWORD *)(v33 + 8);
        if ( v21 == v20 )
          goto LABEL_34;
      }
      else
      {
LABEL_46:
        *v32 = v23;
LABEL_23:
        v20 += 40;
        if ( v21 == v20 )
          goto LABEL_34;
      }
    }
    v7 = 1;
    v38 = 0;
    while ( v30 != -1 )
    {
      if ( v30 == -2 && !v38 )
        v38 = (int *)v29;
      v28 = v27 & (v7 + v28);
      v29 = v26 + 16LL * v28;
      v30 = *(_DWORD *)v29;
      if ( v24 == *(_DWORD *)v29 )
        goto LABEL_29;
      v7 = (unsigned int)(v7 + 1);
    }
    v37 = *(_DWORD *)(a1 + 1048);
    v36 = 4;
    if ( !v38 )
      v38 = (int *)v29;
    v40 = 12;
    ++*(_QWORD *)(a1 + 1040);
    v39 = (v37 >> 1) + 1;
    if ( !v25 )
    {
      v36 = *(_DWORD *)(a1 + 1064);
LABEL_40:
      v40 = 3 * v36;
    }
    if ( 4 * v39 >= v40 )
    {
      v64 = v23;
      v70 = v22;
      sub_2ED9250(a1 + 1040, 2 * v36);
      v22 = v70;
      v23 = v64;
      if ( (*(_BYTE *)(a1 + 1048) & 1) != 0 )
      {
        v49 = a1 + 1056;
        v50 = 3;
      }
      else
      {
        v48 = *(_DWORD *)(a1 + 1064);
        v49 = *(_QWORD *)(a1 + 1056);
        if ( !v48 )
          goto LABEL_108;
        v50 = v48 - 1;
      }
      v51 = v50 & (37 * v24);
      v38 = (int *)(v49 + 16LL * v51);
      v52 = *v38;
      if ( v24 == *v38 )
        goto LABEL_74;
      v60 = 1;
      v58 = 0;
      while ( v52 != -1 )
      {
        if ( !v58 && v52 == -2 )
          v58 = v38;
        v51 = v50 & (v60 + v51);
        v7 = (unsigned int)(v60 + 1);
        v38 = (int *)(v49 + 16LL * v51);
        v52 = *v38;
        if ( v24 == *v38 )
          goto LABEL_74;
        ++v60;
      }
    }
    else
    {
      if ( v36 - *(_DWORD *)(a1 + 1052) - v39 > v36 >> 3 )
      {
LABEL_43:
        *(_DWORD *)(a1 + 1048) = (2 * (v37 >> 1) + 2) | v37 & 1;
        if ( *v38 != -1 )
          --*(_DWORD *)(a1 + 1052);
        *v38 = v24;
        v32 = (unsigned __int64 *)(v38 + 2);
        *v32 = 0;
        goto LABEL_46;
      }
      v65 = v23;
      v71 = v22;
      sub_2ED9250(a1 + 1040, v36);
      v22 = v71;
      v23 = v65;
      if ( (*(_BYTE *)(a1 + 1048) & 1) != 0 )
      {
        v53 = a1 + 1056;
        v54 = 3;
      }
      else
      {
        v59 = *(_DWORD *)(a1 + 1064);
        v53 = *(_QWORD *)(a1 + 1056);
        if ( !v59 )
        {
LABEL_108:
          *(_DWORD *)(a1 + 1048) = (2 * (*(_DWORD *)(a1 + 1048) >> 1) + 2) | *(_DWORD *)(a1 + 1048) & 1;
          BUG();
        }
        v54 = v59 - 1;
      }
      v55 = v54 & (37 * v24);
      v38 = (int *)(v53 + 16LL * v55);
      v56 = *v38;
      if ( v24 == *v38 )
      {
LABEL_74:
        v37 = *(_DWORD *)(a1 + 1048);
        goto LABEL_43;
      }
      v57 = 1;
      v58 = 0;
      while ( v56 != -1 )
      {
        if ( !v58 && v56 == -2 )
          v58 = v38;
        v55 = v54 & (v57 + v55);
        v7 = (unsigned int)(v57 + 1);
        v38 = (int *)(v53 + 16LL * v55);
        v56 = *v38;
        if ( v24 == *v38 )
          goto LABEL_74;
        ++v57;
      }
    }
    if ( v58 )
      v38 = v58;
    goto LABEL_74;
  }
LABEL_34:
  result = (__m128i *)sub_F38F60(a1 + 1120, (__int64)&v76, (__int64 *)&v74);
  if ( !(_BYTE)result )
  {
    v41 = *(_DWORD *)(a1 + 1136);
    result = v74;
    ++*(_QWORD *)(a1 + 1120);
    v42 = v41 + 1;
    v43 = *(_DWORD *)(a1 + 1144);
    v75 = result;
    if ( 4 * v42 >= 3 * v43 )
    {
      v43 *= 2;
    }
    else if ( v43 - *(_DWORD *)(a1 + 1140) - v42 > v43 >> 3 )
    {
      goto LABEL_51;
    }
    sub_F3D0B0(a1 + 1120, v43);
    sub_F38F60(a1 + 1120, (__int64)&v76, (__int64 *)&v75);
    v42 = *(_DWORD *)(a1 + 1136) + 1;
    result = v75;
LABEL_51:
    *(_DWORD *)(a1 + 1136) = v42;
    if ( result->m128i_i64[0] || result[1].m128i_i8[8] || result[2].m128i_i64[0] )
      --*(_DWORD *)(a1 + 1140);
    *result = _mm_loadu_si128(&v76);
    result[1] = _mm_loadu_si128(&v77);
    result[2].m128i_i64[0] = v78;
  }
  return result;
}
