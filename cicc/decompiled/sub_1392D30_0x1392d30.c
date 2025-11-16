// Function: sub_1392D30
// Address: 0x1392d30
//
__int64 __fastcall sub_1392D30(__int64 a1, __m128i **a2)
{
  __m128i **v2; // r11
  __int64 v3; // rbx
  int *v4; // r12
  int *v5; // r15
  __m128i *v6; // r13
  __int64 v7; // rdi
  const __m128i *v9; // r11
  int v10; // ebx
  unsigned int v11; // r9d
  _DWORD *v12; // r8
  int v13; // ecx
  __m128i *v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // r9
  _DWORD *v17; // rsi
  unsigned int v18; // eax
  _DWORD *v19; // rdx
  unsigned int v20; // edx
  int *v21; // r8
  int v22; // r8d
  int v23; // edx
  unsigned int v24; // r8d
  int *v25; // rsi
  int v26; // r9d
  __int64 v27; // rsi
  __int64 v28; // r8
  _DWORD *v29; // rsi
  unsigned int v30; // eax
  _DWORD *v31; // rdx
  unsigned int v32; // edx
  int *v33; // r9
  int v34; // r9d
  int v35; // edx
  unsigned int v36; // r8d
  int *v37; // rsi
  int v38; // r9d
  int v40; // esi
  int v41; // esi
  __int64 v42; // rax
  __int64 v43; // r10
  __int64 v44; // r9
  __int64 v45; // rcx
  _DWORD *v46; // rsi
  unsigned int v47; // eax
  _DWORD *v48; // rdx
  unsigned int v49; // edx
  int *v50; // r8
  int v51; // r8d
  int v52; // edx
  unsigned int v53; // esi
  int *v54; // rcx
  int v55; // r8d
  int v56; // ecx
  int v57; // r12d
  _DWORD *v58; // rax
  int v59; // ecx
  unsigned int v60; // edx
  int v61; // r8d
  int v62; // edi
  _DWORD *v63; // rsi
  unsigned int v64; // edx
  int v65; // edi
  int v66; // r8d
  int v67; // r11d
  int v68; // r11d
  int v69; // [rsp+8h] [rbp-68h]
  const __m128i *v70; // [rsp+8h] [rbp-68h]
  const __m128i *v71; // [rsp+8h] [rbp-68h]
  __int64 v72; // [rsp+18h] [rbp-58h]
  __int64 v73; // [rsp+20h] [rbp-50h] BYREF
  __int64 v74; // [rsp+28h] [rbp-48h]
  __int64 v75; // [rsp+30h] [rbp-40h]
  unsigned int v76; // [rsp+38h] [rbp-38h]

  v2 = a2;
  v3 = a1;
  v4 = *(int **)(a1 + 32);
  v5 = *(int **)(a1 + 40);
  v73 = 0;
  v74 = 0;
  v6 = a2[1];
  v7 = 0;
  v75 = 0;
  v76 = 0;
  if ( v4 != v5 )
  {
    v72 = v3;
    while ( 1 )
    {
      while ( v4[6] != -1 )
      {
LABEL_3:
        v4 += 8;
        if ( v5 == v4 )
          goto LABEL_11;
      }
      v9 = *a2;
      v10 = *v4;
      if ( v76 )
      {
        v11 = (v76 - 1) & (37 * v10);
        v12 = (_DWORD *)(v7 + 8LL * v11);
        v13 = *v12;
        if ( v10 == *v12 )
          goto LABEL_7;
        v69 = 1;
        v58 = 0;
        while ( v13 != -1 )
        {
          if ( v58 || v13 != -2 )
            v12 = v58;
          v11 = (v76 - 1) & (v69 + v11);
          v13 = *(_DWORD *)(v7 + 8LL * v11);
          if ( v10 == v13 )
            goto LABEL_7;
          ++v69;
          v58 = v12;
          v12 = (_DWORD *)(v7 + 8LL * v11);
        }
        if ( !v58 )
          v58 = v12;
        ++v73;
        v59 = v75 + 1;
        if ( 4 * ((int)v75 + 1) < 3 * v76 )
        {
          if ( v76 - HIDWORD(v75) - v59 > v76 >> 3 )
            goto LABEL_79;
          v71 = v9;
          sub_1392B70((__int64)&v73, v76);
          if ( !v76 )
          {
LABEL_114:
            LODWORD(v75) = v75 + 1;
            BUG();
          }
          v63 = 0;
          v9 = v71;
          v64 = (v76 - 1) & (37 * v10);
          v59 = v75 + 1;
          v65 = 1;
          v58 = (_DWORD *)(v74 + 8LL * v64);
          v66 = *v58;
          if ( v10 == *v58 )
            goto LABEL_79;
          while ( v66 != -1 )
          {
            if ( v66 == -2 && !v63 )
              v63 = v58;
            v64 = (v76 - 1) & (v65 + v64);
            v58 = (_DWORD *)(v74 + 8LL * v64);
            v66 = *v58;
            if ( v10 == *v58 )
              goto LABEL_79;
            ++v65;
          }
          goto LABEL_97;
        }
      }
      else
      {
        ++v73;
      }
      v70 = v9;
      sub_1392B70((__int64)&v73, 2 * v76);
      if ( !v76 )
        goto LABEL_114;
      v9 = v70;
      v60 = (v76 - 1) & (37 * v10);
      v59 = v75 + 1;
      v58 = (_DWORD *)(v74 + 8LL * v60);
      v61 = *v58;
      if ( v10 == *v58 )
        goto LABEL_79;
      v62 = 1;
      v63 = 0;
      while ( v61 != -1 )
      {
        if ( !v63 && v61 == -2 )
          v63 = v58;
        v60 = (v76 - 1) & (v62 + v60);
        v58 = (_DWORD *)(v74 + 8LL * v60);
        v61 = *v58;
        if ( v10 == *v58 )
          goto LABEL_79;
        ++v62;
      }
LABEL_97:
      if ( v63 )
        v58 = v63;
LABEL_79:
      LODWORD(v75) = v59;
      if ( *v58 != -1 )
        --HIDWORD(v75);
      *v58 = v10;
      v58[1] = v6 - v9;
      v6 = a2[1];
LABEL_7:
      if ( a2[2] == v6 )
      {
        sub_138FE20((const __m128i **)a2, v6, (const __m128i *)(v4 + 2));
        v6 = a2[1];
        v7 = v74;
        goto LABEL_3;
      }
      if ( v6 )
      {
        *v6 = _mm_loadu_si128((const __m128i *)(v4 + 2));
        v6 = a2[1];
      }
      ++v6;
      v4 += 8;
      v7 = v74;
      a2[1] = v6;
      if ( v5 == v4 )
      {
LABEL_11:
        v3 = v72;
        v2 = a2;
        break;
      }
    }
  }
  v14 = *v2;
  if ( *v2 != v6 )
  {
    while ( 1 )
    {
      v15 = v14->m128i_u32[0];
      if ( (_DWORD)v15 == -1 )
        goto LABEL_23;
      v16 = *(_QWORD *)(v3 + 32);
      v17 = (_DWORD *)(v16 + 32 * v15);
      v18 = v17[6];
      v19 = v17;
      if ( v18 != -1 )
      {
        v20 = v17[6];
        do
        {
          v21 = (int *)(v16 + 32LL * v20);
          v20 = v21[6];
        }
        while ( v20 != -1 );
        v22 = *v21;
        while ( 1 )
        {
          v17[6] = v22;
          v19 = (_DWORD *)(v16 + 32LL * v18);
          v18 = v19[6];
          if ( v18 == -1 )
            break;
          v16 = *(_QWORD *)(v3 + 32);
          v17 = v19;
        }
      }
      if ( !v76 )
        goto LABEL_38;
      v23 = *v19;
      v24 = (v76 - 1) & (37 * v23);
      v25 = (int *)(v7 + 8LL * v24);
      v26 = *v25;
      if ( *v25 != v23 )
        break;
LABEL_22:
      v14->m128i_i32[0] = v25[1];
      v7 = v74;
LABEL_23:
      v27 = v14->m128i_u32[1];
      if ( (_DWORD)v27 != -1 )
      {
        v28 = *(_QWORD *)(v3 + 32);
        v29 = (_DWORD *)(v28 + 32 * v27);
        v30 = v29[6];
        v31 = v29;
        if ( v30 != -1 )
        {
          v32 = v29[6];
          do
          {
            v33 = (int *)(v28 + 32LL * v32);
            v32 = v33[6];
          }
          while ( v32 != -1 );
          v34 = *v33;
          while ( 1 )
          {
            v29[6] = v34;
            v31 = (_DWORD *)(v28 + 32LL * v30);
            v30 = v31[6];
            if ( v30 == -1 )
              break;
            v28 = *(_QWORD *)(v3 + 32);
            v29 = v31;
          }
        }
        if ( v76 )
        {
          v35 = *v31;
          v36 = (v76 - 1) & (37 * v35);
          v37 = (int *)(v7 + 8LL * v36);
          v38 = *v37;
          if ( v35 == *v37 )
          {
LABEL_32:
            v14->m128i_i32[1] = v37[1];
            v7 = v74;
            goto LABEL_33;
          }
          v41 = 1;
          while ( v38 != -1 )
          {
            v67 = v41 + 1;
            v36 = (v76 - 1) & (v41 + v36);
            v37 = (int *)(v7 + 8LL * v36);
            v38 = *v37;
            if ( v35 == *v37 )
              goto LABEL_32;
            v41 = v67;
          }
        }
        v37 = (int *)(v7 + 8LL * v76);
        goto LABEL_32;
      }
LABEL_33:
      if ( v6 == ++v14 )
        goto LABEL_34;
    }
    v40 = 1;
    while ( v26 != -1 )
    {
      v68 = v40 + 1;
      v24 = (v76 - 1) & (v40 + v24);
      v25 = (int *)(v7 + 8LL * v24);
      v26 = *v25;
      if ( v23 == *v25 )
        goto LABEL_22;
      v40 = v68;
    }
LABEL_38:
    v25 = (int *)(v7 + 8LL * v76);
    goto LABEL_22;
  }
LABEL_34:
  if ( *(_DWORD *)(v3 + 16) )
  {
    v42 = *(_QWORD *)(v3 + 8);
    v43 = v42 + 24LL * *(unsigned int *)(v3 + 24);
    if ( v42 != v43 )
    {
      while ( 1 )
      {
        v44 = v42;
        if ( *(_QWORD *)v42 != -8 )
          break;
        if ( *(_DWORD *)(v42 + 8) != -1 )
          goto LABEL_46;
LABEL_69:
        v42 += 24;
        if ( v43 == v42 )
          return j___libc_free_0(v7);
      }
      if ( *(_QWORD *)v42 == -16 && *(_DWORD *)(v42 + 8) == -2 )
        goto LABEL_69;
LABEL_46:
      if ( v43 != v42 )
      {
LABEL_47:
        v45 = *(_QWORD *)(v3 + 32);
        v46 = (_DWORD *)(v45 + 32LL * *(unsigned int *)(v44 + 16));
        v47 = v46[6];
        v48 = v46;
        if ( v47 != -1 )
        {
          v49 = v46[6];
          do
          {
            v50 = (int *)(v45 + 32LL * v49);
            v49 = v50[6];
          }
          while ( v49 != -1 );
          v51 = *v50;
          while ( 1 )
          {
            v46[6] = v51;
            v48 = (_DWORD *)(v45 + 32LL * v47);
            v47 = v48[6];
            if ( v47 == -1 )
              break;
            v45 = *(_QWORD *)(v3 + 32);
            v46 = v48;
          }
        }
        if ( v76 )
        {
          v52 = *v48;
          v53 = (v76 - 1) & (37 * v52);
          v54 = (int *)(v7 + 8LL * v53);
          v55 = *v54;
          if ( *v54 == v52 )
            goto LABEL_55;
          v56 = 1;
          while ( v55 != -1 )
          {
            v57 = v56 + 1;
            v53 = (v76 - 1) & (v56 + v53);
            v54 = (int *)(v7 + 8LL * v53);
            v55 = *v54;
            if ( v52 == *v54 )
              goto LABEL_55;
            v56 = v57;
          }
        }
        v54 = (int *)(v7 + 8LL * v76);
LABEL_55:
        v44 += 24;
        *(_DWORD *)(v44 - 8) = v54[1];
        if ( v44 == v43 )
          return j___libc_free_0(v7);
        do
        {
          if ( *(_QWORD *)v44 == -8 )
          {
            if ( *(_DWORD *)(v44 + 8) != -1 )
              goto LABEL_58;
          }
          else if ( *(_QWORD *)v44 != -16 || *(_DWORD *)(v44 + 8) != -2 )
          {
LABEL_58:
            if ( v44 == v43 )
              return j___libc_free_0(v7);
            goto LABEL_47;
          }
          v44 += 24;
        }
        while ( v43 != v44 );
      }
    }
  }
  return j___libc_free_0(v7);
}
