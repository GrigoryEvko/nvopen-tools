// Function: sub_210AD80
// Address: 0x210ad80
//
__int64 __fastcall sub_210AD80(__int64 *a1, _QWORD *a2)
{
  __int64 *v2; // r12
  char v3; // al
  char v4; // bl
  __int64 *v5; // r13
  unsigned int v6; // r14d
  __int64 v7; // r15
  int v8; // r8d
  __int64 v9; // r10
  __int64 v10; // rax
  unsigned int v11; // r9d
  __int64 *v12; // rax
  unsigned int v13; // eax
  int v14; // edx
  unsigned __int64 v15; // rdi
  __int64 *v17; // r11
  unsigned int v18; // esi
  __int64 v19; // rbx
  __int64 v20; // rdi
  __int64 v21; // r15
  __int64 *v22; // r14
  unsigned int v23; // edx
  __int64 v24; // rax
  __int64 v25; // r8
  _BYTE *v26; // r13
  int v27; // edx
  unsigned int v28; // ecx
  __int64 v29; // r8
  __int64 v30; // rdx
  __int64 v31; // rdi
  __int64 *v32; // r14
  __int64 v33; // rbx
  __int64 v34; // rdx
  __int64 v35; // r13
  __int64 v36; // r12
  __int64 v37; // r15
  __int32 v38; // edx
  __int64 v39; // rdx
  int v40; // r8d
  int v41; // r9d
  __int64 v42; // rbx
  __int64 v43; // rax
  int v44; // r11d
  unsigned int v45; // ebx
  unsigned int v46; // r8d
  __int64 v47; // rax
  __int64 v48; // rcx
  int v49; // eax
  __int64 v50; // rax
  int v51; // r15d
  __int64 v52; // r12
  unsigned int v53; // eax
  int v54; // ecx
  __int64 v55; // rdx
  __int64 v56; // r8
  __int64 v57; // r13
  size_t v58; // rbx
  __int64 v59; // r12
  __int64 *v60; // rax
  __int64 v61; // rdx
  int v62; // r10d
  __int64 v63; // r8
  unsigned int v64; // r13d
  int v65; // r9d
  __int64 v66; // rsi
  int v67; // r11d
  __int64 v68; // r10
  __int64 v69; // r11
  int v70; // r10d
  unsigned int v71; // ecx
  __int64 v72; // r8
  int v73; // r10d
  __int64 v74; // r9
  int v75; // r10d
  int v76; // edx
  __int64 v77; // r10
  __int64 v78; // [rsp+8h] [rbp-118h]
  int v79; // [rsp+8h] [rbp-118h]
  int v80; // [rsp+8h] [rbp-118h]
  __int64 *v81; // [rsp+10h] [rbp-110h]
  unsigned int v82; // [rsp+10h] [rbp-110h]
  __int64 *v84; // [rsp+28h] [rbp-F8h]
  __int64 *v85; // [rsp+28h] [rbp-F8h]
  __int64 *v86; // [rsp+28h] [rbp-F8h]
  __m128i v87; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v88; // [rsp+40h] [rbp-E0h]
  __int64 v89; // [rsp+48h] [rbp-D8h]
  __int64 v90; // [rsp+50h] [rbp-D0h]
  _BYTE *v91; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v92; // [rsp+68h] [rbp-B8h]
  _BYTE v93[176]; // [rsp+70h] [rbp-B0h] BYREF

  v2 = a1;
  v3 = sub_21076B0((__int64)a1, (__int64)a2);
  if ( v3 )
  {
    v4 = v3;
    v5 = (__int64 *)a2[8];
    v84 = (__int64 *)a2[9];
    if ( v84 == v5 )
    {
      v57 = a1[4];
      v58 = a1[5];
      v59 = a1[2];
      v60 = (__int64 *)sub_1DD5EE0((__int64)a2);
      sub_21072E0(9u, (__int64)a2, v60, v59, v58, v57);
      return *(unsigned int *)(*(_QWORD *)(v61 + 32) + 8LL);
    }
    v6 = 0;
    v91 = v93;
    v92 = 0x800000000LL;
    do
    {
      while ( 1 )
      {
        v7 = *v5;
        v9 = (unsigned int)sub_210AA20(a1, *v5);
        v10 = (unsigned int)v92;
        v11 = v9;
        if ( (unsigned int)v92 >= HIDWORD(v92) )
        {
          v78 = v9;
          v82 = v9;
          sub_16CD150((__int64)&v91, v93, 0, 16, v8, v9);
          v10 = (unsigned int)v92;
          v9 = v78;
          v11 = v82;
        }
        v12 = (__int64 *)&v91[16 * v10];
        *v12 = v7;
        v12[1] = v9;
        v13 = v92;
        v14 = v92 + 1;
        LODWORD(v92) = v92 + 1;
        if ( !v4 )
          break;
        v6 = v11;
        ++v5;
        v4 = 0;
        if ( v84 == v5 )
          goto LABEL_11;
      }
      if ( v11 != v6 )
        v6 = 0;
      ++v5;
      v4 = 0;
    }
    while ( v84 != v5 );
LABEL_11:
    if ( v6 )
      goto LABEL_12;
    v85 = a2 + 3;
    v17 = a2 + 3;
    if ( a2 + 3 == (_QWORD *)(a2[3] & 0xFFFFFFFFFFFFFFF8LL) )
      goto LABEL_39;
    v17 = (__int64 *)a2[4];
    if ( *(_WORD *)v17[2] )
    {
      if ( *(_WORD *)v17[2] != 45 )
        goto LABEL_39;
    }
    v87 = 0u;
    v88 = 0;
    LODWORD(v89) = 0;
    if ( v14 )
    {
      v18 = 0;
      v19 = 0;
      v20 = 0;
      v21 = 16 * (v13 + 1LL);
      v22 = v17;
      while ( 1 )
      {
        v26 = &v91[v19];
        if ( !v18 )
          break;
        v23 = (v18 - 1) & (((unsigned int)*(_QWORD *)v26 >> 9) ^ ((unsigned int)*(_QWORD *)v26 >> 4));
        v24 = v20 + 16LL * v23;
        v25 = *(_QWORD *)v24;
        if ( *(_QWORD *)v24 == *(_QWORD *)v26 )
        {
LABEL_21:
          v19 += 16;
          *(_DWORD *)(v24 + 8) = *((_DWORD *)v26 + 2);
          if ( v21 == v19 )
            goto LABEL_30;
          goto LABEL_22;
        }
        v67 = 1;
        v68 = 0;
        while ( v25 != -8 )
        {
          if ( v25 == -16 && !v68 )
            v68 = v24;
          v23 = (v18 - 1) & (v67 + v23);
          v24 = v20 + 16LL * v23;
          v25 = *(_QWORD *)v24;
          if ( *(_QWORD *)v26 == *(_QWORD *)v24 )
            goto LABEL_21;
          ++v67;
        }
        if ( v68 )
          v24 = v68;
        ++v87.m128i_i64[0];
        v27 = v88 + 1;
        if ( 4 * ((int)v88 + 1) >= 3 * v18 )
          goto LABEL_25;
        if ( v18 - (v27 + HIDWORD(v88)) <= v18 >> 3 )
        {
          sub_1DA35E0((__int64)&v87, v18);
          if ( !(_DWORD)v89 )
          {
LABEL_138:
            LODWORD(v88) = v88 + 1;
            BUG();
          }
          v69 = 0;
          v70 = 1;
          v27 = v88 + 1;
          v71 = (v89 - 1) & (((unsigned int)*(_QWORD *)v26 >> 9) ^ ((unsigned int)*(_QWORD *)v26 >> 4));
          v24 = v87.m128i_i64[1] + 16LL * v71;
          v72 = *(_QWORD *)v24;
          if ( *(_QWORD *)v26 != *(_QWORD *)v24 )
          {
            while ( v72 != -8 )
            {
              if ( !v69 && v72 == -16 )
                v69 = v24;
              v71 = (v89 - 1) & (v70 + v71);
              v24 = v87.m128i_i64[1] + 16LL * v71;
              v72 = *(_QWORD *)v24;
              if ( *(_QWORD *)v26 == *(_QWORD *)v24 )
                goto LABEL_27;
              ++v70;
            }
            goto LABEL_100;
          }
        }
LABEL_27:
        LODWORD(v88) = v27;
        if ( *(_QWORD *)v24 != -8 )
          --HIDWORD(v88);
        v30 = *(_QWORD *)v26;
        *(_DWORD *)(v24 + 8) = 0;
        v19 += 16;
        *(_QWORD *)v24 = v30;
        *(_DWORD *)(v24 + 8) = *((_DWORD *)v26 + 2);
        if ( v21 == v19 )
        {
LABEL_30:
          v31 = v87.m128i_i64[1];
          v17 = v22;
          goto LABEL_31;
        }
LABEL_22:
        v20 = v87.m128i_i64[1];
        v18 = v89;
      }
      ++v87.m128i_i64[0];
LABEL_25:
      sub_1DA35E0((__int64)&v87, 2 * v18);
      if ( !(_DWORD)v89 )
        goto LABEL_138;
      v27 = v88 + 1;
      v28 = (v89 - 1) & (((unsigned int)*(_QWORD *)v26 >> 9) ^ ((unsigned int)*(_QWORD *)v26 >> 4));
      v24 = v87.m128i_i64[1] + 16LL * v28;
      v29 = *(_QWORD *)v24;
      if ( *(_QWORD *)v26 != *(_QWORD *)v24 )
      {
        v75 = 1;
        v69 = 0;
        while ( v29 != -8 )
        {
          if ( !v69 && v29 == -16 )
            v69 = v24;
          v28 = (v89 - 1) & (v75 + v28);
          v24 = v87.m128i_i64[1] + 16LL * v28;
          v29 = *(_QWORD *)v24;
          if ( *(_QWORD *)v26 == *(_QWORD *)v24 )
            goto LABEL_27;
          ++v75;
        }
LABEL_100:
        if ( v69 )
          v24 = v69;
        goto LABEL_27;
      }
      goto LABEL_27;
    }
    v31 = 0;
LABEL_31:
    if ( v85 == v17 )
      goto LABEL_36;
    v81 = v2;
    v32 = v17;
LABEL_33:
    if ( *(_WORD *)v32[2] && *(_WORD *)v32[2] != 45 )
      goto LABEL_35;
    v44 = *((_DWORD *)v32 + 10);
    if ( v44 == 1 )
    {
LABEL_91:
      v2 = v81;
      v6 = *(_DWORD *)(v32[4] + 8);
      j___libc_free_0(v31);
      if ( !v6 )
        goto LABEL_37;
      goto LABEL_12;
    }
    v45 = 1;
    while ( 1 )
    {
      v50 = v32[4];
      v51 = *(_DWORD *)(v50 + 40LL * v45 + 8);
      v52 = *(_QWORD *)(v50 + 40LL * (v45 + 1) + 24);
      if ( !(_DWORD)v89 )
        break;
      v46 = (v89 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
      v47 = v31 + 16LL * v46;
      v48 = *(_QWORD *)v47;
      if ( v52 == *(_QWORD *)v47 )
      {
        v49 = *(_DWORD *)(v47 + 8);
        goto LABEL_55;
      }
      v62 = 1;
      v55 = 0;
      while ( v48 != -8 )
      {
        if ( v55 || v48 != -16 )
          v47 = v55;
        v76 = v62 + 1;
        v46 = (v89 - 1) & (v62 + v46);
        v77 = v31 + 16LL * v46;
        v48 = *(_QWORD *)v77;
        if ( v52 == *(_QWORD *)v77 )
        {
          v49 = *(_DWORD *)(v77 + 8);
          goto LABEL_55;
        }
        v62 = v76;
        v55 = v47;
        v47 = v31 + 16LL * v46;
      }
      if ( !v55 )
        v55 = v47;
      ++v87.m128i_i64[0];
      v54 = v88 + 1;
      if ( 4 * ((int)v88 + 1) >= (unsigned int)(3 * v89) )
        goto LABEL_59;
      if ( (int)v89 - HIDWORD(v88) - v54 <= (unsigned int)v89 >> 3 )
      {
        v80 = v44;
        sub_1DA35E0((__int64)&v87, v89);
        if ( !(_DWORD)v89 )
          goto LABEL_138;
        v63 = 0;
        v64 = (v89 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
        v44 = v80;
        v65 = 1;
        v54 = v88 + 1;
        v55 = v87.m128i_i64[1] + 16LL * v64;
        v66 = *(_QWORD *)v55;
        if ( v52 != *(_QWORD *)v55 )
        {
          while ( v66 != -8 )
          {
            if ( v66 == -16 && !v63 )
              v63 = v55;
            v64 = (v89 - 1) & (v65 + v64);
            v55 = v87.m128i_i64[1] + 16LL * v64;
            v66 = *(_QWORD *)v55;
            if ( v52 == *(_QWORD *)v55 )
              goto LABEL_61;
            ++v65;
          }
          if ( v63 )
            v55 = v63;
        }
      }
LABEL_61:
      LODWORD(v88) = v54;
      if ( *(_QWORD *)v55 != -8 )
        --HIDWORD(v88);
      *(_QWORD *)v55 = v52;
      v49 = 0;
      *(_DWORD *)(v55 + 8) = 0;
LABEL_55:
      if ( v51 != v49 )
      {
        if ( (*(_BYTE *)v32 & 4) == 0 )
        {
          while ( (*((_BYTE *)v32 + 46) & 8) != 0 )
            v32 = (__int64 *)v32[1];
        }
        v32 = (__int64 *)v32[1];
        v31 = v87.m128i_i64[1];
        if ( v85 == v32 )
        {
LABEL_35:
          v2 = v81;
LABEL_36:
          j___libc_free_0(v31);
LABEL_37:
          if ( v85 == (__int64 *)(a2[3] & 0xFFFFFFFFFFFFFFF8LL) )
            v17 = v85;
          else
            v17 = (__int64 *)a2[4];
LABEL_39:
          v33 = sub_21072E0(0, (__int64)a2, v17, v2[2], v2[5], v2[4]);
          v35 = v34;
          if ( (_DWORD)v92 )
          {
            v86 = v2;
            v36 = 0;
            v37 = 16LL * (unsigned int)v92;
            do
            {
              v38 = *(_DWORD *)&v91[v36 + 8];
              v87.m128i_i64[0] = 0;
              v88 = 0;
              v87.m128i_i32[2] = v38;
              v89 = 0;
              v90 = 0;
              sub_1E1A9C0(v35, v33, &v87);
              v39 = *(_QWORD *)&v91[v36];
              v36 += 16;
              v87.m128i_i8[0] = 4;
              v88 = 0;
              v89 = v39;
              v87.m128i_i32[0] &= 0xFFF000FF;
              sub_1E1A9C0(v35, v33, &v87);
            }
            while ( v37 != v36 );
            v2 = v86;
          }
          v6 = sub_1E17820(v35);
          if ( !v6 )
          {
            v42 = v2[3];
            if ( v42 )
            {
              v43 = *(unsigned int *)(v42 + 8);
              if ( (unsigned int)v43 >= *(_DWORD *)(v42 + 12) )
              {
                sub_16CD150(v2[3], (const void *)(v42 + 16), 0, 8, v40, v41);
                v43 = *(unsigned int *)(v42 + 8);
              }
              *(_QWORD *)(*(_QWORD *)v42 + 8 * v43) = v35;
              ++*(_DWORD *)(v42 + 8);
            }
            v15 = (unsigned __int64)v91;
            v6 = *(_DWORD *)(*(_QWORD *)(v35 + 32) + 8LL);
            if ( v91 == v93 )
              return v6;
LABEL_13:
            _libc_free(v15);
            return v6;
          }
          sub_1E16240(v35);
LABEL_12:
          v15 = (unsigned __int64)v91;
          if ( v91 == v93 )
            return v6;
          goto LABEL_13;
        }
        goto LABEL_33;
      }
      v45 += 2;
      v31 = v87.m128i_i64[1];
      if ( v44 == v45 )
        goto LABEL_91;
    }
    ++v87.m128i_i64[0];
LABEL_59:
    v79 = v44;
    sub_1DA35E0((__int64)&v87, 2 * v89);
    if ( !(_DWORD)v89 )
      goto LABEL_138;
    v44 = v79;
    v53 = (v89 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
    v54 = v88 + 1;
    v55 = v87.m128i_i64[1] + 16LL * v53;
    v56 = *(_QWORD *)v55;
    if ( v52 != *(_QWORD *)v55 )
    {
      v73 = 1;
      v74 = 0;
      while ( v56 != -8 )
      {
        if ( v56 == -16 && !v74 )
          v74 = v55;
        v53 = (v89 - 1) & (v73 + v53);
        v55 = v87.m128i_i64[1] + 16LL * v53;
        v56 = *(_QWORD *)v55;
        if ( v52 == *(_QWORD *)v55 )
          goto LABEL_61;
        ++v73;
      }
      if ( v74 )
        v55 = v74;
    }
    goto LABEL_61;
  }
  return sub_210AA20(a1, (__int64)a2);
}
