// Function: sub_21FBAD0
// Address: 0x21fbad0
//
void __fastcall sub_21FBAD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  unsigned int v6; // eax
  __int64 v7; // rdx
  unsigned int v8; // r15d
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // rax
  int v15; // r9d
  unsigned int v16; // edi
  unsigned __int32 v17; // r12d
  int v18; // r15d
  unsigned int v19; // r8d
  __int64 v20; // rdx
  __int16 ***v21; // rdx
  _BYTE *v22; // r12
  int v23; // ecx
  __int64 v24; // rdx
  __m128i *v25; // rdx
  int v26; // edi
  __int64 v27; // rdx
  __int64 v28; // rax
  int v29; // ecx
  __int64 v30; // r9
  unsigned int v31; // edi
  int *v32; // rsi
  int v33; // r8d
  unsigned int v34; // esi
  __int64 m128i_i64; // r8
  __int64 v36; // rdi
  __int64 v37; // rax
  int v38; // ecx
  _BYTE *v39; // r14
  __int64 v40; // rdx
  _BYTE *v41; // r13
  __int64 v42; // rdi
  int v43; // edx
  __int64 v44; // rcx
  int v45; // r9d
  __int64 v46; // rsi
  __int64 v47; // rax
  int v48; // r11d
  __m128i *v49; // rsi
  __m128i *v50; // rax
  int v51; // r11d
  __int32 v52; // eax
  __m128i *v53; // rax
  int v54; // eax
  int v55; // ecx
  __int64 v56; // rdi
  unsigned int v57; // eax
  int v58; // esi
  int v59; // r8d
  int v60; // r14d
  int *v61; // r11
  int v62; // eax
  int v63; // edx
  _QWORD *v64; // rax
  int v65; // esi
  int v66; // r11d
  int v67; // r8d
  int v68; // r8d
  __int64 v69; // r9
  unsigned int v70; // eax
  int v71; // edi
  int *v72; // r11
  int v73; // esi
  int *v74; // rcx
  int v75; // edi
  int v76; // edi
  __int64 v77; // r8
  unsigned int v78; // r12d
  int v79; // esi
  int *v80; // r9
  int v81; // ecx
  int *v82; // rax
  __int64 v83; // [rsp+0h] [rbp-280h]
  int v84; // [rsp+8h] [rbp-278h]
  __int64 v85; // [rsp+8h] [rbp-278h]
  int v86; // [rsp+14h] [rbp-26Ch]
  unsigned int v87; // [rsp+18h] [rbp-268h]
  int *v88; // [rsp+18h] [rbp-268h]
  __m128i v89; // [rsp+20h] [rbp-260h] BYREF
  __m128i v90; // [rsp+30h] [rbp-250h] BYREF
  _BYTE *v91; // [rsp+40h] [rbp-240h] BYREF
  __int64 v92; // [rsp+48h] [rbp-238h]
  _BYTE v93[560]; // [rsp+50h] [rbp-230h] BYREF

  v4 = a1;
  v91 = v93;
  v92 = 0x1000000000LL;
  v6 = sub_21F8260(a1, a2);
  if ( !v6 )
    goto LABEL_19;
  v7 = *(_QWORD *)(a2 + 32);
  v8 = v6;
  v9 = *(_QWORD *)(v7 + 40LL * (v6 + 1) + 24);
  if ( (_DWORD)v9 != 5 )
  {
    if ( (_DWORD)v9 )
      goto LABEL_19;
  }
  v10 = v7 + 40LL * (v8 + 5);
  if ( *(_BYTE *)v10 )
  {
    if ( *(_BYTE *)v10 != 5 )
      goto LABEL_19;
    v86 = *(_DWORD *)(v10 + 24);
    if ( v86 == -1 )
      goto LABEL_19;
  }
  else
  {
    v28 = *(unsigned int *)(a3 + 24);
    if ( !(_DWORD)v28 )
      goto LABEL_19;
    v29 = *(_DWORD *)(v10 + 8);
    v30 = *(_QWORD *)(a3 + 8);
    v31 = (v28 - 1) & (37 * v29);
    v32 = (int *)(v30 + 8LL * v31);
    v33 = *v32;
    if ( v29 != *v32 )
    {
      v65 = 1;
      while ( v33 != -1 )
      {
        v66 = v65 + 1;
        v31 = (v28 - 1) & (v31 + v65);
        v32 = (int *)(v30 + 8LL * v31);
        v33 = *v32;
        if ( v29 == *v32 )
          goto LABEL_31;
        v65 = v66;
      }
      goto LABEL_19;
    }
LABEL_31:
    if ( v32 == (int *)(v30 + 8 * v28) || (v86 = v32[1], v86 == -1) )
    {
LABEL_19:
      v22 = v91;
      goto LABEL_20;
    }
  }
  if ( *(_DWORD *)(v4 + 344) )
  {
    v54 = *(_DWORD *)(v4 + 352);
    if ( v54 )
    {
      v55 = v54 - 1;
      v56 = *(_QWORD *)(v4 + 336);
      v57 = (v54 - 1) & (37 * v86);
      v58 = *(_DWORD *)(v56 + 4LL * v57);
      if ( v86 != v58 )
      {
        v59 = 1;
        while ( v58 != -1 )
        {
          v57 = v55 & (v59 + v57);
          v58 = *(_DWORD *)(v56 + 4LL * v57);
          if ( v86 == v58 )
            goto LABEL_19;
          ++v59;
        }
        goto LABEL_8;
      }
      goto LABEL_19;
    }
  }
LABEL_8:
  v11 = *(_QWORD *)(v7 + 40LL * (v8 + 4) + 24);
  v12 = v11 + 7;
  if ( v11 >= 0 )
    v12 = *(_QWORD *)(v7 + 40LL * (v8 + 4) + 24);
  LODWORD(v13) = 0;
  v14 = v12 >> 3;
  v15 = v14;
  if ( v8 + 7 == *(_DWORD *)(a2 + 40) )
    v13 = *(_QWORD *)(v7 + 40LL * (v8 + 6) + 24);
  v16 = v8;
  v17 = 0;
  v18 = v13;
  v19 = v16;
  while ( 1 )
  {
    v20 = v7 + 40LL * v17;
    if ( *(_BYTE *)v20 )
      goto LABEL_18;
    v21 = (__int16 ***)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v4 + 480) + 24LL)
                                  + 16LL * (*(_DWORD *)(v20 + 8) & 0x7FFFFFFF))
                      & 0xFFFFFFFFFFFFFFF8LL);
    if ( v21 == &off_4A025A0 || v21 == &off_4A027A0 || v21 == &off_4A02720 || v21 == &off_4A024A0 )
    {
      v23 = 0;
    }
    else
    {
      if ( v21 != &off_4A02520 && v21 != &off_4A02620 )
      {
LABEL_18:
        sub_21FB820(v4, v86);
        v22 = v91;
        goto LABEL_20;
      }
      v23 = 1;
    }
    *(__int64 *)((char *)v89.m128i_i64 + 4) = __PAIR64__(v23, v15);
    v90.m128i_i64[0] = a2;
    v90.m128i_i32[2] = v17;
    v89.m128i_i32[0] = v18 + v14 * v17;
    v24 = (unsigned int)v92;
    if ( (unsigned int)v92 >= HIDWORD(v92) )
    {
      v83 = v14;
      v84 = v15;
      v87 = v19;
      sub_16CD150((__int64)&v91, v93, 0, 32, v19, v15);
      v24 = (unsigned int)v92;
      v14 = v83;
      v15 = v84;
      v19 = v87;
    }
    v25 = (__m128i *)&v91[32 * v24];
    ++v17;
    *v25 = _mm_loadu_si128(&v89);
    v26 = v92;
    v25[1] = _mm_loadu_si128(&v90);
    v27 = (unsigned int)(v26 + 1);
    LODWORD(v92) = v26 + 1;
    if ( v19 <= v17 )
      break;
    v7 = *(_QWORD *)(a2 + 32);
  }
  v34 = *(_DWORD *)(v4 + 256);
  if ( !v34 )
  {
    ++*(_QWORD *)(v4 + 232);
    goto LABEL_102;
  }
  LODWORD(m128i_i64) = v34 - 1;
  v36 = *(_QWORD *)(v4 + 240);
  LODWORD(v37) = (v34 - 1) & (37 * v86);
  v88 = (int *)(v36 + 16LL * (unsigned int)v37);
  v38 = *v88;
  if ( *v88 == v86 )
  {
LABEL_36:
    if ( *((_QWORD *)v88 + 1) )
      goto LABEL_37;
    goto LABEL_83;
  }
  v60 = 1;
  v61 = 0;
  while ( v38 != 0x7FFFFFFF )
  {
    if ( v38 == 0x80000000 && !v61 )
      v61 = v88;
    v37 = (unsigned int)m128i_i64 & ((_DWORD)v37 + v60);
    v88 = (int *)(v36 + 16 * v37);
    v38 = *v88;
    if ( *v88 == v86 )
      goto LABEL_36;
    ++v60;
  }
  v62 = *(_DWORD *)(v4 + 248);
  if ( !v61 )
    v61 = v88;
  ++*(_QWORD *)(v4 + 232);
  v63 = v62 + 1;
  v88 = v61;
  if ( 4 * (v62 + 1) >= 3 * v34 )
  {
LABEL_102:
    sub_21F8E70(v4 + 232, 2 * v34);
    v67 = *(_DWORD *)(v4 + 256);
    if ( v67 )
    {
      v68 = v67 - 1;
      v69 = *(_QWORD *)(v4 + 240);
      v70 = v68 & (37 * v86);
      v63 = *(_DWORD *)(v4 + 248) + 1;
      v88 = (int *)(v69 + 16LL * v70);
      v71 = *v88;
      if ( *v88 != v86 )
      {
        v72 = (int *)(v69 + 16LL * (v68 & (unsigned int)(37 * v86)));
        v73 = 1;
        v74 = 0;
        while ( v71 != 0x7FFFFFFF )
        {
          if ( !v74 && v71 == 0x80000000 )
            v74 = v72;
          v70 = v68 & (v73 + v70);
          v72 = (int *)(v69 + 16LL * v70);
          v71 = *v72;
          if ( *v72 == v86 )
          {
            v88 = (int *)(v69 + 16LL * v70);
            goto LABEL_80;
          }
          ++v73;
        }
        if ( !v74 )
          v74 = v72;
        v88 = v74;
      }
      goto LABEL_80;
    }
    goto LABEL_133;
  }
  if ( v34 - *(_DWORD *)(v4 + 252) - v63 <= v34 >> 3 )
  {
    sub_21F8E70(v4 + 232, v34);
    v75 = *(_DWORD *)(v4 + 256);
    if ( v75 )
    {
      v76 = v75 - 1;
      v77 = *(_QWORD *)(v4 + 240);
      v78 = v76 & (37 * v86);
      v79 = *(_DWORD *)(v77 + 16LL * v78);
      v88 = (int *)(v77 + 16LL * v78);
      v63 = *(_DWORD *)(v4 + 248) + 1;
      if ( v79 != v86 )
      {
        v80 = (int *)(v77 + 16LL * v78);
        v81 = 1;
        v82 = 0;
        while ( v79 != 0x7FFFFFFF )
        {
          if ( !v82 && v79 == 0x80000000 )
            v82 = v80;
          v78 = v76 & (v81 + v78);
          v80 = (int *)(v77 + 16LL * v78);
          v79 = *v80;
          if ( *v80 == v86 )
          {
            v88 = (int *)(v77 + 16LL * v78);
            goto LABEL_80;
          }
          ++v81;
        }
        if ( !v82 )
          v82 = v80;
        v88 = v82;
      }
      goto LABEL_80;
    }
LABEL_133:
    ++*(_DWORD *)(v4 + 248);
    BUG();
  }
LABEL_80:
  *(_DWORD *)(v4 + 248) = v63;
  if ( *v88 != 0x7FFFFFFF )
    --*(_DWORD *)(v4 + 252);
  *((_QWORD *)v88 + 1) = 0;
  *v88 = v86;
LABEL_83:
  v64 = (_QWORD *)sub_22077B0(528);
  if ( v64 )
  {
    *v64 = v64 + 2;
    v64[1] = 0x1000000000LL;
  }
  *((_QWORD *)v88 + 1) = v64;
  v27 = (unsigned int)v92;
LABEL_37:
  v39 = v91;
  v40 = 32 * v27;
  v22 = &v91[v40];
  if ( v91 != &v91[v40] )
  {
    v85 = v4;
    v41 = &v91[v40];
    while ( 1 )
    {
      v42 = *((_QWORD *)v88 + 1);
      v43 = *(_DWORD *)v39;
      v44 = *((unsigned int *)v39 + 1);
      v45 = *((_DWORD *)v39 + 2);
      v46 = *((_QWORD *)v39 + 2);
      v90.m128i_i32[2] = *((_DWORD *)v39 + 6);
      v89.m128i_i64[0] = __PAIR64__(v44, v43);
      v89.m128i_i32[2] = v45;
      v90.m128i_i64[0] = v46;
      v47 = *(unsigned int *)(v42 + 8);
      if ( !*(_DWORD *)(v42 + 8) )
      {
        if ( !*(_DWORD *)(v42 + 12) )
        {
          sub_16CD150(v42, (const void *)(v42 + 16), 0, 32, m128i_i64, v45);
          v47 = *(unsigned int *)(v42 + 8);
        }
        v53 = (__m128i *)(*(_QWORD *)v42 + 32 * v47);
        *v53 = _mm_loadu_si128(&v89);
        v53[1] = _mm_loadu_si128(&v90);
        ++*(_DWORD *)(v42 + 8);
        goto LABEL_41;
      }
      m128i_i64 = *(_QWORD *)v42;
      v48 = **(_DWORD **)v42;
      if ( v48 > v43 )
      {
        if ( v48 < (int)v44 + v43 )
          goto LABEL_56;
        sub_21F84E0(v42, *(__m128i **)v42, &v89, v44, m128i_i64, v45);
        goto LABEL_41;
      }
      v49 = (__m128i *)(m128i_i64 + 32);
      v50 = (__m128i *)(m128i_i64 + 32 * v47);
      if ( v50 == (__m128i *)(m128i_i64 + 32) )
      {
        if ( v48 >= v43 )
        {
          if ( v48 != v43 )
          {
LABEL_91:
            if ( *(_DWORD *)(v42 + 8) >= *(_DWORD *)(v42 + 12) )
            {
              sub_16CD150(v42, (const void *)(v42 + 16), 0, 32, m128i_i64, v45);
              v49 = (__m128i *)(*(_QWORD *)v42 + 32LL * *(unsigned int *)(v42 + 8));
            }
            *v49 = _mm_loadu_si128(&v89);
            v49[1] = _mm_loadu_si128(&v90);
            ++*(_DWORD *)(v42 + 8);
            goto LABEL_41;
          }
LABEL_65:
          if ( (_DWORD)v44 != *(_DWORD *)(m128i_i64 + 4) || v45 != *(_DWORD *)(m128i_i64 + 8) )
            goto LABEL_56;
          goto LABEL_53;
        }
        if ( v43 < *(_DWORD *)(m128i_i64 + 4) + v48 )
          goto LABEL_56;
      }
      else
      {
        while ( 1 )
        {
          m128i_i64 = (__int64)v49[-2].m128i_i64;
          if ( v49->m128i_i32[0] >= v43 )
            break;
          if ( v50 == &v49[2] )
          {
            m128i_i64 = (__int64)v49;
            v49 = v50;
            break;
          }
          v49 += 2;
        }
        v51 = *(_DWORD *)m128i_i64;
        if ( v43 > *(_DWORD *)m128i_i64 && v43 < v51 + *(_DWORD *)(m128i_i64 + 4) )
        {
LABEL_56:
          v4 = v85;
          goto LABEL_18;
        }
        if ( v43 >= v51 )
        {
          if ( v43 == v51 )
            goto LABEL_65;
        }
        else
        {
          m128i_i64 = (unsigned int)(v44 + v43);
          if ( v51 < (int)m128i_i64 )
            goto LABEL_56;
        }
      }
LABEL_53:
      if ( v50 == v49 )
        goto LABEL_91;
      v52 = v49->m128i_i32[0];
      if ( v43 >= v49->m128i_i32[0] )
      {
        if ( v43 <= v52 )
        {
          if ( v43 == v52 && ((_DWORD)v44 != v49->m128i_i32[1] || v45 != v49->m128i_i32[2]) )
            goto LABEL_56;
        }
        else if ( v43 < v49->m128i_i32[1] + v52 )
        {
          goto LABEL_56;
        }
      }
      else if ( v52 < (int)v44 + v43 )
      {
        goto LABEL_56;
      }
      sub_21F84E0(v42, v49, &v89, v44, m128i_i64, v45);
LABEL_41:
      v39 += 32;
      if ( v41 == v39 )
        goto LABEL_19;
    }
  }
LABEL_20:
  if ( v22 != v93 )
    _libc_free((unsigned __int64)v22);
}
