// Function: sub_3016BA0
// Address: 0x3016ba0
//
void __fastcall sub_3016BA0(__int64 a1, const __m128i *a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int32 v8; // ebx
  __int64 v9; // rax
  __int64 *v10; // rbx
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // rax
  __int64 *i; // r13
  __int64 v14; // rax
  bool v15; // zf
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  _QWORD *v19; // rax
  unsigned int v20; // r13d
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // r13
  __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rsi
  __int64 v27; // rax
  _QWORD *v28; // rax
  unsigned int v29; // ebx
  const __m128i *v30; // rdi
  unsigned __int64 v31; // r9
  unsigned int v32; // esi
  __int64 v33; // r14
  __int64 v34; // r9
  int v35; // r11d
  __int64 *v36; // rcx
  unsigned int v37; // r8d
  _QWORD *v38; // rax
  __int64 v39; // rdi
  unsigned int *v40; // rax
  __int64 v41; // r14
  _BYTE *v42; // r13
  char v43; // al
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rdx
  int v48; // ecx
  int v49; // eax
  __int64 v50; // rsi
  int v51; // ecx
  unsigned int v52; // eax
  const __m128i *v53; // rdx
  int v54; // edi
  int v55; // eax
  int v56; // eax
  __int64 v57; // rax
  _QWORD *v58; // rax
  unsigned int v59; // r13d
  __int64 v60; // rbx
  __int64 v61; // rdx
  __int64 v62; // rdi
  __int64 v63; // rsi
  __int64 j; // rax
  unsigned int v65; // ecx
  int v66; // esi
  int v67; // esi
  __int64 v68; // r8
  __int64 v69; // rdx
  __int64 v70; // rdi
  int v71; // r11d
  __int64 *v72; // r9
  int v73; // esi
  int v74; // esi
  __int64 v75; // r9
  __int64 *v76; // r8
  int v77; // r11d
  __int64 v78; // rdx
  __int64 v79; // rdi
  unsigned int v80; // [rsp+4h] [rbp-9Ch]
  __int64 v81; // [rsp+8h] [rbp-98h]
  int v82; // [rsp+10h] [rbp-90h]
  unsigned int v83; // [rsp+14h] [rbp-8Ch]
  bool v84; // [rsp+23h] [rbp-7Dh]
  unsigned int v85; // [rsp+24h] [rbp-7Ch]
  __int64 v86; // [rsp+28h] [rbp-78h]
  const __m128i *v88; // [rsp+30h] [rbp-70h]
  __int64 v89; // [rsp+38h] [rbp-68h]
  __int64 *v90; // [rsp+38h] [rbp-68h]
  __int64 v91; // [rsp+48h] [rbp-58h] BYREF
  const __m128i *v92; // [rsp+50h] [rbp-50h] BYREF
  __int64 v93; // [rsp+58h] [rbp-48h]
  _BYTE v94[64]; // [rsp+60h] [rbp-40h] BYREF

  v89 = a2[2].m128i_i64[1];
  if ( a2->m128i_i8[0] == 39 )
  {
    v8 = a2->m128i_i32[1];
    v92 = (const __m128i *)v94;
    v93 = 0x200000000LL;
    v9 = a2[-1].m128i_i64[1];
    v10 = (__int64 *)(v9 + 32LL * (v8 & 0x7FFFFFF));
    v11 = v9 + 32;
    v12 = v9 + 64;
    if ( (a2->m128i_i8[2] & 1) != 0 )
      v11 = v12;
    for ( i = (__int64 *)v11; v10 != i; LODWORD(v93) = v93 + 1 )
    {
      v14 = sub_AA4FF0(*i);
      v15 = v14 == 0;
      v16 = v14 - 24;
      v17 = (unsigned int)v93;
      if ( v15 )
        v16 = 0;
      v11 = (unsigned int)v93 + 1LL;
      if ( v11 > HIDWORD(v93) )
      {
        v86 = v16;
        sub_C8D5F0((__int64)&v92, v94, (unsigned int)v93 + 1LL, 8u, v11, a6);
        v17 = (unsigned int)v93;
        v16 = v86;
      }
      i += 4;
      v92->m128i_i64[v17] = v16;
    }
    v18 = *(unsigned int *)(a1 + 168);
    if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 172) )
    {
      sub_C8D5F0(a1 + 160, (const void *)(a1 + 176), v18 + 1, 0x10u, v11, a6);
      v18 = *(unsigned int *)(a1 + 168);
    }
    v19 = (_QWORD *)(*(_QWORD *)(a1 + 160) + 16 * v18);
    *v19 = a3;
    v19[1] = 0;
    v20 = *(_DWORD *)(a1 + 168);
    v91 = (__int64)a2;
    v85 = v20;
    *(_DWORD *)(a1 + 168) = v20 + 1;
    *(_DWORD *)sub_3014430(a1, &v91) = v20;
    v23 = *(_QWORD *)(v89 + 16);
    if ( v23 )
    {
      while ( 1 )
      {
        v24 = *(_QWORD *)(v23 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v24 - 30) <= 0xAu )
          break;
        v23 = *(_QWORD *)(v23 + 8);
        if ( !v23 )
          goto LABEL_21;
      }
LABEL_16:
      v25 = sub_3012100(*(_QWORD *)(v24 + 40), *(_QWORD *)a2[-1].m128i_i64[1]);
      if ( v25 )
      {
        v26 = sub_AA4FF0(v25);
        if ( v26 )
          v26 -= 24;
        sub_3016BA0(a1, v26, v85);
      }
      while ( 1 )
      {
        v23 = *(_QWORD *)(v23 + 8);
        if ( !v23 )
          break;
        v24 = *(_QWORD *)(v23 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v24 - 30) <= 0xAu )
          goto LABEL_16;
      }
    }
LABEL_21:
    v27 = *(unsigned int *)(a1 + 168);
    if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 172) )
    {
      sub_C8D5F0(a1 + 160, (const void *)(a1 + 176), v27 + 1, 0x10u, v21, v22);
      v27 = *(unsigned int *)(a1 + 168);
    }
    v28 = (_QWORD *)(*(_QWORD *)(a1 + 160) + 16 * v27);
    *v28 = a3;
    v28[1] = 0;
    v29 = *(_DWORD *)(a1 + 168);
    *(_DWORD *)(a1 + 168) = v29 + 1;
    v82 = v29 - 1;
    v84 = sub_CC7F40(*(_QWORD *)(*(_QWORD *)(v89 + 72) + 40LL) + 232LL);
    if ( v84 )
    {
      sub_3012CF0(a1, (_BYTE *)v85, v82, v29, v92, (unsigned int)v93);
      v30 = v92;
      v83 = *(_DWORD *)(a1 + 248) - 1;
      v88 = (const __m128i *)((char *)v92 + 8 * (unsigned int)v93);
      if ( v88 == v92 )
      {
        v48 = *(_DWORD *)(a1 + 168) - 1;
LABEL_46:
        *(_DWORD *)(*(_QWORD *)(a1 + 240) + ((unsigned __int64)v83 << 6) + 8) = v48;
        goto LABEL_47;
      }
    }
    else
    {
      v30 = v92;
      v31 = (unsigned int)v93;
      v83 = *(_DWORD *)(a1 + 248) - 1;
      v88 = (const __m128i *)((char *)v92 + 8 * (unsigned int)v93);
      if ( v88 == v92 )
      {
        v48 = *(_DWORD *)(a1 + 168) - 1;
        goto LABEL_93;
      }
    }
    v90 = (__int64 *)v30;
    v81 = a1 + 32;
    while ( 1 )
    {
      v32 = *(_DWORD *)(a1 + 56);
      v33 = *v90;
      if ( !v32 )
        break;
      v34 = *(_QWORD *)(a1 + 40);
      v35 = 1;
      v36 = 0;
      v37 = (v32 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
      v38 = (_QWORD *)(v34 + 16LL * v37);
      v39 = *v38;
      if ( v33 != *v38 )
      {
        while ( v39 != -4096 )
        {
          if ( v39 == -8192 && !v36 )
            v36 = v38;
          v37 = (v32 - 1) & (v35 + v37);
          v38 = (_QWORD *)(v34 + 16LL * v37);
          v39 = *v38;
          if ( v33 == *v38 )
            goto LABEL_28;
          ++v35;
        }
        if ( !v36 )
          v36 = v38;
        v55 = *(_DWORD *)(a1 + 48);
        ++*(_QWORD *)(a1 + 32);
        v56 = v55 + 1;
        if ( 4 * v56 < 3 * v32 )
        {
          if ( v32 - *(_DWORD *)(a1 + 52) - v56 <= v32 >> 3 )
          {
            v80 = ((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4);
            sub_30169C0(v81, v32);
            v73 = *(_DWORD *)(a1 + 56);
            if ( !v73 )
            {
LABEL_119:
              ++*(_DWORD *)(a1 + 48);
              BUG();
            }
            v74 = v73 - 1;
            v75 = *(_QWORD *)(a1 + 40);
            v76 = 0;
            v77 = 1;
            v56 = *(_DWORD *)(a1 + 48) + 1;
            LODWORD(v78) = v74 & v80;
            v36 = (__int64 *)(v75 + 16LL * (v74 & v80));
            v79 = *v36;
            if ( v33 != *v36 )
            {
              while ( v79 != -4096 )
              {
                if ( v79 == -8192 && !v76 )
                  v76 = v36;
                v78 = v74 & (unsigned int)(v78 + v77);
                v36 = (__int64 *)(v75 + 16 * v78);
                v79 = *v36;
                if ( v33 == *v36 )
                  goto LABEL_70;
                ++v77;
              }
              if ( v76 )
                v36 = v76;
            }
          }
          goto LABEL_70;
        }
LABEL_95:
        sub_30169C0(v81, 2 * v32);
        v66 = *(_DWORD *)(a1 + 56);
        if ( !v66 )
          goto LABEL_119;
        v67 = v66 - 1;
        v68 = *(_QWORD *)(a1 + 40);
        LODWORD(v69) = v67 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
        v56 = *(_DWORD *)(a1 + 48) + 1;
        v36 = (__int64 *)(v68 + 16LL * (unsigned int)v69);
        v70 = *v36;
        if ( v33 != *v36 )
        {
          v71 = 1;
          v72 = 0;
          while ( v70 != -4096 )
          {
            if ( !v72 && v70 == -8192 )
              v72 = v36;
            v69 = v67 & (unsigned int)(v69 + v71);
            v36 = (__int64 *)(v68 + 16 * v69);
            v70 = *v36;
            if ( v33 == *v36 )
              goto LABEL_70;
            ++v71;
          }
          if ( v72 )
            v36 = v72;
        }
LABEL_70:
        *(_DWORD *)(a1 + 48) = v56;
        if ( *v36 != -4096 )
          --*(_DWORD *)(a1 + 52);
        *v36 = v33;
        v40 = (unsigned int *)(v36 + 1);
        *((_DWORD *)v36 + 2) = 0;
        goto LABEL_29;
      }
LABEL_28:
      v40 = (unsigned int *)(v38 + 1);
LABEL_29:
      *v40 = v29;
      v91 = v33;
      *(_DWORD *)sub_3014430(a1, &v91) = v29;
      v41 = *(_QWORD *)(v33 + 16);
      if ( v41 )
      {
        while ( 1 )
        {
          v42 = *(_BYTE **)(v41 + 24);
          v43 = *v42;
          if ( *v42 != 39 )
            goto LABEL_37;
          if ( (v42[2] & 1) == 0 )
            break;
          v44 = *(_QWORD *)(*((_QWORD *)v42 - 1) + 32LL);
          if ( !v44 )
            break;
          if ( (a2->m128i_i8[2] & 1) != 0 )
          {
            v45 = *(_QWORD *)(a2[-1].m128i_i64[1] + 32);
            if ( v44 == v45 )
            {
              if ( v45 )
                break;
            }
          }
LABEL_43:
          v41 = *(_QWORD *)(v41 + 8);
          if ( !v41 )
            goto LABEL_44;
        }
        sub_3016BA0(a1, *(_QWORD *)(v41 + 24), v29);
        v43 = *v42;
LABEL_37:
        if ( v43 == 80 )
        {
          v46 = sub_3011DA0((__int64)v42);
          if ( !v46 || (a2->m128i_i8[2] & 1) != 0 && (v47 = *(_QWORD *)(a2[-1].m128i_i64[1] + 32), v46 == v47) && v47 )
            sub_3016BA0(a1, v42, v29);
        }
        goto LABEL_43;
      }
LABEL_44:
      if ( v88 == (const __m128i *)++v90 )
      {
        v30 = v92;
        v48 = *(_DWORD *)(a1 + 168) - 1;
        if ( v84 )
          goto LABEL_46;
        v31 = (unsigned int)v93;
LABEL_93:
        sub_3012CF0(a1, (_BYTE *)v85, v82, v48, v30, v31);
        v30 = v92;
LABEL_47:
        if ( v30 != (const __m128i *)v94 )
          _libc_free((unsigned __int64)v30);
        return;
      }
    }
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_95;
  }
  v49 = *(_DWORD *)(a1 + 24);
  v50 = *(_QWORD *)(a1 + 8);
  if ( v49 )
  {
    v51 = v49 - 1;
    v52 = (v49 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v53 = *(const __m128i **)(v50 + 16LL * v52);
    if ( a2 == v53 )
      return;
    v54 = 1;
    while ( v53 != (const __m128i *)-4096LL )
    {
      a5 = (unsigned int)(v54 + 1);
      v52 = v51 & (v54 + v52);
      v53 = *(const __m128i **)(v50 + 16LL * v52);
      if ( a2 == v53 )
        return;
      ++v54;
    }
  }
  v57 = *(unsigned int *)(a1 + 168);
  if ( v57 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 172) )
  {
    sub_C8D5F0(a1 + 160, (const void *)(a1 + 176), v57 + 1, 0x10u, a5, a6);
    v57 = *(unsigned int *)(a1 + 168);
  }
  v58 = (_QWORD *)(*(_QWORD *)(a1 + 160) + 16 * v57);
  v58[1] = v89 & 0xFFFFFFFFFFFFFFFBLL;
  *v58 = a3;
  v59 = *(_DWORD *)(a1 + 168);
  v92 = a2;
  *(_DWORD *)(a1 + 168) = v59 + 1;
  *(_DWORD *)sub_3014430(a1, (__int64 *)&v92) = v59;
  v60 = *(_QWORD *)(v89 + 16);
  if ( v60 )
  {
    while ( 1 )
    {
      v61 = *(_QWORD *)(v60 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v61 - 30) <= 0xAu )
        break;
      v60 = *(_QWORD *)(v60 + 8);
      if ( !v60 )
        goto LABEL_84;
    }
LABEL_79:
    v62 = sub_3012100(*(_QWORD *)(v61 + 40), a2[-2].m128i_i64[0]);
    if ( v62 )
    {
      v63 = sub_AA4FF0(v62);
      if ( v63 )
        v63 -= 24;
      sub_3016BA0(a1, v63, v59);
    }
    while ( 1 )
    {
      v60 = *(_QWORD *)(v60 + 8);
      if ( !v60 )
        break;
      v61 = *(_QWORD *)(v60 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v61 - 30) <= 0xAu )
        goto LABEL_79;
    }
  }
LABEL_84:
  for ( j = a2[1].m128i_i64[0]; j; j = *(_QWORD *)(j + 8) )
  {
    v65 = **(unsigned __int8 **)(j + 24) - 39;
    if ( v65 <= 0x38 && ((1LL << v65) & 0x100060000000001LL) != 0 )
      sub_C64ED0("Cleanup funclets for the MSVC++ personality cannot contain exceptional actions", 1u);
  }
}
