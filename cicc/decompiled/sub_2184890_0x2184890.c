// Function: sub_2184890
// Address: 0x2184890
//
__int64 __fastcall sub_2184890(__int64 a1, int *a2)
{
  __int64 v2; // r14
  _DWORD *v3; // rax
  unsigned int v4; // r12d
  _DWORD *v5; // rdx
  __int64 v6; // rbx
  int v7; // esi
  __int64 v8; // rcx
  int v9; // r8d
  int v10; // r9d
  unsigned int v11; // r15d
  __int64 v12; // rax
  __int64 v13; // rbx
  unsigned __int64 v14; // rbx
  __int64 i; // rbx
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // rbx
  __int64 v19; // r14
  __int64 v20; // r13
  int v21; // esi
  __int64 v22; // rdx
  int v23; // r12d
  int v24; // ecx
  __int64 v25; // rdx
  __int64 *v26; // rsi
  __int64 v27; // rcx
  __int64 v28; // rax
  int v29; // esi
  __int64 v30; // rcx
  unsigned int v31; // edx
  __int64 *v32; // rax
  __int64 v33; // rdi
  _QWORD *v34; // rax
  _QWORD *v35; // rdx
  __int64 v36; // rax
  __int64 j; // rbx
  __int64 v38; // r13
  __int64 v39; // rbx
  __int64 v40; // r13
  unsigned int v41; // r15d
  char v42; // al
  __int64 v43; // rdx
  int v45; // esi
  int v46; // eax
  __int64 v47; // rax
  unsigned int v48; // eax
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // rdi
  int v52; // esi
  unsigned int v53; // ecx
  int v54; // r9d
  __int64 v55; // rax
  void *v56; // rax
  __int64 v57; // rcx
  int v58; // r9d
  __int64 *v59; // r8
  int v60; // edx
  unsigned int v61; // [rsp+Ch] [rbp-124h]
  unsigned __int64 v62; // [rsp+10h] [rbp-120h]
  __int64 v63; // [rsp+10h] [rbp-120h]
  __int64 v64; // [rsp+18h] [rbp-118h]
  __int64 v65; // [rsp+18h] [rbp-118h]
  unsigned int v66; // [rsp+20h] [rbp-110h]
  __int64 v67; // [rsp+20h] [rbp-110h]
  int v68; // [rsp+28h] [rbp-108h]
  unsigned int v69; // [rsp+28h] [rbp-108h]
  __int64 v70; // [rsp+28h] [rbp-108h]
  __int64 v71; // [rsp+28h] [rbp-108h]
  __int64 v72; // [rsp+30h] [rbp-100h]
  int v74; // [rsp+44h] [rbp-ECh] BYREF
  __int64 v75; // [rsp+48h] [rbp-E8h] BYREF
  void *s; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v77; // [rsp+58h] [rbp-D8h]
  unsigned int v78; // [rsp+60h] [rbp-D0h]
  __int64 v79; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v80; // [rsp+78h] [rbp-B8h]
  __int64 v81; // [rsp+80h] [rbp-B0h]
  int v82; // [rsp+88h] [rbp-A8h]
  __int64 v83; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v84; // [rsp+98h] [rbp-98h]
  __int64 v85; // [rsp+A0h] [rbp-90h]
  __int64 v86; // [rsp+A8h] [rbp-88h]
  __int64 v87; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v88; // [rsp+B8h] [rbp-78h]
  __int64 v89; // [rsp+C0h] [rbp-70h]
  unsigned int v90; // [rsp+C8h] [rbp-68h]
  _QWORD v91[4]; // [rsp+D0h] [rbp-60h] BYREF
  char v92; // [rsp+F0h] [rbp-40h]

  v2 = a1;
  v3 = *(_DWORD **)(a1 + 8);
  v4 = *(_DWORD *)(a1 + 16);
  v5 = &v3[*(unsigned int *)(a1 + 24)];
  if ( v4 )
  {
    if ( v5 == v3 )
      goto LABEL_67;
    while ( *v3 > 0xFFFFFFFD )
    {
      if ( v5 == ++v3 )
        goto LABEL_67;
    }
    if ( v5 == v3 )
    {
LABEL_67:
      v4 = 0;
    }
    else
    {
      v49 = *(_QWORD *)(a1 + 72);
      v4 = 0;
      v50 = *(_QWORD *)(*(_QWORD *)(a1 + 80) + 24LL);
      v51 = *(_QWORD *)(v49 + 280);
      v52 = *(_DWORD *)(v49 + 288) * ((__int64)(*(_QWORD *)(v49 + 264) - *(_QWORD *)(v49 + 256)) >> 3);
LABEL_70:
      v53 = v4 + 2;
      ++v4;
      if ( *(_DWORD *)(v51
                     + 24LL
                     * (v52
                      + (unsigned int)*(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v50 + 16LL * (*v3 & 0x7FFFFFFF))
                                                                      & 0xFFFFFFFFFFFFFFF8LL)
                                                          + 24LL))) > 0x20u )
        v4 = v53;
      while ( v5 != ++v3 )
      {
        if ( *v3 <= 0xFFFFFFFD )
        {
          if ( v5 != v3 )
            goto LABEL_70;
          break;
        }
      }
    }
  }
  v6 = *(_QWORD *)(v2 + 64);
  v72 = *(_QWORD *)(v6 + 32);
  if ( v72 != v6 + 24 )
  {
    v79 = 0;
    v80 = 0;
    v81 = 0;
    v7 = *a2;
    v82 = 0;
    sub_1BFC1A0((__int64)&s, v7, 0);
    if ( v77 )
      memset(s, 0, 8 * v77);
    v8 = *(_QWORD *)(sub_1C01EA0((__int64)a2, *(_QWORD *)(v2 + 64)) + 8);
    v11 = *(_DWORD *)(v8 + 16);
    if ( v78 < v11 )
    {
      v69 = v77;
      if ( v11 <= (unsigned __int64)(v77 << 6) )
        goto LABEL_60;
      v63 = v8;
      v55 = 2 * v77;
      if ( (v11 + 63) >> 6 >= (unsigned __int64)(2 * v77) )
        v55 = (v11 + 63) >> 6;
      v67 = v55;
      v64 = 8 * v55;
      v56 = realloc((unsigned __int64)s, 8 * v55, 8 * (int)v55, v8, v9, v10);
      v57 = v63;
      if ( !v56 )
      {
        if ( v64 )
        {
          sub_16BD1C0("Allocation failed", 1u);
          v56 = 0;
          v57 = v63;
        }
        else
        {
          v56 = (void *)malloc(1u);
          v57 = v63;
          if ( !v56 )
          {
            sub_16BD1C0("Allocation failed", 1u);
            v57 = v63;
            v56 = 0;
          }
        }
      }
      s = v56;
      v65 = v57;
      v77 = v67;
      sub_13A4C60((__int64)&s, 0);
      v8 = v65;
      if ( v77 != v69 )
      {
        memset((char *)s + 8 * v69, 0, 8 * (v77 - v69));
        v8 = v65;
      }
      v48 = v78;
      if ( v11 > v78 )
      {
LABEL_60:
        v70 = v8;
        sub_13A4C60((__int64)&s, 0);
        v48 = v78;
        v8 = v70;
      }
      v78 = v11;
      if ( v48 > v11 )
      {
        v71 = v8;
        sub_13A4C60((__int64)&s, 0);
        v8 = v71;
      }
      v11 = *(_DWORD *)(v8 + 16);
    }
    v12 = 0;
    if ( (v11 + 63) >> 6 )
    {
      do
      {
        *((_QWORD *)s + v12) |= *(_QWORD *)(*(_QWORD *)v8 + 8 * v12);
        ++v12;
      }
      while ( (v11 + 63) >> 6 != v12 );
    }
    v86 = 0;
    v90 = 0;
    v13 = *(_QWORD *)(v6 + 24);
    v83 = 0;
    v84 = 0;
    v85 = 0;
    v87 = 0;
    v88 = 0;
    v89 = 0;
    v14 = v13 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v14 )
      BUG();
    if ( (*(_QWORD *)v14 & 4) == 0 && (*(_BYTE *)(v14 + 46) & 4) != 0 )
    {
      for ( i = *(_QWORD *)v14; ; i = *(_QWORD *)v14 )
      {
        v14 = i & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v14 + 46) & 4) == 0 )
          break;
      }
    }
    v75 = v14;
    if ( **(_WORD **)(v14 + 16) && **(_WORD **)(v14 + 16) != 45 )
    {
      v66 = v4;
      v68 = 0;
      v61 = v4;
      while ( 1 )
      {
        v16 = *(unsigned int *)(v14 + 40);
        if ( !(_DWORD)v16 )
          goto LABEL_27;
        v62 = v14;
        v17 = 0;
        v18 = v2;
        v19 = 40 * v16;
        do
        {
          while ( 1 )
          {
            v20 = v17 + *(_QWORD *)(v75 + 32);
            if ( *(_BYTE *)v20 )
              goto LABEL_25;
            v21 = *(_DWORD *)(v20 + 8);
            v74 = v21;
            if ( v21 >= 0 )
              goto LABEL_25;
            v22 = *(_QWORD *)(v18 + 72);
            v23 = (*(_DWORD *)(*(_QWORD *)(v22 + 280)
                             + 24LL
                             * (*(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v18 + 80) + 24LL)
                                                                            + 16LL * (v21 & 0x7FFFFFFF))
                                                                & 0xFFFFFFFFFFFFFFF8LL)
                                                    + 24LL)
                              + *(_DWORD *)(v22 + 288)
                              * (unsigned int)((__int64)(*(_QWORD *)(v22 + 264) - *(_QWORD *)(v22 + 256)) >> 3))) > 0x20u)
                + 1;
            v24 = sub_217F620((__int64)a2, v21);
            if ( v24 >= 0 )
            {
              v25 = 1LL << v24;
              v26 = (__int64 *)((char *)s + 8 * ((unsigned int)v24 >> 6));
              v27 = *v26;
              v28 = v25 & *v26;
              if ( (*(_BYTE *)(v20 + 3) & 0x10) != 0 )
              {
                if ( v28 )
                {
                  v66 -= v23;
                  *v26 = v27 & ~v25;
                }
              }
              else if ( !v28 )
              {
                v66 += v23;
                *v26 = v25 | v27;
              }
              goto LABEL_25;
            }
            if ( (*(_BYTE *)(v20 + 3) & 0x10) == 0 )
              break;
            if ( (unsigned __int8)sub_1DF91F0((__int64)&v83, &v74, v91) )
            {
              v17 += 40;
              *(_DWORD *)v91[0] = -2;
              v68 -= v23;
              LODWORD(v85) = v85 - 1;
              ++HIDWORD(v85);
              if ( v17 == v19 )
                goto LABEL_26;
            }
            else
            {
LABEL_25:
              v17 += 40;
              if ( v17 == v19 )
                goto LABEL_26;
            }
          }
          sub_217F7B0((__int64)v91, (__int64)&v83, &v74);
          v54 = v68 + v23;
          if ( !v92 )
            v54 = v68;
          v17 += 40;
          v68 = v54;
        }
        while ( v17 != v19 );
LABEL_26:
        v2 = v18;
        v14 = v62;
LABEL_27:
        v29 = v90;
        if ( !v90 )
        {
          ++v87;
LABEL_106:
          v29 = 2 * v90;
LABEL_107:
          sub_1DC6D40((__int64)&v87, v29);
          sub_1FD4240((__int64)&v87, &v75, v91);
          v32 = (__int64 *)v91[0];
          v30 = v75;
          v60 = v89 + 1;
          goto LABEL_99;
        }
        v30 = v75;
        v31 = (v90 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
        v32 = (__int64 *)(v88 + 16LL * v31);
        v33 = *v32;
        if ( v75 == *v32 )
          goto LABEL_29;
        v58 = 1;
        v59 = 0;
        while ( v33 != -8 )
        {
          if ( v33 == -16 && !v59 )
            v59 = v32;
          v31 = (v90 - 1) & (v58 + v31);
          v32 = (__int64 *)(v88 + 16LL * v31);
          v33 = *v32;
          if ( v75 == *v32 )
            goto LABEL_29;
          ++v58;
        }
        if ( v59 )
          v32 = v59;
        ++v87;
        v60 = v89 + 1;
        if ( 4 * ((int)v89 + 1) >= 3 * v90 )
          goto LABEL_106;
        if ( v90 - HIDWORD(v89) - v60 <= v90 >> 3 )
          goto LABEL_107;
LABEL_99:
        LODWORD(v89) = v60;
        if ( *v32 != -8 )
          --HIDWORD(v89);
        *v32 = v30;
        *((_DWORD *)v32 + 2) = 0;
LABEL_29:
        *((_DWORD *)v32 + 2) = v66 + v68;
        if ( v72 != v14 )
        {
          v34 = (_QWORD *)(*(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL);
          v35 = v34;
          if ( !v34 )
            BUG();
          v14 = *(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL;
          v36 = *v34;
          if ( (v36 & 4) == 0 && (*((_BYTE *)v35 + 46) & 4) != 0 )
          {
            for ( j = v36; ; j = *(_QWORD *)v14 )
            {
              v14 = j & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_BYTE *)(v14 + 46) & 4) == 0 )
                break;
            }
          }
          v75 = v14;
          if ( **(_WORD **)(v14 + 16) != 45 )
          {
            if ( **(_WORD **)(v14 + 16) )
              continue;
          }
        }
        v4 = v61;
        break;
      }
    }
    v38 = *(_QWORD *)(v2 + 64);
    v39 = *(_QWORD *)(v38 + 32);
    v40 = v38 + 24;
    if ( v40 != v39 )
    {
      while ( 1 )
      {
        v75 = v39;
        if ( **(_WORD **)(v39 + 16) != 45 && **(_WORD **)(v39 + 16) && *(_DWORD *)(v39 + 40) )
          break;
LABEL_47:
        if ( (*(_BYTE *)v39 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v39 + 46) & 8) != 0 )
            v39 = *(_QWORD *)(v39 + 8);
        }
        v39 = *(_QWORD *)(v39 + 8);
        if ( v40 == v39 )
          goto LABEL_49;
      }
      v41 = sub_217F8F0(v2, v39, (__int64)&v79);
      v42 = sub_1FD4240((__int64)&v87, &v75, v91);
      v43 = v91[0];
      if ( v42 )
      {
        v41 += *(_DWORD *)(v91[0] + 8LL);
LABEL_45:
        if ( v4 < v41 )
          v4 = v41;
        goto LABEL_47;
      }
      v45 = v90;
      ++v87;
      v46 = v89 + 1;
      if ( 4 * ((int)v89 + 1) >= 3 * v90 )
      {
        v45 = 2 * v90;
      }
      else if ( v90 - HIDWORD(v89) - v46 > v90 >> 3 )
      {
LABEL_56:
        LODWORD(v89) = v46;
        if ( *(_QWORD *)v43 != -8 )
          --HIDWORD(v89);
        v47 = v75;
        *(_DWORD *)(v43 + 8) = 0;
        *(_QWORD *)v43 = v47;
        goto LABEL_45;
      }
      sub_1DC6D40((__int64)&v87, v45);
      sub_1FD4240((__int64)&v87, &v75, v91);
      v43 = v91[0];
      v46 = v89 + 1;
      goto LABEL_56;
    }
LABEL_49:
    j___libc_free_0(v88);
    j___libc_free_0(v84);
    _libc_free((unsigned __int64)s);
    j___libc_free_0(v80);
  }
  return v4;
}
