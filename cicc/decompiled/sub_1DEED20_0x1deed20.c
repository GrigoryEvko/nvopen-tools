// Function: sub_1DEED20
// Address: 0x1deed20
//
void __fastcall sub_1DEED20(__int64 a1)
{
  __int64 v2; // rax
  __int64 *v3; // r12
  __int64 v4; // rdx
  __int64 v5; // rcx
  unsigned int v6; // r14d
  unsigned __int64 v7; // r13
  __int64 v8; // rbx
  int v9; // r8d
  int v10; // r9d
  unsigned __int64 v11; // r13
  unsigned __int64 v12; // rbx
  __int64 v13; // rdx
  unsigned int v14; // esi
  __int64 v15; // r8
  unsigned int v16; // ecx
  __int64 *v17; // rax
  __int64 v18; // rdi
  __int64 (*v19)(); // r14
  __int64 v20; // r8
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 (*v25)(); // rax
  int v26; // r8d
  int v27; // r9d
  __int64 v28; // rax
  __int64 v29; // r13
  unsigned int v30; // esi
  int v31; // ecx
  __int64 v32; // rdx
  __int64 v33; // rax
  _QWORD **v34; // rbx
  _QWORD **i; // r12
  _QWORD *v36; // rsi
  __int64 v37; // rbx
  __int64 v38; // rbx
  __int64 v39; // r13
  __int64 v40; // r13
  __int64 **v41; // rbx
  __int64 v42; // rax
  unsigned __int64 *v43; // r12
  __int64 v44; // r14
  __int64 (*v45)(); // r10
  unsigned __int64 *v46; // rdx
  unsigned __int64 v47; // rsi
  unsigned __int64 v48; // rdx
  __int64 v49; // rdi
  __int64 v50; // rdx
  __int64 (*v51)(); // rax
  __int64 *v52; // rax
  int v53; // r11d
  __int64 *v54; // r10
  int v55; // ecx
  char v56; // al
  __int64 v57; // rdi
  __int64 (*v58)(); // r10
  unsigned __int64 v59; // rdi
  int v60; // r13d
  __int64 *v61; // r11
  int v62; // ecx
  int v63; // ecx
  int v64; // r13d
  int v65; // r13d
  __int64 v66; // r11
  unsigned int v67; // esi
  __int64 v68; // r8
  int v69; // r10d
  __int64 *v70; // rdi
  __int64 *v71; // [rsp+10h] [rbp-170h]
  __int64 v72; // [rsp+18h] [rbp-168h]
  __int64 *v73; // [rsp+20h] [rbp-160h]
  __int64 (*v74)(); // [rsp+28h] [rbp-158h]
  __int64 v75; // [rsp+38h] [rbp-148h] BYREF
  __int64 v76; // [rsp+40h] [rbp-140h] BYREF
  __int64 v77; // [rsp+48h] [rbp-138h] BYREF
  __int64 *v78; // [rsp+50h] [rbp-130h] BYREF
  _BYTE *v79; // [rsp+58h] [rbp-128h]
  _BYTE *v80; // [rsp+60h] [rbp-120h]
  __int64 v81; // [rsp+68h] [rbp-118h]
  int v82; // [rsp+70h] [rbp-110h]
  _BYTE v83[40]; // [rsp+78h] [rbp-108h] BYREF
  _BYTE *v84; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v85; // [rsp+A8h] [rbp-D8h]
  _BYTE v86[208]; // [rsp+B0h] [rbp-D0h] BYREF

  v84 = v86;
  v85 = 0x400000000LL;
  v2 = *(_QWORD *)(a1 + 552);
  v72 = a1 + 888;
  v3 = *(__int64 **)(v2 + 328);
  v71 = (__int64 *)(v2 + 320);
  if ( v3 != (__int64 *)(v2 + 320) )
  {
    while ( 1 )
    {
      v4 = *(_QWORD *)(a1 + 784);
      v5 = *(_QWORD *)(a1 + 792);
      *(_QWORD *)(a1 + 864) += 64LL;
      if ( ((v4 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v4 + 64 <= v5 - v4 )
      {
        v12 = (v4 + 7) & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(a1 + 784) = v12 + 64;
      }
      else
      {
        v6 = *(_DWORD *)(a1 + 808);
        v7 = 4096LL << (v6 >> 7);
        if ( v6 >> 7 >= 0x1E )
          v7 = 0x40000000000LL;
        v8 = malloc(v7);
        if ( !v8 )
        {
          sub_16BD1C0("Allocation failed", 1u);
          v6 = *(_DWORD *)(a1 + 808);
        }
        if ( v6 >= *(_DWORD *)(a1 + 812) )
        {
          sub_16CD150(a1 + 800, (const void *)(a1 + 816), 0, 8, v9, v10);
          v6 = *(_DWORD *)(a1 + 808);
        }
        v11 = v8 + v7;
        *(_QWORD *)(*(_QWORD *)(a1 + 800) + 8LL * v6) = v8;
        v12 = (v8 + 7) & 0xFFFFFFFFFFFFFFF8LL;
        ++*(_DWORD *)(a1 + 808);
        *(_QWORD *)(a1 + 792) = v11;
        *(_QWORD *)(a1 + 784) = v12 + 64;
      }
      v77 = (__int64)v3;
      *(_QWORD *)v12 = v12 + 16;
      *(_QWORD *)(v12 + 8) = 0x400000001LL;
      v13 = v77;
      *(_DWORD *)(v12 + 56) = 0;
      *(_QWORD *)(v12 + 48) = v72;
      *(_QWORD *)(v12 + 16) = v13;
      v14 = *(_DWORD *)(a1 + 912);
      if ( !v14 )
        break;
      v15 = *(_QWORD *)(a1 + 896);
      v16 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v17 = (__int64 *)(v15 + 16LL * v16);
      v18 = *v17;
      if ( v13 != *v17 )
      {
        v60 = 1;
        v61 = 0;
        while ( v18 != -8 )
        {
          if ( !v61 && v18 == -16 )
            v61 = v17;
          v16 = (v14 - 1) & (v16 + v60);
          v17 = (__int64 *)(v15 + 16LL * v16);
          v18 = *v17;
          if ( v13 == *v17 )
            goto LABEL_12;
          ++v60;
        }
        v62 = *(_DWORD *)(a1 + 904);
        if ( v61 )
          v17 = v61;
        ++*(_QWORD *)(a1 + 888);
        v63 = v62 + 1;
        if ( 4 * v63 < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(a1 + 908) - v63 <= v14 >> 3 )
          {
            sub_1DE4DF0(v72, v14);
            sub_1DE30F0(v72, &v77, &v78);
            v17 = v78;
            v13 = v77;
            v63 = *(_DWORD *)(a1 + 904) + 1;
          }
LABEL_67:
          *(_DWORD *)(a1 + 904) = v63;
          if ( *v17 != -8 )
            --*(_DWORD *)(a1 + 908);
          *v17 = v13;
          v17[1] = 0;
          goto LABEL_12;
        }
LABEL_76:
        sub_1DE4DF0(v72, 2 * v14);
        v64 = *(_DWORD *)(a1 + 912);
        if ( !v64 )
        {
          ++*(_DWORD *)(a1 + 904);
          BUG();
        }
        v13 = v77;
        v65 = v64 - 1;
        v66 = *(_QWORD *)(a1 + 896);
        v63 = *(_DWORD *)(a1 + 904) + 1;
        v67 = v65 & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
        v17 = (__int64 *)(v66 + 16LL * v67);
        v68 = *v17;
        if ( *v17 != v77 )
        {
          v69 = 1;
          v70 = 0;
          while ( v68 != -8 )
          {
            if ( !v70 && v68 == -16 )
              v70 = v17;
            v67 = v65 & (v67 + v69);
            v17 = (__int64 *)(v66 + 16LL * v67);
            v68 = *v17;
            if ( v77 == *v17 )
              goto LABEL_67;
            ++v69;
          }
          if ( v70 )
            v17 = v70;
        }
        goto LABEL_67;
      }
LABEL_12:
      v17[1] = v12;
      v19 = sub_1D820E0;
      while ( 1 )
      {
        v24 = *(_QWORD *)(a1 + 592);
        v76 = 0;
        LODWORD(v85) = 0;
        v75 = 0;
        v25 = *(__int64 (**)())(*(_QWORD *)v24 + 264LL);
        if ( v25 != sub_1D820E0
          && !((unsigned __int8 (__fastcall *)(__int64, __int64 *, __int64 *, __int64 *, _BYTE **, _QWORD))v25)(
                v24,
                v3,
                &v75,
                &v76,
                &v84,
                0) )
        {
          break;
        }
        if ( !sub_1DD6C00(v3) )
          break;
        v3 = (__int64 *)v3[1];
        v77 = (__int64)v3;
        v28 = *(unsigned int *)(v12 + 8);
        if ( (unsigned int)v28 >= *(_DWORD *)(v12 + 12) )
        {
          sub_16CD150(v12, (const void *)(v12 + 16), 0, 8, v26, v27);
          v28 = *(unsigned int *)(v12 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v12 + 8 * v28) = v77;
        v29 = *(_QWORD *)(v12 + 48);
        ++*(_DWORD *)(v12 + 8);
        v30 = *(_DWORD *)(v29 + 24);
        if ( v30 )
        {
          v20 = *(_QWORD *)(v29 + 8);
          v21 = (v30 - 1) & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
          v22 = (__int64 *)(v20 + 16LL * v21);
          v23 = *v22;
          if ( *v22 == v77 )
            goto LABEL_14;
          v53 = 1;
          v54 = 0;
          while ( v23 != -8 )
          {
            if ( v23 == -16 && !v54 )
              v54 = v22;
            v21 = (v30 - 1) & (v53 + v21);
            v22 = (__int64 *)(v20 + 16LL * v21);
            v23 = *v22;
            if ( v77 == *v22 )
              goto LABEL_14;
            ++v53;
          }
          v55 = *(_DWORD *)(v29 + 16);
          if ( v54 )
            v22 = v54;
          ++*(_QWORD *)v29;
          v31 = v55 + 1;
          if ( 4 * v31 < 3 * v30 )
          {
            if ( v30 - *(_DWORD *)(v29 + 20) - v31 > v30 >> 3 )
              goto LABEL_23;
            goto LABEL_22;
          }
        }
        else
        {
          ++*(_QWORD *)v29;
        }
        v30 *= 2;
LABEL_22:
        sub_1DE4DF0(v29, v30);
        sub_1DE30F0(v29, &v77, &v78);
        v22 = v78;
        v31 = *(_DWORD *)(v29 + 16) + 1;
LABEL_23:
        *(_DWORD *)(v29 + 16) = v31;
        if ( *v22 != -8 )
          --*(_DWORD *)(v29 + 20);
        v32 = v77;
        v22[1] = 0;
        *v22 = v32;
LABEL_14:
        v22[1] = v12;
      }
      v3 = (__int64 *)v3[1];
      if ( v71 == v3 )
        goto LABEL_28;
    }
    ++*(_QWORD *)(a1 + 888);
    goto LABEL_76;
  }
  v19 = sub_1D820E0;
LABEL_28:
  v33 = *(_QWORD *)(a1 + 576);
  *(_QWORD *)(a1 + 584) = 0;
  v34 = *(_QWORD ***)(v33 + 272);
  for ( i = *(_QWORD ***)(v33 + 264); v34 != i; ++i )
  {
    v36 = *i;
    sub_1DECA30(a1, v36);
  }
  v37 = *(_QWORD *)(a1 + 552);
  v78 = 0;
  v79 = v83;
  v80 = v83;
  v38 = v37 + 320;
  v81 = 4;
  v82 = 0;
  v39 = *(_QWORD *)(v38 + 8);
  if ( v38 != v39 )
  {
    do
    {
      sub_1DE5670(a1, v39, (__int64)&v78, 0);
      v39 = *(_QWORD *)(v39 + 8);
    }
    while ( v38 != v39 );
    v39 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 328LL);
  }
  v77 = v39;
  v40 = sub_1DE4FA0(v72, &v77)[1];
  sub_1DEB620(a1, *(_QWORD *)(*(_QWORD *)(a1 + 552) + 328LL), v40, 0);
  v41 = *(__int64 ***)v40;
  v42 = *(_QWORD *)(a1 + 552);
  v43 = *(unsigned __int64 **)(v42 + 328);
  if ( *(_QWORD *)v40 + 8LL * *(unsigned int *)(v40 + 8) != *(_QWORD *)v40 )
  {
    v44 = *(_QWORD *)v40 + 8LL * *(unsigned int *)(v40 + 8);
    v45 = sub_1D820E0;
    do
    {
      while ( 1 )
      {
        v52 = *v41;
        if ( *v41 == (__int64 *)v43 )
        {
          v43 = (unsigned __int64 *)v43[1];
        }
        else
        {
          v46 = (unsigned __int64 *)v52[1];
          if ( v43 != v46 && v46 != (unsigned __int64 *)v52 )
          {
            v47 = *v46 & 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)((*v52 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v46;
            *v46 = *v46 & 7 | *v52 & 0xFFFFFFFFFFFFFFF8LL;
            v48 = *v43;
            *(_QWORD *)(v47 + 8) = v43;
            v48 &= 0xFFFFFFFFFFFFFFF8LL;
            *v52 = v48 | *v52 & 7;
            *(_QWORD *)(v48 + 8) = v52;
            *v43 = v47 | *v43 & 7;
          }
        }
        if ( **(__int64 ***)v40 != v52 )
        {
          v49 = *(_QWORD *)(a1 + 592);
          v50 = *v52;
          v76 = 0;
          v77 = 0;
          LODWORD(v85) = 0;
          v51 = *(__int64 (**)())(*(_QWORD *)v49 + 264LL);
          if ( v51 != v45 )
          {
            v74 = v45;
            v73 = (__int64 *)(v50 & 0xFFFFFFFFFFFFFFF8LL);
            v56 = ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v51)(
                    v49,
                    v50 & 0xFFFFFFFFFFFFFFF8LL,
                    &v76,
                    &v77,
                    &v84,
                    0);
            v45 = v74;
            if ( !v56 )
              break;
          }
        }
        if ( (__int64 **)v44 == ++v41 )
          goto LABEL_53;
      }
      ++v41;
      sub_1DD7120(v73);
      v45 = v74;
    }
    while ( (__int64 **)v44 != v41 );
LABEL_53:
    v42 = *(_QWORD *)(a1 + 552);
    v19 = v45;
  }
  v57 = *(_QWORD *)(a1 + 592);
  v76 = 0;
  v77 = 0;
  LODWORD(v85) = 0;
  v58 = *(__int64 (**)())(*(_QWORD *)v57 + 264LL);
  if ( v58 != v19
    && !((unsigned __int8 (__fastcall *)(__int64, unsigned __int64, __int64 *, __int64 *, _BYTE **, _QWORD))v58)(
          v57,
          *(_QWORD *)(v42 + 320) & 0xFFFFFFFFFFFFFFF8LL,
          &v76,
          &v77,
          &v84,
          0) )
  {
    sub_1DD7120((__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 320LL) & 0xFFFFFFFFFFFFFFF8LL));
  }
  *(_DWORD *)(a1 + 240) = 0;
  v59 = (unsigned __int64)v80;
  *(_DWORD *)(a1 + 384) = 0;
  if ( (_BYTE *)v59 != v79 )
    _libc_free(v59);
  if ( v84 != v86 )
    _libc_free((unsigned __int64)v84);
}
