// Function: sub_16135E0
// Address: 0x16135e0
//
__int64 __fastcall sub_16135E0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r9
  __int64 v9; // r12
  int v11; // edx
  __int64 *v12; // r12
  __int64 *i; // r13
  __int64 v14; // rsi
  __int64 *v15; // r12
  __int64 *j; // r13
  __int64 v17; // rsi
  __int64 *v18; // r12
  __int64 *k; // r13
  __int64 v20; // rsi
  __int64 *v21; // r12
  __int64 *m; // r13
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // r9
  unsigned int v26; // esi
  __int64 v27; // rcx
  __int64 v28; // r8
  unsigned int v29; // edx
  __int64 *v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rdx
  unsigned __int64 v34; // r8
  __int64 v35; // rax
  unsigned __int64 v36; // r8
  __int64 v37; // rdx
  __int64 v38; // r12
  unsigned __int64 v39; // r8
  int v40; // esi
  int v41; // ecx
  int v42; // edx
  __int64 v43; // rdx
  int v44; // edi
  int v45; // r10d
  int v46; // r10d
  __int64 *v47; // r11
  int v48; // edx
  unsigned int v49; // [rsp+10h] [rbp-1C0h]
  unsigned __int64 v50; // [rsp+10h] [rbp-1C0h]
  unsigned __int64 v51; // [rsp+10h] [rbp-1C0h]
  __int64 v52; // [rsp+18h] [rbp-1B8h]
  __int64 v53; // [rsp+18h] [rbp-1B8h]
  __int64 v54; // [rsp+18h] [rbp-1B8h]
  unsigned __int64 v55; // [rsp+20h] [rbp-1B0h]
  unsigned __int64 v56; // [rsp+20h] [rbp-1B0h]
  unsigned __int64 v57; // [rsp+20h] [rbp-1B0h]
  unsigned __int64 v58; // [rsp+20h] [rbp-1B0h]
  __int64 v59; // [rsp+20h] [rbp-1B0h]
  __int64 v60; // [rsp+20h] [rbp-1B0h]
  __int64 v61; // [rsp+48h] [rbp-188h] BYREF
  __int64 v62; // [rsp+50h] [rbp-180h] BYREF
  __int64 *v63; // [rsp+58h] [rbp-178h] BYREF
  unsigned __int64 v64[2]; // [rsp+60h] [rbp-170h] BYREF
  _BYTE v65[128]; // [rsp+70h] [rbp-160h] BYREF
  __int64 *v66; // [rsp+F0h] [rbp-E0h] BYREF
  __int64 v67; // [rsp+F8h] [rbp-D8h]
  _BYTE v68[64]; // [rsp+100h] [rbp-D0h] BYREF
  __int64 *v69; // [rsp+140h] [rbp-90h] BYREF
  __int64 v70; // [rsp+148h] [rbp-88h]
  _BYTE v71[16]; // [rsp+150h] [rbp-80h] BYREF
  __int64 *v72; // [rsp+160h] [rbp-70h] BYREF
  __int64 v73; // [rsp+168h] [rbp-68h]
  _BYTE v74[16]; // [rsp+170h] [rbp-60h] BYREF
  __int64 *v75; // [rsp+180h] [rbp-50h] BYREF
  __int64 v76; // [rsp+188h] [rbp-48h]
  _BYTE v77[64]; // [rsp+190h] [rbp-40h] BYREF

  v4 = *(unsigned int *)(a1 + 696);
  v61 = a2;
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 680);
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16 * v4) )
        return v7[1];
    }
    else
    {
      v11 = 1;
      while ( v8 != -8 )
      {
        v45 = v11 + 1;
        v6 = (v4 - 1) & (v11 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        v11 = v45;
      }
    }
  }
  v76 = 0;
  v67 = 0x800000000LL;
  v69 = (__int64 *)v71;
  v70 = 0x200000000LL;
  v73 = 0x200000000LL;
  v72 = (__int64 *)v74;
  v66 = (__int64 *)v68;
  v75 = (__int64 *)v77;
  v77[0] = 0;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a2 + 88LL))(a2);
  v64[0] = (unsigned __int64)v65;
  v64[1] = 0x2000000000LL;
  sub_16BD430(v64, v77[0]);
  sub_16BD4B0(v64, (unsigned int)v67);
  v12 = &v66[(unsigned int)v67];
  for ( i = v66; v12 != i; ++i )
  {
    v14 = *i;
    sub_16BD4C0(v64, v14);
  }
  sub_16BD4B0(v64, (unsigned int)v70);
  v15 = &v69[(unsigned int)v70];
  for ( j = v69; v15 != j; ++j )
  {
    v17 = *j;
    sub_16BD4C0(v64, v17);
  }
  sub_16BD4B0(v64, (unsigned int)v73);
  v18 = &v72[(unsigned int)v73];
  for ( k = v72; v18 != k; ++k )
  {
    v20 = *k;
    sub_16BD4C0(v64, v20);
  }
  sub_16BD4B0(v64, (unsigned int)v76);
  v21 = &v75[(unsigned int)v76];
  for ( m = v75; v21 != m; ++m )
  {
    v23 = *m;
    sub_16BD4C0(v64, v23);
  }
  v62 = 0;
  v24 = sub_16BDDE0(a1 + 544, v64, &v62);
  v25 = a1 + 544;
  v9 = v24 + 8;
  if ( !v24 )
  {
    v32 = *(_QWORD *)(a1 + 568);
    v33 = *(_QWORD *)(a1 + 576);
    *(_QWORD *)(a1 + 648) += 176LL;
    if ( ((v32 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v32 + 176 <= v33 - v32 )
    {
      v39 = (v32 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(a1 + 568) = v39 + 176;
    }
    else
    {
      v34 = 0x40000000000LL;
      v49 = *(_DWORD *)(a1 + 592);
      if ( v49 >> 7 < 0x1E )
        v34 = 4096LL << (v49 >> 7);
      v55 = v34;
      v35 = malloc(v34);
      v36 = v55;
      v37 = v49;
      v25 = a1 + 544;
      v38 = v35;
      if ( !v35 )
      {
        sub_16BD1C0("Allocation failed");
        v37 = *(unsigned int *)(a1 + 592);
        v36 = v55;
        v25 = a1 + 544;
      }
      if ( (unsigned int)v37 >= *(_DWORD *)(a1 + 596) )
      {
        v51 = v36;
        v60 = v25;
        sub_16CD150(a1 + 584, a1 + 600, 0, 8);
        v37 = *(unsigned int *)(a1 + 592);
        v36 = v51;
        v25 = v60;
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 584) + 8 * v37) = v38;
      *(_QWORD *)(a1 + 576) = v38 + v36;
      v39 = (v38 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      ++*(_DWORD *)(a1 + 592);
      *(_QWORD *)(a1 + 568) = v39 + 176;
    }
    *(_QWORD *)v39 = 0;
    v9 = v39 + 8;
    *(_QWORD *)(v39 + 8) = v39 + 24;
    v40 = v67;
    *(_QWORD *)(v39 + 16) = 0x800000000LL;
    if ( v40 )
    {
      v50 = v39;
      v59 = v25;
      sub_160D2F0(v39 + 8, (__int64)&v66);
      v39 = v50;
      v25 = v59;
    }
    *(_QWORD *)(v39 + 88) = v39 + 104;
    v41 = v70;
    *(_QWORD *)(v39 + 96) = 0x200000000LL;
    if ( v41 )
    {
      v54 = v25;
      v58 = v39;
      sub_160D2F0(v39 + 88, (__int64)&v69);
      v25 = v54;
      v39 = v58;
    }
    *(_QWORD *)(v39 + 120) = v39 + 136;
    v42 = v73;
    *(_QWORD *)(v39 + 128) = 0x200000000LL;
    if ( v42 )
    {
      v53 = v25;
      v57 = v39;
      sub_160D2F0(v39 + 120, (__int64)&v72);
      v25 = v53;
      v39 = v57;
    }
    *(_QWORD *)(v39 + 160) = 0;
    *(_QWORD *)(v39 + 152) = v39 + 168;
    if ( (_DWORD)v76 )
    {
      v52 = v25;
      v56 = v39;
      sub_160D2F0(v39 + 152, (__int64)&v75);
      v25 = v52;
      v39 = v56;
    }
    v43 = v62;
    *(_BYTE *)(v39 + 168) = v77[0];
    sub_16BDA20(v25, v39, v43);
  }
  v26 = *(_DWORD *)(a1 + 696);
  if ( !v26 )
  {
    ++*(_QWORD *)(a1 + 672);
    goto LABEL_47;
  }
  v27 = v61;
  v28 = *(_QWORD *)(a1 + 680);
  v29 = (v26 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
  v30 = (__int64 *)(v28 + 16LL * v29);
  v31 = *v30;
  if ( *v30 != v61 )
  {
    v46 = 1;
    v47 = 0;
    while ( v31 != -8 )
    {
      if ( v31 == -16 && !v47 )
        v47 = v30;
      v29 = (v26 - 1) & (v46 + v29);
      v30 = (__int64 *)(v28 + 16LL * v29);
      v31 = *v30;
      if ( v61 == *v30 )
        goto LABEL_19;
      ++v46;
    }
    v48 = *(_DWORD *)(a1 + 688);
    if ( v47 )
      v30 = v47;
    ++*(_QWORD *)(a1 + 672);
    v44 = v48 + 1;
    if ( 4 * (v48 + 1) < 3 * v26 )
    {
      if ( v26 - *(_DWORD *)(a1 + 692) - v44 > v26 >> 3 )
        goto LABEL_49;
      goto LABEL_48;
    }
LABEL_47:
    v26 *= 2;
LABEL_48:
    sub_1613420(a1 + 672, v26);
    sub_1612A40(a1 + 672, &v61, &v63);
    v30 = v63;
    v27 = v61;
    v44 = *(_DWORD *)(a1 + 688) + 1;
LABEL_49:
    *(_DWORD *)(a1 + 688) = v44;
    if ( *v30 != -8 )
      --*(_DWORD *)(a1 + 692);
    *v30 = v27;
    v30[1] = 0;
  }
LABEL_19:
  v30[1] = v9;
  if ( (_BYTE *)v64[0] != v65 )
    _libc_free(v64[0]);
  if ( v75 != (__int64 *)v77 )
    _libc_free((unsigned __int64)v75);
  if ( v72 != (__int64 *)v74 )
    _libc_free((unsigned __int64)v72);
  if ( v69 != (__int64 *)v71 )
    _libc_free((unsigned __int64)v69);
  if ( v66 != (__int64 *)v68 )
    _libc_free((unsigned __int64)v66);
  return v9;
}
