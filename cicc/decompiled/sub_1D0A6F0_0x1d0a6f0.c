// Function: sub_1D0A6F0
// Address: 0x1d0a6f0
//
void *__fastcall sub_1D0A6F0(__int64 a1)
{
  char v2; // al
  __int64 v3; // r12
  void *v4; // rax
  void *v5; // rcx
  __int64 v6; // rdi
  __int64 v7; // r12
  void *v8; // rax
  void *v9; // rcx
  __int64 v10; // rdi
  int v11; // eax
  __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *i; // rdx
  void (*v15)(void); // rax
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdi
  void (*v19)(void); // rax
  int v20; // eax
  _BYTE *v21; // r8
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  const void *v24; // r15
  unsigned __int64 v25; // rcx
  __int64 v26; // r14
  __int64 v27; // rax
  char *v28; // r12
  signed __int64 v29; // rdx
  _QWORD *v30; // rdi
  bool (__fastcall *v31)(__int64); // rax
  __int64 v32; // rax
  __int64 v33; // r13
  unsigned int v34; // esi
  __int64 v35; // rdi
  __int64 (*v36)(); // rax
  unsigned int v37; // r9d
  _BYTE *v38; // rsi
  __int64 v39; // r8
  __int64 v40; // rcx
  __int64 v41; // rdi
  __int64 v42; // r15
  __int64 v43; // rax
  __int64 k; // r13
  __int64 v45; // rax
  __int64 v46; // r13
  _QWORD *v47; // rax
  _QWORD *m; // rdi
  unsigned __int64 v49; // rdx
  char v50; // si
  _DWORD *v51; // rdi
  __int64 (*v52)(); // rax
  unsigned int v53; // eax
  unsigned int v54; // esi
  unsigned int v55; // ecx
  _QWORD *v56; // rdi
  unsigned int v57; // eax
  int v58; // eax
  unsigned __int64 v59; // rax
  unsigned __int64 v60; // rax
  int v61; // r13d
  __int64 v62; // r12
  _QWORD *v63; // rax
  __int64 v64; // rdx
  _QWORD *j; // rdx
  __int64 *v66; // rax
  __int64 *v67; // rdx
  __int64 *n; // rax
  __int64 v69; // rcx
  __int64 v70; // rsi
  _QWORD *v71; // rdi
  __int64 (*v72)(void); // rax
  void *result; // rax
  _BYTE *v74; // rdx
  _BYTE *v75; // rdi
  unsigned int v76; // eax
  __int64 v77; // r15
  int v78; // eax
  unsigned int *v79; // rax
  __int64 v80; // rax
  unsigned int v81; // eax
  int v82; // r15d
  __int64 v83; // rsi
  _QWORD *v84; // rax
  unsigned __int64 v85; // [rsp+0h] [rbp-50h]
  unsigned __int64 v86; // [rsp+0h] [rbp-50h]
  unsigned __int64 v87; // [rsp+0h] [rbp-50h]
  __int64 v88; // [rsp+8h] [rbp-48h]
  __int64 v89[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = byte_4FC13A0;
  *(_DWORD *)(a1 + 712) = 0;
  *(_QWORD *)(a1 + 720) = 0;
  *(_DWORD *)(a1 + 716) = -((unsigned __int8)v2 ^ 1);
  v3 = (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 24) + 16LL) + 1);
  v4 = (void *)sub_2207820(8 * v3);
  v5 = v4;
  if ( v4 && v3 )
    v5 = memset(v4, 0, 8 * v3);
  v6 = *(_QWORD *)(a1 + 728);
  *(_QWORD *)(a1 + 728) = v5;
  if ( v6 )
    j_j___libc_free_0_0(v6);
  v7 = (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 24) + 16LL) + 1);
  v8 = (void *)sub_2207820(8 * v7);
  v9 = v8;
  if ( v8 && v7 )
    v9 = memset(v8, 0, 8 * v7);
  v10 = *(_QWORD *)(a1 + 736);
  *(_QWORD *)(a1 + 736) = v9;
  if ( v10 )
    j_j___libc_free_0_0(v10);
  v11 = *(_DWORD *)(a1 + 928);
  ++*(_QWORD *)(a1 + 912);
  if ( !v11 )
  {
    if ( !*(_DWORD *)(a1 + 932) )
      goto LABEL_17;
    v12 = *(unsigned int *)(a1 + 936);
    if ( (unsigned int)v12 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 920));
      *(_QWORD *)(a1 + 920) = 0;
      *(_QWORD *)(a1 + 928) = 0;
      *(_DWORD *)(a1 + 936) = 0;
      goto LABEL_17;
    }
    goto LABEL_14;
  }
  v55 = 4 * v11;
  v12 = *(unsigned int *)(a1 + 936);
  if ( (unsigned int)(4 * v11) < 0x40 )
    v55 = 64;
  if ( (unsigned int)v12 <= v55 )
  {
LABEL_14:
    v13 = *(_QWORD **)(a1 + 920);
    for ( i = &v13[2 * v12]; i != v13; v13 += 2 )
      *v13 = -8;
    *(_QWORD *)(a1 + 928) = 0;
    goto LABEL_17;
  }
  v56 = *(_QWORD **)(a1 + 920);
  v57 = v11 - 1;
  if ( !v57 )
  {
    v62 = 2048;
    v61 = 128;
LABEL_81:
    j___libc_free_0(v56);
    *(_DWORD *)(a1 + 936) = v61;
    v63 = (_QWORD *)sub_22077B0(v62);
    v64 = *(unsigned int *)(a1 + 936);
    *(_QWORD *)(a1 + 928) = 0;
    *(_QWORD *)(a1 + 920) = v63;
    for ( j = &v63[2 * v64]; j != v63; v63 += 2 )
    {
      if ( v63 )
        *v63 = -8;
    }
    goto LABEL_17;
  }
  _BitScanReverse(&v57, v57);
  v58 = 1 << (33 - (v57 ^ 0x1F));
  if ( v58 < 64 )
    v58 = 64;
  if ( (_DWORD)v12 != v58 )
  {
    v59 = (4 * v58 / 3u + 1) | ((unsigned __int64)(4 * v58 / 3u + 1) >> 1);
    v60 = ((v59 | (v59 >> 2)) >> 4) | v59 | (v59 >> 2) | ((((v59 | (v59 >> 2)) >> 4) | v59 | (v59 >> 2)) >> 8);
    v61 = (v60 | (v60 >> 16)) + 1;
    v62 = 16 * ((v60 | (v60 >> 16)) + 1);
    goto LABEL_81;
  }
  *(_QWORD *)(a1 + 928) = 0;
  v84 = &v56[2 * (unsigned int)v12];
  do
  {
    if ( v56 )
      *v56 = -8;
    v56 += 2;
  }
  while ( v84 != v56 );
LABEL_17:
  sub_1D10D90(a1, 0);
  sub_1F02930(a1 + 824);
  (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 672) + 32LL))(*(_QWORD *)(a1 + 672), a1 + 48);
  v15 = *(void (**)(void))(**(_QWORD **)(a1 + 704) + 32LL);
  if ( v15 != nullsub_678 )
    v15();
  sub_1D06C30(a1, a1 + 344);
  v16 = *(_QWORD *)(a1 + 48);
  if ( *(_QWORD *)(a1 + 56) == v16 )
  {
    v88 = a1 + 640;
  }
  else
  {
    v17 = v16 + 272LL * *(int *)(*(_QWORD *)(*(_QWORD *)(a1 + 624) + 176LL) + 28LL);
    *(_BYTE *)(v17 + 229) |= 2u;
    v18 = *(_QWORD *)(a1 + 672);
    v19 = *(void (**)(void))(*(_QWORD *)v18 + 88LL);
    if ( (char *)v19 == (char *)sub_1D047D0 )
    {
      v89[0] = v17;
      v20 = *(_DWORD *)(v18 + 40) + 1;
      *(_DWORD *)(v18 + 40) = v20;
      *(_DWORD *)(v17 + 196) = v20;
      v21 = *(_BYTE **)(v18 + 24);
      if ( v21 == *(_BYTE **)(v18 + 32) )
      {
        sub_1CFD630(v18 + 16, v21, v89);
      }
      else
      {
        if ( v21 )
        {
          *(_QWORD *)v21 = v17;
          v21 = *(_BYTE **)(v18 + 24);
        }
        *(_QWORD *)(v18 + 24) = v21 + 8;
      }
    }
    else
    {
      v19();
    }
    v88 = a1 + 640;
    v22 = *(_QWORD *)(a1 + 56) - *(_QWORD *)(a1 + 48);
    v23 = 0xF0F0F0F0F0F0F0F1LL * (v22 >> 4);
    if ( v22 < 0 )
      sub_4262D8((__int64)"vector::reserve");
    v24 = *(const void **)(a1 + 640);
    if ( v23 > (__int64)(*(_QWORD *)(a1 + 656) - (_QWORD)v24) >> 3 )
    {
      v25 = 0x8787878787878788LL * (v22 >> 4);
      v26 = *(_QWORD *)(a1 + 648) - (_QWORD)v24;
      if ( v23 )
      {
        v85 = 0x8787878787878788LL * (v22 >> 4);
        v27 = sub_22077B0(v85);
        v24 = *(const void **)(a1 + 640);
        v25 = v85;
        v28 = (char *)v27;
        v29 = *(_QWORD *)(a1 + 648) - (_QWORD)v24;
        if ( v29 <= 0 )
        {
LABEL_29:
          if ( !v24 )
          {
LABEL_30:
            *(_QWORD *)(a1 + 640) = v28;
            *(_QWORD *)(a1 + 648) = &v28[v26];
            *(_QWORD *)(a1 + 656) = &v28[v25];
            goto LABEL_31;
          }
          v83 = *(_QWORD *)(a1 + 656) - (_QWORD)v24;
LABEL_129:
          v87 = v25;
          j_j___libc_free_0(v24, v83);
          v25 = v87;
          goto LABEL_30;
        }
      }
      else
      {
        v29 = *(_QWORD *)(a1 + 648) - (_QWORD)v24;
        v28 = 0;
        if ( v26 <= 0 )
          goto LABEL_29;
      }
      v86 = v25;
      memmove(v28, v24, v29);
      v25 = v86;
      v83 = *(_QWORD *)(a1 + 656) - (_QWORD)v24;
      goto LABEL_129;
    }
  }
LABEL_31:
  v30 = *(_QWORD **)(a1 + 672);
  v31 = *(bool (__fastcall **)(__int64))(*v30 + 64LL);
LABEL_32:
  if ( v31 == sub_1D00EA0 )
    goto LABEL_33;
  while ( 2 )
  {
    if ( ((unsigned __int8 (*)(void))v31)() && !*(_DWORD *)(a1 + 752) )
      goto LABEL_90;
LABEL_35:
    v32 = sub_1D07BB0(a1);
    v33 = v32;
    if ( byte_4FC13A0 )
      goto LABEL_42;
    if ( (*(_BYTE *)(v32 + 236) & 2) == 0 )
      sub_1F01F70(v32);
    v34 = *(_DWORD *)(v33 + 244);
    if ( v34 > *(_DWORD *)(a1 + 712) )
      sub_1D04A20(a1, v34);
    if ( (*(_BYTE *)(v33 + 228) & 2) != 0 )
      goto LABEL_42;
    v35 = *(_QWORD *)(a1 + 704);
    v36 = *(__int64 (**)())(*(_QWORD *)v35 + 24LL);
    if ( v36 == sub_1D00B90 )
      goto LABEL_42;
    v82 = 0;
    do
    {
      if ( !((unsigned int (__fastcall *)(__int64, __int64, _QWORD))v36)(v35, v33, (unsigned int)-v82) )
        break;
      v35 = *(_QWORD *)(a1 + 704);
      ++v82;
      v36 = *(__int64 (**)())(*(_QWORD *)v35 + 24LL);
    }
    while ( v36 != sub_1D00B90 );
    v37 = *(_DWORD *)(a1 + 712);
    if ( v37 < v82 + v37 )
    {
      sub_1D04A20(a1, v82 + v37);
LABEL_42:
      v37 = *(_DWORD *)(a1 + 712);
    }
    v89[0] = v33;
    sub_1F020C0(v33, v37);
    sub_1D013C0(a1, v89[0]);
    v38 = *(_BYTE **)(a1 + 648);
    if ( v38 == *(_BYTE **)(a1 + 656) )
    {
      sub_1CFD630(v88, v38, v89);
      v39 = v89[0];
    }
    else
    {
      v39 = v89[0];
      if ( v38 )
      {
        *(_QWORD *)v38 = v89[0];
        v38 = *(_BYTE **)(a1 + 648);
      }
      *(_QWORD *)(a1 + 648) = v38 + 8;
    }
    (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 672) + 120LL))(*(_QWORD *)(a1 + 672), v39);
    if ( !*(_DWORD *)(*(_QWORD *)(a1 + 704) + 8LL) && (unsigned int)dword_4FC0AE0 <= 1 )
    {
      v81 = *(_DWORD *)(a1 + 712);
      if ( v81 < v81 + 1 )
        sub_1D04A20(a1, v81 + 1);
    }
    sub_1D06C30(a1, v89[0]);
    v40 = v89[0];
    v41 = *(_QWORD *)(a1 + 728);
    v42 = *(_QWORD *)(v89[0] + 112);
    v43 = 16LL * *(unsigned int *)(v89[0] + 120);
    for ( k = v42 + v43; k != v42; v42 += 16 )
    {
      if ( (*(_BYTE *)v42 & 6) == 0 )
      {
        v45 = *(unsigned int *)(v42 + 8);
        if ( (_DWORD)v45 )
        {
          if ( *(_QWORD *)(v41 + 8 * v45) == v40 )
          {
            --*(_DWORD *)(a1 + 724);
            *(_QWORD *)(v41 + 8LL * *(unsigned int *)(v42 + 8)) = 0;
            *(_QWORD *)(*(_QWORD *)(a1 + 736) + 8LL * *(unsigned int *)(v42 + 8)) = 0;
            sub_1D04AE0(a1, *(_DWORD *)(v42 + 8));
            v41 = *(_QWORD *)(a1 + 728);
            v40 = v89[0];
          }
        }
      }
    }
    v46 = *(unsigned int *)(*(_QWORD *)(a1 + 24) + 16LL);
    if ( *(_QWORD *)(v41 + 8 * v46) == v40 )
    {
      v77 = *(_QWORD *)v40;
      do
      {
        if ( !v77 )
          break;
        if ( *(__int16 *)(v77 + 24) < 0 && ~*(__int16 *)(v77 + 24) == *(_DWORD *)(*(_QWORD *)(a1 + 16) + 36LL) )
        {
          v80 = *(_QWORD *)(a1 + 728);
          --*(_DWORD *)(a1 + 724);
          *(_QWORD *)(v80 + 8 * v46) = 0;
          *(_QWORD *)(*(_QWORD *)(a1 + 736) + 8 * v46) = 0;
          sub_1D04AE0(a1, v46);
        }
        v78 = *(_DWORD *)(v77 + 56);
        if ( !v78 )
          break;
        v79 = (unsigned int *)(*(_QWORD *)(v77 + 32) + 40LL * (unsigned int)(v78 - 1));
        v77 = *(_QWORD *)v79;
      }
      while ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v79 + 40LL) + 16LL * v79[2]) == 111 );
      v40 = v89[0];
    }
    if ( (*(_BYTE *)(v40 + 228) & 1) != 0 )
    {
      v47 = *(_QWORD **)(v40 + 32);
      for ( m = &v47[2 * *(unsigned int *)(v40 + 40)]; v47 != m; v47 += 2 )
      {
        if ( (*v47 & 6) == 0 )
        {
          v49 = *v47 & 0xFFFFFFFFFFFFFFF8LL;
          v50 = *(_BYTE *)(v49 + 228);
          if ( (v50 & 1) != 0 )
            *(_BYTE *)(v49 + 228) = v50 & 0xFE;
        }
      }
    }
    *(_BYTE *)(v40 + 229) |= 4u;
    v51 = *(_DWORD **)(a1 + 704);
    if ( v51[2] )
    {
      if ( !*(_QWORD *)v40 || *(__int16 *)(*(_QWORD *)v40 + 24LL) >= 0 )
      {
        v52 = *(__int64 (**)())(*(_QWORD *)v51 + 16LL);
        if ( v52 == sub_1D00B80 )
          goto LABEL_66;
        goto LABEL_114;
      }
    }
    else
    {
      if ( (unsigned int)dword_4FC0AE0 <= 1 )
        goto LABEL_66;
      if ( !*(_QWORD *)v40 || *(__int16 *)(*(_QWORD *)v40 + 24LL) >= 0 )
      {
LABEL_101:
        if ( *(_DWORD *)(a1 + 720) != dword_4FC0AE0 )
          goto LABEL_66;
        goto LABEL_102;
      }
    }
    ++*(_DWORD *)(a1 + 720);
    if ( !v51[2] )
      goto LABEL_101;
    v52 = *(__int64 (**)())(*(_QWORD *)v51 + 16LL);
    if ( v52 == sub_1D00B80 )
      goto LABEL_66;
LABEL_114:
    if ( !(unsigned __int8)v52() )
    {
      if ( *(_DWORD *)(*(_QWORD *)(a1 + 704) + 8LL) )
        goto LABEL_66;
      goto LABEL_101;
    }
LABEL_102:
    v76 = *(_DWORD *)(a1 + 712);
    v54 = v76 + 1;
    if ( v76 < v76 + 1 )
      goto LABEL_72;
LABEL_66:
    while ( 1 )
    {
      v30 = *(_QWORD **)(a1 + 672);
      v31 = *(bool (__fastcall **)(__int64))(*v30 + 64LL);
      if ( v31 != sub_1D00EA0 )
        break;
      if ( v30[3] != v30[2] )
        goto LABEL_32;
LABEL_68:
      if ( *(_QWORD *)(a1 + 688) == *(_QWORD *)(a1 + 680) )
        goto LABEL_87;
      v53 = *(_DWORD *)(a1 + 712);
      v54 = v53 + 1;
      if ( v53 + 1 < *(_DWORD *)(a1 + 716) )
        v54 = *(_DWORD *)(a1 + 716);
      if ( v53 < v54 )
LABEL_72:
        sub_1D04A20(a1, v54);
    }
    if ( ((unsigned __int8 (*)(void))v31)() )
      goto LABEL_68;
LABEL_87:
    v30 = *(_QWORD **)(a1 + 672);
    v31 = *(bool (__fastcall **)(__int64))(*v30 + 64LL);
    if ( v31 != sub_1D00EA0 )
      continue;
    break;
  }
LABEL_33:
  if ( v30[3] != v30[2] || *(_DWORD *)(a1 + 752) )
    goto LABEL_35;
LABEL_90:
  v66 = *(__int64 **)(a1 + 648);
  v67 = *(__int64 **)(a1 + 640);
  if ( v66 != v67 )
  {
    for ( n = v66 - 1; v67 < n; n[1] = v69 )
    {
      v69 = *v67;
      v70 = *n;
      ++v67;
      --n;
      *(v67 - 1) = v70;
    }
  }
  v71 = *(_QWORD **)(a1 + 672);
  v72 = *(__int64 (**)(void))(*v71 + 56LL);
  if ( (char *)v72 != (char *)sub_1D00E60 )
    return (void *)v72();
  v71[6] = 0;
  result = (void *)v71[12];
  if ( result != (void *)v71[13] )
    v71[13] = result;
  v74 = (_BYTE *)v71[16];
  v75 = (_BYTE *)v71[15];
  if ( v74 != v75 )
    return memset(v75, 0, v74 - v75);
  return result;
}
