// Function: sub_234E7B0
// Address: 0x234e7b0
//
_QWORD *__fastcall sub_234E7B0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // r13
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rax
  _QWORD *v23; // rdx
  unsigned __int64 v24; // r12
  __int64 v25; // r15
  __int64 *v26; // rbx
  __int64 *v27; // r14
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned int v31; // eax
  __int64 v32; // rdx
  char v33; // al
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rdi
  __int64 v36; // rsi
  __int64 v37; // rdx
  _QWORD *v38; // rax
  _QWORD *v39; // rdx
  unsigned __int64 v40; // r13
  unsigned __int64 v41; // r12
  __int64 v42; // r14
  __int64 *v43; // r15
  __int64 *v44; // rbx
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rax
  unsigned int v48; // eax
  __int64 v49; // rdx
  char v50; // al
  unsigned __int64 v51; // rdi
  unsigned __int64 v52; // rdi
  unsigned int v54; // ecx
  _QWORD *v55; // rdi
  __int64 v56; // r8
  unsigned int v57; // eax
  int v58; // r12d
  unsigned int v59; // eax
  _QWORD *v60; // rax
  _QWORD *j; // rdx
  unsigned int v62; // ecx
  _QWORD *v63; // rdi
  __int64 v64; // r8
  unsigned int v65; // eax
  int v66; // r12d
  unsigned int v67; // eax
  _QWORD *v68; // rax
  _QWORD *i; // rdx
  _QWORD *v70; // r8
  _QWORD *v71; // r8
  unsigned __int64 v73; // [rsp+18h] [rbp-178h]
  __int64 v74; // [rsp+20h] [rbp-170h] BYREF
  _QWORD *v75; // [rsp+28h] [rbp-168h]
  __int64 v76; // [rsp+30h] [rbp-160h]
  unsigned int v77; // [rsp+38h] [rbp-158h]
  unsigned __int64 v78; // [rsp+40h] [rbp-150h]
  unsigned __int64 v79; // [rsp+48h] [rbp-148h]
  __int64 v80; // [rsp+50h] [rbp-140h]
  _BYTE v81[104]; // [rsp+58h] [rbp-138h] BYREF
  __int64 v82; // [rsp+C0h] [rbp-D0h]
  _QWORD *v83; // [rsp+C8h] [rbp-C8h]
  __int64 v84; // [rsp+D0h] [rbp-C0h]
  unsigned int v85; // [rsp+D8h] [rbp-B8h]
  unsigned __int64 v86; // [rsp+E0h] [rbp-B0h]
  unsigned __int64 v87; // [rsp+E8h] [rbp-A8h]
  __int64 v88; // [rsp+F0h] [rbp-A0h]
  _BYTE v89[152]; // [rsp+F8h] [rbp-98h] BYREF

  sub_D51E20((__int64)&v74, a2 + 8, a3, a4);
  v4 = (__int64)v81;
  ++v74;
  v82 = 1;
  v83 = v75;
  v75 = 0;
  v84 = v76;
  v76 = 0;
  v85 = v77;
  v77 = 0;
  v86 = v78;
  v78 = 0;
  v87 = v79;
  v79 = 0;
  v88 = v80;
  v80 = 0;
  sub_234E5E0((__int64)v89, (__int64)v81, v5, v6, v7, v8);
  if ( v78 != v79 )
    v79 = v78;
  v9 = (_QWORD *)sub_22077B0(0xA0u);
  v14 = v9;
  if ( v9 )
  {
    v9[1] = 1;
    v15 = (__int64)(v9 + 8);
    v4 = (__int64)v89;
    ++v82;
    *v9 = &unk_4A15908;
    v16 = (__int64)v83;
    v83 = 0;
    v14[2] = v16;
    v17 = v84;
    v84 = 0;
    v14[3] = v17;
    LODWORD(v17) = v85;
    v85 = 0;
    *((_DWORD *)v14 + 8) = v17;
    v18 = v86;
    v86 = 0;
    v14[5] = v18;
    v19 = v87;
    v87 = 0;
    v14[6] = v19;
    v20 = v88;
    v88 = 0;
    v14[7] = v20;
    sub_234E5E0(v15, (__int64)v89, v10, v11, v12, v13);
    if ( v86 != v87 )
      v87 = v86;
  }
  ++v82;
  if ( !(_DWORD)v84 )
  {
    if ( !HIDWORD(v84) )
      goto LABEL_12;
    v21 = v85;
    if ( v85 <= 0x40 )
      goto LABEL_9;
    v4 = 16LL * v85;
    sub_C7D6A0((__int64)v83, v4, 8);
    v85 = 0;
LABEL_99:
    v83 = 0;
LABEL_11:
    v84 = 0;
    goto LABEL_12;
  }
  v62 = 4 * v84;
  v4 = 64;
  v21 = v85;
  if ( (unsigned int)(4 * v84) < 0x40 )
    v62 = 64;
  if ( v85 <= v62 )
  {
LABEL_9:
    v22 = v83;
    v23 = &v83[2 * v21];
    if ( v83 != v23 )
    {
      do
      {
        *v22 = -4096;
        v22 += 2;
      }
      while ( v23 != v22 );
    }
    goto LABEL_11;
  }
  v63 = v83;
  v64 = 2LL * v85;
  if ( (_DWORD)v84 == 1 )
  {
    v66 = 64;
  }
  else
  {
    _BitScanReverse(&v65, v84 - 1);
    v66 = 1 << (33 - (v65 ^ 0x1F));
    if ( v66 < 64 )
      v66 = 64;
    if ( v85 == v66 )
    {
      v84 = 0;
      v71 = &v83[v64];
      do
      {
        if ( v63 )
          *v63 = -4096;
        v63 += 2;
      }
      while ( v71 != v63 );
      goto LABEL_12;
    }
  }
  v4 = 16LL * v85;
  sub_C7D6A0((__int64)v83, v64 * 8, 8);
  v67 = sub_2309150(v66);
  v85 = v67;
  if ( !v67 )
    goto LABEL_99;
  v4 = 8;
  v68 = (_QWORD *)sub_C7D670(16LL * v67, 8);
  v84 = 0;
  v83 = v68;
  for ( i = &v68[2 * v85]; i != v68; v68 += 2 )
  {
    if ( v68 )
      *v68 = -4096;
  }
LABEL_12:
  v24 = v86;
  v73 = v87;
  if ( v86 != v87 )
  {
    do
    {
      v25 = *(_QWORD *)v24;
      v26 = *(__int64 **)(*(_QWORD *)v24 + 16LL);
      if ( *(__int64 **)(*(_QWORD *)v24 + 8LL) == v26 )
      {
        *(_BYTE *)(v25 + 152) = 1;
      }
      else
      {
        v27 = *(__int64 **)(*(_QWORD *)v24 + 8LL);
        do
        {
          v28 = *v27++;
          sub_D47BB0(v28, v4);
        }
        while ( v26 != v27 );
        *(_BYTE *)(v25 + 152) = 1;
        v29 = *(_QWORD *)(v25 + 8);
        if ( *(_QWORD *)(v25 + 16) != v29 )
          *(_QWORD *)(v25 + 16) = v29;
      }
      v30 = *(_QWORD *)(v25 + 32);
      if ( v30 != *(_QWORD *)(v25 + 40) )
        *(_QWORD *)(v25 + 40) = v30;
      ++*(_QWORD *)(v25 + 56);
      if ( *(_BYTE *)(v25 + 84) )
      {
        *(_QWORD *)v25 = 0;
      }
      else
      {
        v31 = 4 * (*(_DWORD *)(v25 + 76) - *(_DWORD *)(v25 + 80));
        v32 = *(unsigned int *)(v25 + 72);
        if ( v31 < 0x20 )
          v31 = 32;
        if ( (unsigned int)v32 > v31 )
        {
          sub_C8C990(v25 + 56, v4);
        }
        else
        {
          v4 = 0xFFFFFFFFLL;
          memset(*(void **)(v25 + 64), -1, 8 * v32);
        }
        v33 = *(_BYTE *)(v25 + 84);
        *(_QWORD *)v25 = 0;
        if ( !v33 )
          _libc_free(*(_QWORD *)(v25 + 64));
      }
      v34 = *(_QWORD *)(v25 + 32);
      if ( v34 )
      {
        v4 = *(_QWORD *)(v25 + 48) - v34;
        j_j___libc_free_0(v34);
      }
      v35 = *(_QWORD *)(v25 + 8);
      if ( v35 )
      {
        v4 = *(_QWORD *)(v25 + 24) - v35;
        j_j___libc_free_0(v35);
      }
      v24 += 8LL;
    }
    while ( v73 != v24 );
    if ( v86 != v87 )
      v87 = v86;
  }
  sub_E66D20((__int64)v89);
  sub_B72320((__int64)v89, v4);
  if ( v86 )
    j_j___libc_free_0(v86);
  v36 = 16LL * v85;
  sub_C7D6A0((__int64)v83, v36, 8);
  ++v74;
  *a1 = v14;
  if ( !(_DWORD)v76 )
  {
    if ( !HIDWORD(v76) )
      goto LABEL_42;
    v37 = v77;
    if ( v77 <= 0x40 )
      goto LABEL_39;
    v36 = 16LL * v77;
    sub_C7D6A0((__int64)v75, v36, 8);
    v77 = 0;
LABEL_101:
    v75 = 0;
LABEL_41:
    v76 = 0;
    goto LABEL_42;
  }
  v54 = 4 * v76;
  v36 = 64;
  v37 = v77;
  if ( (unsigned int)(4 * v76) < 0x40 )
    v54 = 64;
  if ( v77 <= v54 )
  {
LABEL_39:
    v38 = v75;
    v39 = &v75[2 * v37];
    if ( v75 != v39 )
    {
      do
      {
        *v38 = -4096;
        v38 += 2;
      }
      while ( v39 != v38 );
    }
    goto LABEL_41;
  }
  v55 = v75;
  v56 = 2LL * v77;
  if ( (_DWORD)v76 == 1 )
  {
    v58 = 64;
  }
  else
  {
    _BitScanReverse(&v57, v76 - 1);
    v58 = 1 << (33 - (v57 ^ 0x1F));
    if ( v58 < 64 )
      v58 = 64;
    if ( v58 == v77 )
    {
      v76 = 0;
      v70 = &v75[v56];
      do
      {
        if ( v55 )
          *v55 = -4096;
        v55 += 2;
      }
      while ( v70 != v55 );
      goto LABEL_42;
    }
  }
  v36 = 16LL * v77;
  sub_C7D6A0((__int64)v75, v56 * 8, 8);
  v59 = sub_2309150(v58);
  v77 = v59;
  if ( !v59 )
    goto LABEL_101;
  v36 = 8;
  v60 = (_QWORD *)sub_C7D670(16LL * v59, 8);
  v76 = 0;
  v75 = v60;
  for ( j = &v60[2 * v77]; j != v60; v60 += 2 )
  {
    if ( v60 )
      *v60 = -4096;
  }
LABEL_42:
  v40 = v79;
  v41 = v78;
  if ( v78 != v79 )
  {
    do
    {
      v42 = *(_QWORD *)v41;
      v43 = *(__int64 **)(*(_QWORD *)v41 + 8LL);
      v44 = *(__int64 **)(*(_QWORD *)v41 + 16LL);
      if ( v43 == v44 )
      {
        *(_BYTE *)(v42 + 152) = 1;
      }
      else
      {
        do
        {
          v45 = *v43++;
          sub_D47BB0(v45, v36);
        }
        while ( v44 != v43 );
        *(_BYTE *)(v42 + 152) = 1;
        v46 = *(_QWORD *)(v42 + 8);
        if ( *(_QWORD *)(v42 + 16) != v46 )
          *(_QWORD *)(v42 + 16) = v46;
      }
      v47 = *(_QWORD *)(v42 + 32);
      if ( v47 != *(_QWORD *)(v42 + 40) )
        *(_QWORD *)(v42 + 40) = v47;
      ++*(_QWORD *)(v42 + 56);
      if ( *(_BYTE *)(v42 + 84) )
      {
        *(_QWORD *)v42 = 0;
      }
      else
      {
        v48 = 4 * (*(_DWORD *)(v42 + 76) - *(_DWORD *)(v42 + 80));
        v49 = *(unsigned int *)(v42 + 72);
        if ( v48 < 0x20 )
          v48 = 32;
        if ( (unsigned int)v49 > v48 )
        {
          sub_C8C990(v42 + 56, v36);
        }
        else
        {
          v36 = 0xFFFFFFFFLL;
          memset(*(void **)(v42 + 64), -1, 8 * v49);
        }
        v50 = *(_BYTE *)(v42 + 84);
        *(_QWORD *)v42 = 0;
        if ( !v50 )
          _libc_free(*(_QWORD *)(v42 + 64));
      }
      v51 = *(_QWORD *)(v42 + 32);
      if ( v51 )
      {
        v36 = *(_QWORD *)(v42 + 48) - v51;
        j_j___libc_free_0(v51);
      }
      v52 = *(_QWORD *)(v42 + 8);
      if ( v52 )
      {
        v36 = *(_QWORD *)(v42 + 24) - v52;
        j_j___libc_free_0(v52);
      }
      v41 += 8LL;
    }
    while ( v40 != v41 );
    if ( v78 != v79 )
      v79 = v78;
  }
  sub_E66D20((__int64)v81);
  sub_B72320((__int64)v81, v36);
  if ( v78 )
    j_j___libc_free_0(v78);
  sub_C7D6A0((__int64)v75, 16LL * v77, 8);
  return a1;
}
