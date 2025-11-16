// Function: sub_2186590
// Address: 0x2186590
//
__int64 __fastcall sub_2186590(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  int v8; // eax
  __int64 v9; // rdx
  _QWORD *v10; // rax
  _QWORD *i; // rdx
  __int64 v12; // rbx
  __int64 v13; // rcx
  __int64 v14; // r9
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // rdi
  bool v18; // zf
  __int64 v19; // rsi
  int *v20; // r8
  unsigned int v21; // eax
  unsigned int v22; // esi
  unsigned int v23; // r8d
  int v24; // edi
  int v25; // eax
  __int64 v26; // rdx
  _QWORD *v27; // rax
  _QWORD *v28; // rdx
  int v29; // r9d
  __int64 v30; // rax
  unsigned int v31; // r8d
  unsigned int v32; // ecx
  _QWORD *v33; // rdi
  unsigned int v34; // eax
  int v35; // r9d
  unsigned int v36; // eax
  _QWORD *v37; // rax
  _QWORD *j; // rdx
  int v39; // r8d
  int v40; // r9d
  __int64 v41; // rax
  _QWORD *v42; // rdi
  __int64 v43; // r15
  __int64 v44; // r14
  __int64 v45; // r13
  int v46; // edx
  __int64 *v47; // rcx
  __int64 v48; // r10
  int v50; // r8d
  int v51; // r9d
  __int64 v52; // rax
  int v53; // ecx
  int v54; // r8d
  __int64 *v55; // r11
  int v56; // edi
  unsigned int v57; // ecx
  unsigned int v58; // eax
  int v59; // r15d
  unsigned int v60; // eax
  _QWORD *v61; // rax
  unsigned int v62; // [rsp+14h] [rbp-11Ch]
  int v63; // [rsp+14h] [rbp-11Ch]
  int v64; // [rsp+14h] [rbp-11Ch]
  __int64 v65; // [rsp+28h] [rbp-108h]
  const void *v67; // [rsp+30h] [rbp-100h]
  unsigned int v69; // [rsp+3Ch] [rbp-F4h]
  int v70; // [rsp+3Ch] [rbp-F4h]
  unsigned int v71; // [rsp+3Ch] [rbp-F4h]
  unsigned int v72; // [rsp+3Ch] [rbp-F4h]
  __int64 v73; // [rsp+48h] [rbp-E8h] BYREF
  __int64 v74; // [rsp+50h] [rbp-E0h] BYREF
  _QWORD *v75; // [rsp+58h] [rbp-D8h]
  __int64 v76; // [rsp+60h] [rbp-D0h]
  __int64 v77; // [rsp+68h] [rbp-C8h]
  _QWORD v78[6]; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v79; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v80; // [rsp+A8h] [rbp-88h]
  __int64 v81; // [rsp+B0h] [rbp-80h]
  __int64 v82; // [rsp+B8h] [rbp-78h]
  __int64 v83; // [rsp+C0h] [rbp-70h]
  __int64 v84; // [rsp+C8h] [rbp-68h]
  __int64 v85; // [rsp+D0h] [rbp-60h]
  __int64 v86; // [rsp+D8h] [rbp-58h]
  __int64 v87; // [rsp+E0h] [rbp-50h]

  v8 = *(_DWORD *)(a3 + 16);
  ++*(_QWORD *)a3;
  if ( v8 )
  {
    v57 = 4 * v8;
    v9 = *(unsigned int *)(a3 + 24);
    if ( (unsigned int)(4 * v8) < 0x40 )
      v57 = 64;
    if ( (unsigned int)v9 <= v57 )
      goto LABEL_4;
    v58 = v8 - 1;
    if ( v58 )
    {
      _BitScanReverse(&v58, v58);
      v59 = 1 << (33 - (v58 ^ 0x1F));
      if ( v59 < 64 )
        v59 = 64;
      if ( (_DWORD)v9 == v59 )
        goto LABEL_78;
    }
    else
    {
      v59 = 64;
    }
    j___libc_free_0(*(_QWORD *)(a3 + 8));
    v60 = sub_217D900(v59);
    *(_DWORD *)(a3 + 24) = v60;
    if ( !v60 )
      goto LABEL_88;
    *(_QWORD *)(a3 + 8) = sub_22077B0(16LL * v60);
LABEL_78:
    sub_217EE20(a3);
    goto LABEL_7;
  }
  if ( *(_DWORD *)(a3 + 20) )
  {
    v9 = *(unsigned int *)(a3 + 24);
    if ( (unsigned int)v9 <= 0x40 )
    {
LABEL_4:
      v10 = *(_QWORD **)(a3 + 8);
      for ( i = &v10[2 * v9]; i != v10; v10 += 2 )
        *v10 = -8;
      goto LABEL_6;
    }
    j___libc_free_0(*(_QWORD *)(a3 + 8));
    *(_DWORD *)(a3 + 24) = 0;
LABEL_88:
    *(_QWORD *)(a3 + 8) = 0;
LABEL_6:
    *(_QWORD *)(a3 + 16) = 0;
  }
LABEL_7:
  v74 = 0;
  v75 = 0;
  v12 = *(_QWORD *)(a2 + 328);
  v76 = 0;
  v77 = 0;
  v65 = a2 + 320;
  if ( v12 == a2 + 320 )
  {
    v69 = 0;
    v42 = 0;
    goto LABEL_53;
  }
  v69 = 0;
  do
  {
    v19 = *(_QWORD *)(a1 + 240);
    v87 = v12;
    v20 = *(int **)(a1 + 264);
    v73 = v12;
    v79 = 0;
    v80 = 0;
    v81 = 0;
    v82 = 0;
    v83 = 0;
    v84 = 0;
    v85 = 0;
    v86 = 0;
    v21 = sub_21861D0((__int64)&v79, v19, a2, 0, v20);
    v22 = *(_DWORD *)(a3 + 24);
    v23 = v21;
    if ( !v22 )
    {
      ++*(_QWORD *)a3;
LABEL_15:
      v62 = v23;
      v22 *= 2;
LABEL_16:
      sub_1DA35E0(a3, v22);
      sub_217F2A0(a3, &v73, v78);
      v16 = (__int64 *)v78[0];
      v13 = v73;
      v23 = v62;
      v24 = *(_DWORD *)(a3 + 16) + 1;
      goto LABEL_17;
    }
    v13 = v73;
    v14 = *(_QWORD *)(a3 + 8);
    v15 = (v22 - 1) & (((unsigned int)v73 >> 9) ^ ((unsigned int)v73 >> 4));
    v16 = (__int64 *)(v14 + 16LL * v15);
    v17 = *v16;
    if ( v73 == *v16 )
    {
LABEL_10:
      *((_DWORD *)v16 + 2) = v23;
      v18 = v23 == v69;
      if ( v23 > v69 )
        goto LABEL_20;
      goto LABEL_11;
    }
    v64 = 1;
    v55 = 0;
    while ( v17 != -8 )
    {
      if ( v17 == -16 && !v55 )
        v55 = v16;
      v15 = (v22 - 1) & (v64 + v15);
      v16 = (__int64 *)(v14 + 16LL * v15);
      v17 = *v16;
      if ( v73 == *v16 )
        goto LABEL_10;
      ++v64;
    }
    v56 = *(_DWORD *)(a3 + 16);
    if ( v55 )
      v16 = v55;
    ++*(_QWORD *)a3;
    v24 = v56 + 1;
    if ( 4 * v24 >= 3 * v22 )
      goto LABEL_15;
    if ( v22 - *(_DWORD *)(a3 + 20) - v24 <= v22 >> 3 )
    {
      v62 = v23;
      goto LABEL_16;
    }
LABEL_17:
    *(_DWORD *)(a3 + 16) = v24;
    if ( *v16 != -8 )
      --*(_DWORD *)(a3 + 20);
    *((_DWORD *)v16 + 2) = 0;
    *v16 = v13;
    *((_DWORD *)v16 + 2) = v23;
    v18 = v23 == v69;
    if ( v23 > v69 )
    {
LABEL_20:
      v25 = v76;
      ++v74;
      *(_DWORD *)(a4 + 8) = 0;
      if ( v25 )
      {
        v32 = 4 * v25;
        v26 = (unsigned int)v77;
        if ( (unsigned int)(4 * v25) < 0x40 )
          v32 = 64;
        if ( v32 >= (unsigned int)v77 )
        {
LABEL_23:
          v27 = v75;
          v28 = &v75[v26];
          if ( v75 != v28 )
          {
            do
              *v27++ = -8;
            while ( v28 != v27 );
          }
          goto LABEL_25;
        }
        v33 = v75;
        v34 = v25 - 1;
        if ( v34 )
        {
          _BitScanReverse(&v34, v34);
          v35 = 1 << (33 - (v34 ^ 0x1F));
          if ( v35 < 64 )
            v35 = 64;
          if ( (_DWORD)v77 == v35 )
          {
            v76 = 0;
            v61 = &v75[(unsigned int)v77];
            do
            {
              if ( v33 )
                *v33 = -8;
              ++v33;
            }
            while ( v61 != v33 );
            goto LABEL_26;
          }
        }
        else
        {
          v35 = 64;
        }
        v63 = v35;
        v71 = v23;
        j___libc_free_0(v75);
        v36 = sub_217D900(v63);
        v23 = v71;
        LODWORD(v77) = v36;
        if ( v36 )
        {
          v37 = (_QWORD *)sub_22077B0(8LL * v36);
          v76 = 0;
          v75 = v37;
          v23 = v71;
          for ( j = &v37[(unsigned int)v77]; j != v37; ++v37 )
          {
            if ( v37 )
              *v37 = -8;
          }
          goto LABEL_26;
        }
      }
      else
      {
        if ( !HIDWORD(v76) )
        {
LABEL_26:
          v70 = v23;
          sub_2182830((__int64)v78, (__int64)&v74, &v73);
          v30 = *(unsigned int *)(a4 + 8);
          v31 = v70;
          if ( (unsigned int)v30 >= *(_DWORD *)(a4 + 12) )
          {
            sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v70, v29);
            v30 = *(unsigned int *)(a4 + 8);
            v31 = v70;
          }
          v69 = v31;
          *(_QWORD *)(*(_QWORD *)a4 + 8 * v30) = v73;
          ++*(_DWORD *)(a4 + 8);
          goto LABEL_12;
        }
        v26 = (unsigned int)v77;
        if ( (unsigned int)v77 <= 0x40 )
          goto LABEL_23;
        v72 = v23;
        j___libc_free_0(v75);
        v23 = v72;
        LODWORD(v77) = 0;
      }
      v75 = 0;
LABEL_25:
      v76 = 0;
      goto LABEL_26;
    }
LABEL_11:
    if ( v18 )
    {
      sub_2182830((__int64)v78, (__int64)&v74, &v73);
      v41 = *(unsigned int *)(a4 + 8);
      if ( (unsigned int)v41 >= *(_DWORD *)(a4 + 12) )
      {
        sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v39, v40);
        v41 = *(unsigned int *)(a4 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a4 + 8 * v41) = v73;
      ++*(_DWORD *)(a4 + 8);
    }
LABEL_12:
    j___libc_free_0(v84);
    j___libc_free_0(v80);
    v12 = *(_QWORD *)(v12 + 8);
  }
  while ( v65 != v12 );
  v42 = v75;
  v43 = *(_QWORD *)(a2 + 328);
  if ( v12 != v43 )
  {
    v67 = (const void *)(a4 + 16);
    v44 = a4;
    v45 = v43;
    while ( 1 )
    {
      v79 = v45;
      if ( !(_DWORD)v77 )
        goto LABEL_51;
      v46 = (v77 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
      v47 = &v42[v46];
      v48 = *v47;
      if ( *v47 != v45 )
      {
        v53 = 1;
        while ( v48 != -8 )
        {
          v54 = v53 + 1;
          v46 = (v77 - 1) & (v53 + v46);
          v47 = &v42[v46];
          v48 = *v47;
          if ( *v47 == v45 )
            goto LABEL_48;
          v53 = v54;
        }
        goto LABEL_51;
      }
LABEL_48:
      if ( v47 == &v42[(unsigned int)v77] )
      {
LABEL_51:
        if ( *((_DWORD *)sub_2107730(a3, &v79) + 2) > a5 && *((_DWORD *)sub_2107730(a3, &v79) + 2) + 15 > v69 )
        {
          v52 = *(unsigned int *)(v44 + 8);
          if ( (unsigned int)v52 >= *(_DWORD *)(v44 + 12) )
          {
            sub_16CD150(v44, v67, 0, 8, v50, v51);
            v52 = *(unsigned int *)(v44 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v44 + 8 * v52) = v79;
          ++*(_DWORD *)(v44 + 8);
        }
        v45 = *(_QWORD *)(v45 + 8);
        v42 = v75;
        if ( v12 == v45 )
          break;
      }
      else
      {
        v45 = *(_QWORD *)(v45 + 8);
        if ( v12 == v45 )
          break;
      }
    }
  }
LABEL_53:
  j___libc_free_0(v42);
  return v69;
}
