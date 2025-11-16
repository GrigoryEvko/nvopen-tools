// Function: sub_3854CF0
// Address: 0x3854cf0
//
__int64 __fastcall sub_3854CF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 ****v7; // r8
  int v9; // eax
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 ***v14; // r12
  int v15; // eax
  int v16; // esi
  __int64 v17; // rcx
  unsigned int v18; // edx
  __int64 ****v19; // rax
  __int64 ***v20; // rdi
  unsigned int v21; // r8d
  __int64 v22; // r9
  unsigned int v23; // ecx
  __int64 v24; // rdx
  unsigned int v25; // eax
  __int64 *v26; // rsi
  __int64 v27; // rdi
  __int64 v28; // rdi
  unsigned int v29; // edi
  __int64 *v30; // rax
  __int64 v31; // rsi
  unsigned int v32; // edx
  __int64 v33; // rdi
  int v34; // edx
  _QWORD *v35; // rax
  int v37; // r9d
  int v38; // r9d
  __int64 v39; // r10
  unsigned int v40; // ecx
  int v41; // edx
  __int64 v42; // r8
  unsigned __int64 v43; // rsi
  unsigned int v44; // edx
  unsigned __int64 v45; // rdi
  __int64 v46; // rax
  int v47; // eax
  __int64 ****v48; // rax
  __int64 v49; // rax
  int v50; // esi
  int v51; // r10d
  int v52; // r10d
  __int64 *v53; // r9
  int v54; // ecx
  int v55; // r8d
  int v56; // r8d
  __int64 v57; // r9
  int v58; // esi
  __int64 v59; // rbx
  __int64 *v60; // rcx
  __int64 v61; // rdi
  __int64 *v62; // r11
  int v63; // edi
  __int64 *v64; // rsi
  __int64 ****v65; // [rsp+8h] [rbp-88h]
  __int64 ****v66; // [rsp+8h] [rbp-88h]
  __int64 v67; // [rsp+10h] [rbp-80h] BYREF
  __int64 v68; // [rsp+18h] [rbp-78h] BYREF
  __int64 v69; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v70; // [rsp+28h] [rbp-68h] BYREF
  unsigned int v71; // [rsp+30h] [rbp-60h]
  __int64 ****v72; // [rsp+40h] [rbp-50h] BYREF
  __int64 v73; // [rsp+48h] [rbp-48h]
  _QWORD v74[8]; // [rsp+50h] [rbp-40h] BYREF

  v7 = (__int64 ****)v74;
  v73 = 0x200000000LL;
  v9 = *(_DWORD *)(a2 + 20);
  v72 = (__int64 ****)v74;
  v10 = 24LL * (v9 & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v11 = *(_QWORD *)(a2 - 8);
    v12 = v11 + v10;
  }
  else
  {
    v11 = a2 - v10;
    v12 = a2;
  }
  if ( v11 != v12 )
  {
    while ( 1 )
    {
      v14 = *(__int64 ****)v11;
      if ( *(_BYTE *)(*(_QWORD *)v11 + 16LL) > 0x10u )
      {
        v15 = *(_DWORD *)(a1 + 160);
        if ( !v15 )
          goto LABEL_12;
        v16 = v15 - 1;
        v17 = *(_QWORD *)(a1 + 144);
        v18 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v19 = (__int64 ****)(v17 + 16LL * v18);
        v20 = *v19;
        if ( v14 != *v19 )
        {
          v47 = 1;
          while ( v20 != (__int64 ***)-8LL )
          {
            a6 = v47 + 1;
            v18 = v16 & (v47 + v18);
            v19 = (__int64 ****)(v17 + 16LL * v18);
            v20 = *v19;
            if ( v14 == *v19 )
              goto LABEL_11;
            v47 = a6;
          }
          goto LABEL_12;
        }
LABEL_11:
        v14 = v19[1];
        if ( !v14 )
          goto LABEL_12;
      }
      v13 = (unsigned int)v73;
      if ( (unsigned int)v73 >= HIDWORD(v73) )
      {
        v65 = v7;
        sub_16CD150((__int64)&v72, v7, 0, 8, (int)v7, a6);
        v13 = (unsigned int)v73;
        v7 = v65;
      }
      v11 += 24;
      v72[v13] = v14;
      LODWORD(v73) = v73 + 1;
      if ( v12 == v11 )
      {
        v48 = v72;
        goto LABEL_46;
      }
    }
  }
  v48 = (__int64 ****)v74;
LABEL_46:
  v66 = v7;
  v46 = sub_15A4510(*v48, *(__int64 ***)a2, 0);
  v7 = v66;
  if ( v46 )
  {
    v69 = a2;
    sub_38526A0(a1 + 136, &v69)[1] = v46;
    if ( v72 != v66 )
      _libc_free((unsigned __int64)v72);
    return 1;
  }
LABEL_12:
  if ( v72 != v7 )
    _libc_free((unsigned __int64)v72);
  v21 = *(_DWORD *)(a1 + 256);
  v22 = *(_QWORD *)(a2 - 24);
  if ( !v21 )
  {
LABEL_24:
    v69 = 0;
    v71 = 1;
    v70 = 0;
    goto LABEL_25;
  }
  v23 = v21 - 1;
  v24 = *(_QWORD *)(a1 + 240);
  v25 = (v21 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
  v26 = (__int64 *)(v24 + 32LL * v25);
  v27 = *v26;
  if ( v22 != *v26 )
  {
    v50 = 1;
    while ( v27 != -8 )
    {
      v51 = v50 + 1;
      v25 = v23 & (v50 + v25);
      v26 = (__int64 *)(v24 + 32LL * v25);
      v27 = *v26;
      if ( v22 == *v26 )
        goto LABEL_16;
      v50 = v51;
    }
    goto LABEL_24;
  }
LABEL_16:
  v28 = v26[1];
  v69 = v28;
  v71 = *((_DWORD *)v26 + 6);
  if ( v71 > 0x40 )
  {
    sub_16A4FD0((__int64)&v70, (const void **)v26 + 2);
    if ( !v69 )
    {
LABEL_22:
      v22 = *(_QWORD *)(a2 - 24);
      goto LABEL_25;
    }
    v21 = *(_DWORD *)(a1 + 256);
    if ( !v21 )
    {
      ++*(_QWORD *)(a1 + 232);
      goto LABEL_36;
    }
    v24 = *(_QWORD *)(a1 + 240);
    v23 = v21 - 1;
  }
  else
  {
    v70 = v26[2];
    if ( !v28 )
      goto LABEL_25;
  }
  v29 = v23 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v30 = (__int64 *)(v24 + 32LL * v29);
  v31 = *v30;
  if ( a2 == *v30 )
  {
    v32 = *((_DWORD *)v30 + 6);
    goto LABEL_20;
  }
  v52 = 1;
  v53 = 0;
  while ( 1 )
  {
    if ( v31 == -8 )
    {
      v54 = *(_DWORD *)(a1 + 248);
      if ( v53 )
        v30 = v53;
      ++*(_QWORD *)(a1 + 232);
      v41 = v54 + 1;
      if ( 4 * (v54 + 1) < 3 * v21 )
      {
        if ( v21 - (v41 + *(_DWORD *)(a1 + 252)) > v21 >> 3 )
        {
LABEL_38:
          *(_DWORD *)(a1 + 248) = v41;
          if ( *v30 != -8 )
            --*(_DWORD *)(a1 + 252);
          *v30 = a2;
          v33 = (__int64)(v30 + 2);
          *((_DWORD *)v30 + 6) = 1;
          v30[2] = 0;
          v30[1] = v69;
          goto LABEL_41;
        }
        sub_3854320(a1 + 232, v21);
        v55 = *(_DWORD *)(a1 + 256);
        if ( v55 )
        {
          v56 = v55 - 1;
          v57 = *(_QWORD *)(a1 + 240);
          v58 = 1;
          LODWORD(v59) = v56 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v41 = *(_DWORD *)(a1 + 248) + 1;
          v60 = 0;
          v30 = (__int64 *)(v57 + 32LL * (unsigned int)v59);
          v61 = *v30;
          if ( a2 != *v30 )
          {
            while ( v61 != -8 )
            {
              if ( !v60 && v61 == -16 )
                v60 = v30;
              v59 = v56 & (unsigned int)(v59 + v58);
              v30 = (__int64 *)(v57 + 32 * v59);
              v61 = *v30;
              if ( a2 == *v30 )
                goto LABEL_38;
              ++v58;
            }
            if ( v60 )
              v30 = v60;
          }
          goto LABEL_38;
        }
LABEL_92:
        ++*(_DWORD *)(a1 + 248);
        BUG();
      }
LABEL_36:
      sub_3854320(a1 + 232, 2 * v21);
      v37 = *(_DWORD *)(a1 + 256);
      if ( v37 )
      {
        v38 = v37 - 1;
        v39 = *(_QWORD *)(a1 + 240);
        v40 = v38 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v41 = *(_DWORD *)(a1 + 248) + 1;
        v30 = (__int64 *)(v39 + 32LL * v40);
        v42 = *v30;
        if ( a2 != *v30 )
        {
          v63 = 1;
          v64 = 0;
          while ( v42 != -8 )
          {
            if ( !v64 && v42 == -16 )
              v64 = v30;
            v40 = v38 & (v63 + v40);
            v30 = (__int64 *)(v39 + 32LL * v40);
            v42 = *v30;
            if ( a2 == *v30 )
              goto LABEL_38;
            ++v63;
          }
          if ( v64 )
            v30 = v64;
        }
        goto LABEL_38;
      }
      goto LABEL_92;
    }
    if ( v53 || v31 != -16 )
      v30 = v53;
    v29 = v23 & (v52 + v29);
    v62 = (__int64 *)(v24 + 32LL * v29);
    v31 = *v62;
    if ( a2 == *v62 )
      break;
    ++v52;
    v53 = v30;
    v30 = (__int64 *)(v24 + 32LL * v29);
  }
  v32 = *((_DWORD *)v62 + 6);
  v30 = v62;
LABEL_20:
  v33 = (__int64)(v30 + 2);
  v30[1] = v69;
  if ( v32 > 0x40 )
  {
LABEL_21:
    sub_16A51C0(v33, (__int64)&v70);
    goto LABEL_22;
  }
LABEL_41:
  if ( v71 > 0x40 )
    goto LABEL_21;
  v43 = v70;
  v30[2] = v70;
  v44 = v71;
  *((_DWORD *)v30 + 6) = v71;
  v45 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v44;
  if ( v44 > 0x40 )
  {
    v49 = (unsigned int)(((unsigned __int64)v44 + 63) >> 6) - 1;
    *(_QWORD *)(v43 + 8 * v49) &= v45;
  }
  else
  {
    v30[2] = v45 & v43;
  }
  v22 = *(_QWORD *)(a2 - 24);
LABEL_25:
  v34 = *(_DWORD *)(a1 + 184);
  v72 = 0;
  v73 = -1;
  v74[0] = 0;
  v74[1] = 0;
  if ( v34 && *(_DWORD *)(a1 + 216) && sub_384F1D0(a1, v22, &v67, &v72) )
  {
    v68 = a2;
    v35 = sub_176FB00(a1 + 168, &v68);
    v35[1] = v67;
  }
  if ( v71 > 0x40 && v70 )
    j_j___libc_free_0_0(v70);
  return 1;
}
