// Function: sub_27FB730
// Address: 0x27fb730
//
__int64 __fastcall sub_27FB730(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __int64 a7,
        __int64 *a8,
        __int64 a9,
        _BYTE *a10,
        __int64 *a11)
{
  unsigned __int64 *v12; // rax
  __int64 **v13; // r14
  __int64 *v16; // rdi
  __int64 v17; // rcx
  unsigned int v18; // edx
  __int64 *v19; // rax
  __int64 v20; // r8
  __int64 v21; // rdx
  unsigned int v22; // r12d
  unsigned int v23; // eax
  unsigned __int64 v24; // rdi
  __int64 v25; // rcx
  __int64 *v26; // r9
  int v27; // r8d
  unsigned int v28; // esi
  __int64 *v29; // rdx
  __int64 v30; // r10
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  unsigned int v35; // eax
  __int64 *v36; // rdi
  int v38; // edx
  int v39; // r11d
  unsigned int v40; // ecx
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  unsigned int v43; // eax
  __int64 *v44; // rsi
  unsigned int v45; // edx
  unsigned int v46; // r8d
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  int v49; // r11d
  __int64 *v50; // rdi
  int v51; // r10d
  unsigned int v52; // edx
  __int64 v53; // rcx
  int v54; // r11d
  __int64 *v55; // rax
  __int64 *v56; // rdi
  int v57; // r10d
  unsigned int v58; // edx
  __int64 v59; // rcx
  int v60; // r11d
  __int64 v62; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v63; // [rsp+28h] [rbp-A8h]
  __int64 *v64; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v65; // [rsp+38h] [rbp-98h]
  _QWORD *v66; // [rsp+70h] [rbp-60h] BYREF
  __int64 v67; // [rsp+78h] [rbp-58h]
  _QWORD v68[10]; // [rsp+80h] [rbp-50h] BYREF

  v12 = (unsigned __int64 *)&v64;
  v13 = (__int64 **)a6;
  v62 = 0;
  v63 = 1;
  do
  {
    *v12 = -4096;
    v12 += 2;
  }
  while ( v12 != (unsigned __int64 *)&v66 );
  v66 = v68;
  v67 = 0x400000000LL;
  if ( (v63 & 1) != 0 )
  {
    v16 = (__int64 *)&v64;
    v17 = 3;
  }
  else
  {
    v40 = v65;
    v16 = v64;
    if ( !v65 )
    {
      v43 = v63;
      ++v62;
      v44 = 0;
      v45 = ((unsigned int)v63 >> 1) + 1;
LABEL_39:
      v46 = 3 * v40;
      goto LABEL_40;
    }
    v17 = v65 - 1;
  }
  v18 = v17 & (((unsigned int)a7 >> 9) ^ ((unsigned int)a7 >> 4));
  v19 = &v16[2 * v18];
  v20 = *v19;
  if ( *v19 == a7 )
  {
LABEL_6:
    v21 = v19[1];
    if ( v21 != -1 )
    {
      v68[v21] = 0;
      v19[1] = (unsigned int)v67;
      v41 = (unsigned int)v67;
      v17 = HIDWORD(v67);
      v42 = (unsigned int)v67 + 1LL;
      if ( v42 > HIDWORD(v67) )
      {
        sub_C8D5F0((__int64)&v66, v68, v42, 8u, v20, a6);
        v41 = (unsigned int)v67;
      }
      v21 = (__int64)v66;
      v66[v41] = a7;
      LODWORD(v67) = v67 + 1;
    }
    goto LABEL_7;
  }
  v49 = 1;
  v44 = 0;
  while ( v20 != -4096 )
  {
    if ( !v44 && v20 == -8192 )
      v44 = v19;
    a6 = (unsigned int)(v49 + 1);
    v18 = v17 & (v49 + v18);
    v19 = &v16[2 * v18];
    v20 = *v19;
    if ( *v19 == a7 )
      goto LABEL_6;
    ++v49;
  }
  if ( !v44 )
    v44 = v19;
  v43 = v63;
  ++v62;
  v45 = ((unsigned int)v63 >> 1) + 1;
  if ( (v63 & 1) == 0 )
  {
    v40 = v65;
    goto LABEL_39;
  }
  v46 = 12;
  v40 = 4;
LABEL_40:
  if ( 4 * v45 >= v46 )
  {
    sub_F76580((__int64)&v62, 2 * v40);
    if ( (v63 & 1) != 0 )
    {
      v56 = (__int64 *)&v64;
      v57 = 3;
    }
    else
    {
      v56 = v64;
      if ( !v65 )
        goto LABEL_86;
      v57 = v65 - 1;
    }
    v43 = v63;
    v58 = v57 & (((unsigned int)a7 >> 9) ^ ((unsigned int)a7 >> 4));
    v44 = &v56[2 * v58];
    v59 = *v44;
    if ( *v44 == a7 )
      goto LABEL_42;
    v60 = 1;
    v55 = 0;
    while ( v59 != -4096 )
    {
      if ( !v55 && v59 == -8192 )
        v55 = v44;
      a6 = (unsigned int)(v60 + 1);
      v58 = v57 & (v60 + v58);
      v44 = &v56[2 * v58];
      v59 = *v44;
      if ( *v44 == a7 )
        goto LABEL_60;
      ++v60;
    }
    goto LABEL_58;
  }
  if ( v40 - HIDWORD(v63) - v45 <= v40 >> 3 )
  {
    sub_F76580((__int64)&v62, v40);
    if ( (v63 & 1) != 0 )
    {
      v50 = (__int64 *)&v64;
      v51 = 3;
LABEL_55:
      v43 = v63;
      v52 = v51 & (((unsigned int)a7 >> 9) ^ ((unsigned int)a7 >> 4));
      v44 = &v50[2 * v52];
      v53 = *v44;
      if ( *v44 == a7 )
        goto LABEL_42;
      v54 = 1;
      v55 = 0;
      while ( v53 != -4096 )
      {
        if ( v53 == -8192 && !v55 )
          v55 = v44;
        a6 = (unsigned int)(v54 + 1);
        v52 = v51 & (v54 + v52);
        v44 = &v50[2 * v52];
        v53 = *v44;
        if ( *v44 == a7 )
          goto LABEL_60;
        ++v54;
      }
LABEL_58:
      if ( v55 )
        v44 = v55;
LABEL_60:
      v43 = v63;
      goto LABEL_42;
    }
    v50 = v64;
    if ( v65 )
    {
      v51 = v65 - 1;
      goto LABEL_55;
    }
LABEL_86:
    LODWORD(v63) = (2 * ((unsigned int)v63 >> 1) + 2) | v63 & 1;
    BUG();
  }
LABEL_42:
  LODWORD(v63) = (2 * (v43 >> 1) + 2) | v43 & 1;
  if ( *v44 != -4096 )
    --HIDWORD(v63);
  v44[1] = 0;
  *v44 = a7;
  v47 = (unsigned int)v67;
  v17 = HIDWORD(v67);
  v48 = (unsigned int)v67 + 1LL;
  if ( v48 > HIDWORD(v67) )
  {
    sub_C8D5F0((__int64)&v66, v68, v48, 8u, (__int64)&v62, a6);
    v47 = (unsigned int)v67;
  }
  v21 = (__int64)v66;
  v66[v47] = a7;
  LODWORD(v67) = v67 + 1;
LABEL_7:
  v22 = 0;
  sub_F77240(a7, (__int64)&v62, v21, v17, (__int64)&v62, a6);
  v23 = v67;
  if ( (_DWORD)v67 )
  {
    while ( 1 )
    {
      v24 = (unsigned __int64)v66;
      v25 = v66[v23 - 1];
      if ( (v63 & 1) != 0 )
        break;
      v26 = v64;
      if ( v65 )
      {
        v27 = v65 - 1;
LABEL_10:
        v28 = v27 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v29 = &v26[2 * v28];
        v30 = *v29;
        if ( *v29 == v25 )
        {
LABEL_11:
          *v29 = -8192;
          ++HIDWORD(v63);
          v24 = (unsigned __int64)v66;
          LODWORD(v63) = (2 * ((unsigned int)v63 >> 1) - 2) | v63 & 1;
          v23 = v67;
        }
        else
        {
          v38 = 1;
          while ( v30 != -4096 )
          {
            v39 = v38 + 1;
            v28 = v27 & (v38 + v28);
            v29 = &v26[2 * v28];
            v30 = *v29;
            if ( v25 == *v29 )
              goto LABEL_11;
            v38 = v39;
          }
        }
      }
      v31 = v23 - 1;
      v32 = v24 + 8 * v31 - 8;
      while ( 1 )
      {
        LODWORD(v67) = v31;
        if ( !(_DWORD)v31 )
          break;
        v32 -= 8;
        if ( *(_QWORD *)(v32 + 8) )
          break;
        LODWORD(v31) = v31 - 1;
      }
      v33 = **(_QWORD **)(v25 + 32);
      if ( v33 )
      {
        v34 = (unsigned int)(*(_DWORD *)(v33 + 44) + 1);
        v35 = *(_DWORD *)(v33 + 44) + 1;
      }
      else
      {
        v34 = 0;
        v35 = 0;
      }
      v36 = 0;
      if ( v35 < *(_DWORD *)(a4 + 32) )
        v36 = *(__int64 **)(*(_QWORD *)(a4 + 24) + 8 * v34);
      v22 |= sub_27FAF10(v36, a2, a3, a4, a5, v13, v25, a8, a9, a10, a11, a7);
      v23 = v67;
      if ( !(_DWORD)v67 )
        goto LABEL_21;
    }
    v26 = (__int64 *)&v64;
    v27 = 3;
    goto LABEL_10;
  }
LABEL_21:
  if ( v66 != v68 )
    _libc_free((unsigned __int64)v66);
  if ( (v63 & 1) == 0 )
    sub_C7D6A0((__int64)v64, 16LL * v65, 8);
  return v22;
}
