// Function: sub_2D29D10
// Address: 0x2d29d10
//
__int64 __fastcall sub_2D29D10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int *v6; // rax
  unsigned int *v7; // r14
  unsigned int *v8; // rbx
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r9
  __int64 v13; // rcx
  int v14; // edi
  __int64 v15; // rdx
  unsigned __int64 *v16; // rcx
  unsigned __int64 v17; // r8
  __int64 v18; // r9
  unsigned int v19; // r12d
  unsigned int v20; // esi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  unsigned __int64 *v26; // rcx
  unsigned __int64 v27; // r8
  __int64 v28; // r9
  unsigned int v29; // r15d
  __int64 v30; // r15
  __int64 v31; // rdx
  unsigned __int64 v32; // rax
  bool v33; // al
  unsigned int *v34; // rcx
  _BYTE *v35; // r15
  __int64 v36; // r9
  __int64 v37; // rsi
  __int64 v38; // rdx
  __int64 v39; // rsi
  int v40; // edx
  __int64 v41; // rax
  int v42; // edx
  unsigned int v43; // esi
  __int64 v44; // rsi
  unsigned int v45; // esi
  __int64 j; // rax
  __int64 v47; // rcx
  unsigned __int64 v48; // rdx
  __int64 v49; // rax
  __int64 i; // rax
  __int64 v51; // rcx
  unsigned __int64 v52; // rdx
  __int64 v53; // rax
  int v54; // r10d
  __int64 v55; // [rsp+0h] [rbp-1F0h]
  unsigned __int64 v57; // [rsp+18h] [rbp-1D8h]
  __int64 v58; // [rsp+20h] [rbp-1D0h]
  unsigned __int64 v59; // [rsp+28h] [rbp-1C8h]
  int *v60; // [rsp+38h] [rbp-1B8h]
  bool v61; // [rsp+38h] [rbp-1B8h]
  bool v62; // [rsp+38h] [rbp-1B8h]
  bool v63; // [rsp+38h] [rbp-1B8h]
  bool v64; // [rsp+38h] [rbp-1B8h]
  unsigned int *v65; // [rsp+40h] [rbp-1B0h] BYREF
  _BYTE *v66; // [rsp+48h] [rbp-1A8h] BYREF
  __int64 v67; // [rsp+50h] [rbp-1A0h]
  _BYTE v68[72]; // [rsp+58h] [rbp-198h] BYREF
  unsigned int *v69; // [rsp+A0h] [rbp-150h] BYREF
  _BYTE *v70; // [rsp+A8h] [rbp-148h]
  __int64 v71; // [rsp+B0h] [rbp-140h]
  _BYTE v72[72]; // [rsp+B8h] [rbp-138h] BYREF
  _DWORD *v73; // [rsp+100h] [rbp-F0h] BYREF
  _BYTE *v74; // [rsp+108h] [rbp-E8h] BYREF
  __int64 v75; // [rsp+110h] [rbp-E0h]
  _BYTE v76[72]; // [rsp+118h] [rbp-D8h] BYREF
  _DWORD *v77; // [rsp+160h] [rbp-90h] BYREF
  _BYTE *v78; // [rsp+168h] [rbp-88h]
  __int64 v79; // [rsp+170h] [rbp-80h]
  _BYTE v80[120]; // [rsp+178h] [rbp-78h] BYREF

  if ( !*(_DWORD *)(a1 + 16) )
    return 1;
  v6 = *(unsigned int **)(a1 + 8);
  v7 = &v6[54 * *(unsigned int *)(a1 + 24)];
  if ( v6 == v7 )
    return 1;
  while ( *v6 > 0xFFFFFFFD )
  {
    v6 += 54;
    if ( v7 == v6 )
      return 1;
  }
  if ( v7 == v6 )
    return 1;
  v8 = v6;
LABEL_9:
  v9 = *(_QWORD *)(a2 + 8);
  v10 = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)v10 )
  {
    v11 = *v8;
    v12 = (unsigned int)(v10 - 1);
    v13 = (unsigned int)v12 & (37 * (_DWORD)v11);
    v60 = (int *)(v9 + 216 * v13);
    v14 = *v60;
    if ( *v60 != (_DWORD)v11 )
    {
      v54 = 1;
      while ( v14 != -1 )
      {
        v13 = (unsigned int)v12 & (v54 + (_DWORD)v13);
        a5 = v9 + 216LL * (unsigned int)v13;
        v14 = *(_DWORD *)a5;
        if ( (_DWORD)v11 == *(_DWORD *)a5 )
        {
          v60 = (int *)(v9 + 216LL * (unsigned int)v13);
          goto LABEL_11;
        }
        ++v54;
      }
      return 0;
    }
LABEL_11:
    if ( v60 == (int *)(v9 + 216 * v10) )
      return 0;
    v65 = v8 + 2;
    v66 = v68;
    v67 = 0x400000000LL;
    sub_2D29C80((__int64)&v65, 0, v11, v13, a5, v12);
    v19 = v65[48];
    if ( v19 )
    {
      v15 = (unsigned int)v67;
      for ( i = (unsigned int)(v67 - 1); v19 > (unsigned int)i; LODWORD(v67) = v67 + 1 )
      {
        v51 = (__int64)v66;
        v52 = v15 + 1;
        v17 = *(_QWORD *)(*(_QWORD *)&v66[16 * i] + 8LL * *(unsigned int *)&v66[16 * i + 12]) & 0xFFFFFFFFFFFFFFC0LL;
        v53 = (*(_QWORD *)(*(_QWORD *)&v66[16 * i] + 8LL * *(unsigned int *)&v66[16 * i + 12]) & 0x3FLL) + 1;
        if ( v52 > HIDWORD(v67) )
        {
          v58 = v53;
          v59 = v17;
          sub_C8D5F0((__int64)&v66, v68, v52, 0x10u, v17, v18);
          v51 = (__int64)v66;
          v53 = v58;
          v17 = v59;
        }
        v16 = (unsigned __int64 *)(16LL * (unsigned int)v67 + v51);
        *v16 = v17;
        v16[1] = v53;
        i = (unsigned int)v67;
        v15 = (unsigned int)(v67 + 1);
      }
    }
    v20 = v8[51];
    v69 = v8 + 2;
    v70 = v72;
    v71 = 0x400000000LL;
    sub_2D29C80((__int64)&v69, v20, v15, (__int64)v16, v17, v18);
    v74 = v76;
    v75 = 0x400000000LL;
    v73 = v60 + 2;
    sub_2D29C80((__int64)&v73, 0, v21, v22, v23, v24);
    v29 = v73[48];
    if ( v29 )
    {
      v25 = (unsigned int)v75;
      for ( j = (unsigned int)(v75 - 1); v29 > (unsigned int)j; LODWORD(v75) = v75 + 1 )
      {
        v47 = (__int64)v74;
        v48 = v25 + 1;
        v27 = *(_QWORD *)(*(_QWORD *)&v74[16 * j] + 8LL * *(unsigned int *)&v74[16 * j + 12]) & 0xFFFFFFFFFFFFFFC0LL;
        v49 = (*(_QWORD *)(*(_QWORD *)&v74[16 * j] + 8LL * *(unsigned int *)&v74[16 * j + 12]) & 0x3FLL) + 1;
        if ( v48 > HIDWORD(v75) )
        {
          v55 = v49;
          v57 = v27;
          sub_C8D5F0((__int64)&v74, v76, v48, 0x10u, v27, v28);
          v47 = (__int64)v74;
          v49 = v55;
          v27 = v57;
        }
        v26 = (unsigned __int64 *)(16LL * (unsigned int)v75 + v47);
        *v26 = v27;
        v26[1] = v49;
        j = (unsigned int)v75;
        v25 = (unsigned int)(v75 + 1);
      }
    }
    v77 = v60 + 2;
    v78 = v80;
    v79 = 0x400000000LL;
    sub_2D29C80((__int64)&v77, v60[51], v25, (__int64)v26, v27, v28);
    while ( 1 )
    {
      v30 = (unsigned int)v67;
      if ( (_DWORD)v67 && *((_DWORD *)v66 + 3) < *((_DWORD *)v66 + 2) )
      {
        v31 = (__int64)&v66[16 * (unsigned int)v67 - 16];
        v32 = (unsigned __int64)&v70[16 * (unsigned int)v71 - 16];
        if ( *(_DWORD *)(v31 + 12) == *(_DWORD *)(v32 + 12) && *(_QWORD *)v31 == *(_QWORD *)v32 )
          goto LABEL_29;
        v33 = sub_2D28840((__int64)&v73, (__int64)&v77);
        if ( v33 )
          goto LABEL_47;
      }
      else
      {
        if ( !(_DWORD)v71 || *((_DWORD *)v70 + 3) >= *((_DWORD *)v70 + 2) )
        {
LABEL_29:
          v33 = sub_2D28840((__int64)&v73, (__int64)&v77);
          goto LABEL_30;
        }
        v33 = sub_2D28840((__int64)&v73, (__int64)&v77);
        if ( v33 )
        {
LABEL_47:
          v33 = 0;
LABEL_30:
          if ( v78 != v80 )
          {
            v61 = v33;
            _libc_free((unsigned __int64)v78);
            v33 = v61;
          }
          if ( v74 != v76 )
          {
            v62 = v33;
            _libc_free((unsigned __int64)v74);
            v33 = v62;
          }
          if ( v70 != v72 )
          {
            v63 = v33;
            _libc_free((unsigned __int64)v70);
            v33 = v63;
          }
          if ( v66 != v68 )
          {
            v64 = v33;
            _libc_free((unsigned __int64)v66);
            v33 = v64;
          }
          if ( !v33 )
            return 0;
          v8 += 54;
          if ( v8 == v7 )
            return 1;
          while ( *v8 > 0xFFFFFFFD )
          {
            v8 += 54;
            if ( v7 == v8 )
              return 1;
          }
          if ( v7 == v8 )
            return 1;
          goto LABEL_9;
        }
      }
      v34 = v65;
      v35 = &v66[16 * v30 - 16];
      if ( v65[48] )
      {
        a5 = *(_QWORD *)v35;
        v36 = *((unsigned int *)v35 + 3);
        v37 = (__int64)&v74[16 * (unsigned int)v75 - 16];
        v38 = *(_QWORD *)v37;
        v39 = *(unsigned int *)(v37 + 12);
        if ( *(_DWORD *)(*(_QWORD *)v35 + 8 * v36) != *(_DWORD *)(v38 + 8 * v39)
          || *(_DWORD *)(v38 + 8 * v39 + 4) != *(_DWORD *)(a5 + 8 * v36 + 4) )
        {
          goto LABEL_30;
        }
      }
      else
      {
        a5 = *(_QWORD *)v35;
        v36 = *((unsigned int *)v35 + 3);
        v44 = (__int64)&v74[16 * (unsigned int)v75 - 16];
        v38 = *(_QWORD *)v44;
        v39 = *(unsigned int *)(v44 + 12);
        if ( *(_DWORD *)(*(_QWORD *)v35 + 8 * v36) != *(_DWORD *)(v38 + 8 * v39)
          || *(_DWORD *)(v38 + 8 * v39 + 4) != *(_DWORD *)(a5 + 8 * v36 + 4) )
        {
          goto LABEL_30;
        }
      }
      a5 = *(unsigned int *)(a5 + 4 * v36 + 128);
      if ( (_DWORD)a5 != *(_DWORD *)(v38 + 4 * v39 + 128) )
        goto LABEL_30;
      v40 = *((_DWORD *)v35 + 3) + 1;
      *((_DWORD *)v35 + 3) = v40;
      if ( v40 == *(_DWORD *)&v66[16 * (unsigned int)v67 - 8] )
      {
        v45 = v34[48];
        if ( v45 )
          sub_F03D40((__int64 *)&v66, v45);
      }
      v41 = (__int64)&v74[16 * (unsigned int)v75 - 16];
      v42 = *(_DWORD *)(v41 + 12) + 1;
      *(_DWORD *)(v41 + 12) = v42;
      if ( v42 == *(_DWORD *)&v74[16 * (unsigned int)v75 - 8] )
      {
        v43 = v73[48];
        if ( v43 )
          sub_F03D40((__int64 *)&v74, v43);
      }
    }
  }
  return 0;
}
