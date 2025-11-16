// Function: sub_217FDD0
// Address: 0x217fdd0
//
__int64 __fastcall sub_217FDD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 i; // rbx
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // rbx
  __int64 v15; // r13
  __int64 v16; // r12
  int v17; // edi
  int v18; // eax
  int v19; // eax
  unsigned int v21; // eax
  __int64 v22; // r9
  __int64 v23; // rdx
  _QWORD *v24; // r13
  __int64 v25; // rdi
  __int64 *v26; // r15
  int *v27; // r14
  __int64 v28; // r12
  int v29; // ecx
  __int64 v30; // rcx
  int v31; // r8d
  int v32; // r9d
  unsigned __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // r12
  unsigned int v38; // ebx
  __int64 v39; // rax
  __int64 v40; // rax
  _QWORD *v41; // r15
  __int64 v42; // r13
  _QWORD *v43; // rbx
  _QWORD *v44; // r14
  _QWORD *v45; // r12
  unsigned __int64 *v46; // rcx
  unsigned __int64 v47; // rdx
  __int64 v48; // [rsp-8h] [rbp-158h]
  unsigned __int8 v49; // [rsp+10h] [rbp-140h]
  __int64 v50; // [rsp+10h] [rbp-140h]
  _BYTE *v51; // [rsp+20h] [rbp-130h]
  int *v52; // [rsp+20h] [rbp-130h]
  __int64 v53; // [rsp+28h] [rbp-128h]
  __int64 *v54; // [rsp+28h] [rbp-128h]
  unsigned __int64 v56; // [rsp+38h] [rbp-118h]
  unsigned int v57; // [rsp+4Ch] [rbp-104h] BYREF
  __int64 v58; // [rsp+50h] [rbp-100h] BYREF
  __int64 v59; // [rsp+58h] [rbp-F8h]
  __int64 v60; // [rsp+60h] [rbp-F0h]
  __int64 v61; // [rsp+68h] [rbp-E8h]
  _BYTE *v62; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v63; // [rsp+78h] [rbp-D8h]
  _BYTE v64[16]; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v65; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v66; // [rsp+98h] [rbp-B8h]
  __int64 v67; // [rsp+A0h] [rbp-B0h]
  __int64 j; // [rsp+A8h] [rbp-A8h]
  char v69; // [rsp+B0h] [rbp-A0h]
  _BYTE *v70; // [rsp+C0h] [rbp-90h] BYREF
  __int64 v71; // [rsp+C8h] [rbp-88h]
  _BYTE v72[32]; // [rsp+D0h] [rbp-80h] BYREF
  _BYTE *v73; // [rsp+F0h] [rbp-60h] BYREF
  __int64 v74; // [rsp+F8h] [rbp-58h]
  _BYTE v75[80]; // [rsp+100h] [rbp-50h] BYREF

  v6 = *(_QWORD *)(a2 + 328);
  v70 = v72;
  v71 = 0x400000000LL;
  if ( v6 == a2 + 320 )
  {
    v58 = 0;
    v59 = 0;
    v60 = 0;
    v61 = 0;
    goto LABEL_67;
  }
  do
  {
    v7 = *(_QWORD *)(v6 + 32);
    for ( i = v6 + 24; i != v7; v7 = *(_QWORD *)(v7 + 8) )
    {
      while ( 1 )
      {
        if ( (*(_BYTE *)(*(_QWORD *)(v7 + 16) + 10LL) & 1) != 0 || (unsigned __int8)sub_217DBB0(v7, 0) )
        {
          v9 = (unsigned int)v71;
          if ( (unsigned int)v71 >= HIDWORD(v71) )
          {
            sub_16CD150((__int64)&v70, v72, 0, 8, a5, a6);
            v9 = (unsigned int)v71;
          }
          *(_QWORD *)&v70[8 * v9] = v7;
          LODWORD(v71) = v71 + 1;
        }
        if ( (*(_BYTE *)v7 & 4) == 0 )
          break;
        v7 = *(_QWORD *)(v7 + 8);
        if ( i == v7 )
          goto LABEL_12;
      }
      while ( (*(_BYTE *)(v7 + 46) & 8) != 0 )
        v7 = *(_QWORD *)(v7 + 8);
    }
LABEL_12:
    v6 = *(_QWORD *)(v6 + 8);
  }
  while ( a2 + 320 != v6 );
  v10 = (__int64)v70;
  v58 = 0;
  v62 = v64;
  v56 = (unsigned __int64)v70;
  v51 = &v70[8 * (unsigned int)v71];
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v63 = 0x400000000LL;
  if ( v51 == v70 )
  {
LABEL_67:
    v49 = 0;
    goto LABEL_32;
  }
  v49 = 0;
  do
  {
    v11 = *(_QWORD *)v56;
    v53 = *(_QWORD *)(*(_QWORD *)v56 + 24LL);
    v12 = *(unsigned int *)(*(_QWORD *)v56 + 40LL);
    if ( !(_DWORD)v12 )
      goto LABEL_28;
    LODWORD(v13) = 5 * v12;
    v14 = 0;
    v15 = 40 * v12;
    while ( 1 )
    {
      v16 = v14 + *(_QWORD *)(v11 + 32);
      if ( *(_BYTE *)v16 )
        goto LABEL_17;
      if ( (*(_BYTE *)(v16 + 3) & 0x10) != 0 )
        goto LABEL_17;
      v17 = *(_DWORD *)(v16 + 8);
      v57 = v17;
      if ( v17 >= 0 )
        goto LABEL_17;
      v74 = 0x400000000LL;
      v73 = v75;
      sub_217D7E0(v17, *(_QWORD *)(a1 + 248), &v73, v10, v13, a6);
      if ( (_DWORD)v74 == 1 )
      {
        a6 = *(_QWORD *)v73;
        v18 = **(unsigned __int16 **)(*(_QWORD *)v73 + 16LL);
        if ( v18 != 45 )
        {
          if ( **(_WORD **)(*(_QWORD *)v73 + 16LL) )
          {
            v10 = v53;
            if ( v53 != *(_QWORD *)(a6 + 24) && v18 == 140 && *(_BYTE *)(*(_QWORD *)(a6 + 32) + 80LL) == 1 )
              break;
          }
        }
      }
      if ( v73 == v75 )
      {
LABEL_17:
        v14 += 40;
        if ( v15 == v14 )
          goto LABEL_28;
      }
      else
      {
        _libc_free((unsigned __int64)v73);
        v14 += 40;
        if ( v15 == v14 )
          goto LABEL_28;
      }
    }
    v50 = *(_QWORD *)v73;
    v21 = sub_1E6B9A0(
            *(_QWORD *)(a1 + 248),
            *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 248) + 24LL) + 16LL * (v57 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL,
            (unsigned __int8 *)byte_3F871B3,
            0,
            v13,
            a6);
    v22 = v50;
    LODWORD(v50) = v21;
    (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 256) + 152LL))(
      *(_QWORD *)(a1 + 256),
      v53,
      v11,
      v21,
      0,
      v22,
      *(_QWORD *)(a1 + 232));
    sub_1E310D0(v16, v50);
    sub_217F7B0((__int64)&v65, (__int64)&v58, (int *)&v57);
    v10 = v48;
    if ( v69 )
      sub_1525B90((__int64)&v62, &v57);
    if ( v73 != v75 )
      _libc_free((unsigned __int64)v73);
    v49 = 1;
LABEL_28:
    v56 += 8LL;
  }
  while ( (_BYTE *)v56 != v51 );
  v19 = v63;
  if ( (_DWORD)v63 )
  {
    v23 = 0;
    v24 = &v73;
    v25 = 0;
    v65 = 0;
    v26 = &v65;
    v27 = (int *)&v57;
    v66 = 0;
    v67 = 0;
    for ( j = 0; ; v23 = (unsigned int)j )
    {
      v28 = v25 + 4 * v23;
      v29 = *(_DWORD *)&v62[4 * v19 - 4];
      LODWORD(v63) = v19 - 1;
      v57 = v29;
      v31 = sub_1DF91F0((__int64)v26, v27, v24);
      v33 = (unsigned __int64)v73;
      if ( !(_BYTE)v31 )
        v33 = v66 + 4LL * (unsigned int)j;
      if ( v33 == v28 )
      {
        v34 = *(_QWORD *)(a1 + 248);
        if ( (v57 & 0x80000000) != 0 )
        {
          v35 = *(_QWORD *)(*(_QWORD *)(v34 + 24) + 16LL * (v57 & 0x7FFFFFFF) + 8);
        }
        else
        {
          v34 = *(_QWORD *)(v34 + 272);
          v35 = *(_QWORD *)(v34 + 8LL * v57);
        }
        if ( v35 )
        {
          if ( (*(_BYTE *)(v35 + 3) & 0x10) == 0 )
            goto LABEL_45;
          while ( 1 )
          {
            v35 = *(_QWORD *)(v35 + 32);
            if ( !v35 )
              break;
            if ( (*(_BYTE *)(v35 + 3) & 0x10) == 0 )
              goto LABEL_45;
          }
        }
        v36 = sub_217E810(a1, v57, v34, v30, v31, v32);
        v37 = v36;
        if ( v36 )
        {
          v38 = 0;
          if ( *(_DWORD *)(v36 + 40) )
          {
            do
            {
              v39 = *(_QWORD *)(v37 + 32) + 40LL * v38;
              if ( !*(_BYTE *)v39 && (*(_BYTE *)(v39 + 3) & 0x10) == 0 )
              {
                LODWORD(v73) = *(_DWORD *)(v39 + 8);
                sub_1525B90((__int64)&v62, v24);
              }
              ++v38;
            }
            while ( v38 != *(_DWORD *)(v37 + 40) );
          }
          v40 = v37;
          if ( (*(_BYTE *)v37 & 4) == 0 && (*(_BYTE *)(v37 + 46) & 8) != 0 )
          {
            do
              v40 = *(_QWORD *)(v40 + 8);
            while ( (*(_BYTE *)(v40 + 46) & 8) != 0 );
          }
          if ( v37 != *(_QWORD *)(v40 + 8) )
          {
            v54 = v26;
            v41 = v24;
            v42 = *(_QWORD *)(v37 + 24) + 16LL;
            v43 = (_QWORD *)v37;
            v52 = v27;
            v44 = *(_QWORD **)(v40 + 8);
            do
            {
              v45 = v43;
              v43 = (_QWORD *)v43[1];
              sub_1DD5BC0(v42, (__int64)v45);
              v46 = (unsigned __int64 *)v45[1];
              v47 = *v45 & 0xFFFFFFFFFFFFFFF8LL;
              *v46 = v47 | *v46 & 7;
              *(_QWORD *)(v47 + 8) = v46;
              *v45 &= 7uLL;
              v45[1] = 0;
              sub_1DD5C20(v42);
            }
            while ( v44 != v43 );
            v24 = v41;
            v27 = v52;
            v26 = v54;
          }
          sub_217F7B0((__int64)v24, (__int64)v26, v27);
        }
      }
LABEL_45:
      v19 = v63;
      v25 = v66;
      if ( !(_DWORD)v63 )
      {
        j___libc_free_0(v66);
        break;
      }
    }
  }
  if ( v62 != v64 )
    _libc_free((unsigned __int64)v62);
LABEL_32:
  j___libc_free_0(v59);
  if ( v70 != v72 )
    _libc_free((unsigned __int64)v70);
  return v49;
}
