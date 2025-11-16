// Function: sub_29751F0
// Address: 0x29751f0
//
__int64 __fastcall sub_29751F0(__int64 a1, __int64 **a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // rdx
  __int64 v7; // r9
  unsigned __int64 v8; // r13
  __int64 v9; // rcx
  __int64 v10; // r8
  _BYTE *v11; // r15
  __int64 v12; // rsi
  unsigned __int64 v13; // rdi
  __int64 *v14; // rax
  __int64 *v15; // rbx
  __int64 v16; // rcx
  int v17; // r13d
  int v18; // eax
  __int64 v19; // rdx
  signed __int64 v20; // r13
  _QWORD *v21; // rax
  _BYTE *v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 *v25; // r12
  int v26; // r15d
  _QWORD *v27; // r13
  __int64 *v28; // rax
  __int64 v29; // rbx
  __int64 v30; // r9
  __int64 v31; // r12
  __int64 v32; // r10
  __int64 v33; // rdx
  int v34; // r14d
  __int64 v35; // rbx
  __int64 v36; // r12
  int v37; // eax
  int v38; // edi
  __int64 v39; // r11
  __int64 v40; // r15
  __int64 v41; // rsi
  __int64 v42; // rcx
  _QWORD *v43; // rax
  _QWORD *v44; // rcx
  __int64 *v45; // rax
  _QWORD *v46; // rbx
  _QWORD *v47; // r12
  __int64 v48; // rax
  unsigned __int8 v50; // [rsp+17h] [rbp-4A9h]
  __int64 v55; // [rsp+38h] [rbp-488h]
  __int64 v56; // [rsp+38h] [rbp-488h]
  __int64 v57; // [rsp+38h] [rbp-488h]
  __int64 v58; // [rsp+40h] [rbp-480h]
  __int64 v59; // [rsp+50h] [rbp-470h] BYREF
  __int64 *v60; // [rsp+58h] [rbp-468h]
  __int64 v61; // [rsp+60h] [rbp-460h]
  int v62; // [rsp+68h] [rbp-458h]
  unsigned __int8 v63; // [rsp+6Ch] [rbp-454h]
  char v64; // [rsp+70h] [rbp-450h] BYREF
  _QWORD *v65; // [rsp+F0h] [rbp-3D0h] BYREF
  __int64 v66; // [rsp+F8h] [rbp-3C8h]
  _BYTE v67[384]; // [rsp+100h] [rbp-3C0h] BYREF
  _BYTE *v68; // [rsp+280h] [rbp-240h] BYREF
  __int64 v69; // [rsp+288h] [rbp-238h]
  _BYTE v70[560]; // [rsp+290h] [rbp-230h] BYREF

  v68 = v70;
  v69 = 0x2000000000LL;
  sub_D0E1D0(a1, (__int64)&v68);
  v8 = (unsigned __int64)v68;
  v59 = 0;
  v60 = (__int64 *)&v64;
  v9 = (__int64)&v68[16 * (unsigned int)v69];
  v63 = 1;
  v61 = 16;
  v62 = 0;
  if ( v68 == (_BYTE *)v9 )
  {
LABEL_14:
    HIDWORD(v66) = 16;
    v65 = v67;
LABEL_15:
    v17 = 0;
    v18 = 0;
    goto LABEL_38;
  }
  v10 = 1;
  v11 = &v68[16 * (unsigned int)v69];
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v12 = *(_QWORD *)(v8 + 8);
        if ( (_BYTE)v10 )
          break;
LABEL_16:
        v8 += 16LL;
        sub_C8CC70((__int64)&v59, v12, (__int64)v6, v9, v10, v7);
        v10 = v63;
        v13 = (unsigned __int64)v60;
        if ( (_BYTE *)v8 == v11 )
          goto LABEL_9;
      }
      v13 = (unsigned __int64)v60;
      v6 = &v60[HIDWORD(v61)];
      v7 = HIDWORD(v61);
      if ( v60 != v6 )
        break;
LABEL_18:
      if ( HIDWORD(v61) >= (unsigned int)v61 )
        goto LABEL_16;
      v7 = (unsigned int)(HIDWORD(v61) + 1);
      v8 += 16LL;
      ++HIDWORD(v61);
      *v6 = v12;
      v13 = (unsigned __int64)v60;
      ++v59;
      v10 = v63;
      if ( (_BYTE *)v8 == v11 )
        goto LABEL_9;
    }
    v14 = v60;
    while ( v12 != *v14 )
    {
      if ( v6 == ++v14 )
        goto LABEL_18;
    }
    v8 += 16LL;
  }
  while ( (_BYTE *)v8 != v11 );
LABEL_9:
  v15 = (__int64 *)v13;
  if ( (_BYTE)v10 )
    v16 = v13 + 8LL * HIDWORD(v61);
  else
    v16 = v13 + 8LL * (unsigned int)v61;
  if ( v16 == v13 )
    goto LABEL_14;
  while ( (unsigned __int64)*v15 >= 0xFFFFFFFFFFFFFFFELL )
  {
    if ( ++v15 == (__int64 *)v16 )
      goto LABEL_14;
  }
  v65 = v67;
  v66 = 0x1000000000LL;
  if ( v15 == (__int64 *)v16 )
    goto LABEL_15;
  v19 = (__int64)v15;
  v20 = 0;
  while ( 1 )
  {
    v21 = (_QWORD *)(v19 + 8);
    if ( v19 + 8 == v16 )
      break;
    while ( 1 )
    {
      v19 = (__int64)v21;
      if ( *v21 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( (_QWORD *)v16 == ++v21 )
        goto LABEL_26;
    }
    ++v20;
    if ( v21 == (_QWORD *)v16 )
      goto LABEL_27;
  }
LABEL_26:
  ++v20;
LABEL_27:
  v22 = v67;
  if ( v20 > 16 )
  {
    v58 = v16;
    sub_D6B130((__int64)&v65, v20, v19, v16, v10, v7);
    v16 = v58;
    v22 = &v65[3 * (unsigned int)v66];
  }
  v23 = *v15;
  v24 = a3;
  v25 = (__int64 *)v16;
  v26 = v20;
  v27 = v22;
  do
  {
    if ( v27 )
    {
      *v27 = 4;
      v27[1] = 0;
      v27[2] = v23;
      if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
      {
        v55 = v24;
        sub_BD73F0((__int64)v27);
        v24 = v55;
      }
    }
    v28 = v15 + 1;
    if ( v15 + 1 == v25 )
      break;
    while ( 1 )
    {
      v23 = *v28;
      v15 = v28;
      if ( (unsigned __int64)*v28 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v25 == ++v28 )
        goto LABEL_37;
    }
    v27 += 3;
  }
  while ( v28 != v25 );
LABEL_37:
  v17 = v26;
  a3 = v24;
  v18 = v66;
LABEL_38:
  v29 = a3;
  v50 = 0;
  LODWORD(v66) = v17 + v18;
  v30 = a1 + 72;
  v31 = a1 + 72;
  v32 = *(_QWORD *)(a1 + 80);
  if ( v32 != a1 + 72 )
  {
    do
    {
      v33 = v29;
      v34 = 0;
      v35 = v31;
      v36 = v32;
      do
      {
        v39 = v36 - 24;
        v36 = *(_QWORD *)(v36 + 8);
        if ( v33 && v36 != v35 )
        {
          v40 = v39;
          while ( 1 )
          {
            v41 = v36 - 24;
            if ( !v36 )
              v41 = 0;
            if ( !*(_BYTE *)(v33 + 560) )
              break;
            v42 = *(unsigned int *)(v33 + 588);
            if ( (_DWORD)v42 == *(_DWORD *)(v33 + 592) )
              break;
            if ( *(_BYTE *)(v33 + 596) )
            {
              v43 = *(_QWORD **)(v33 + 576);
              v44 = &v43[v42];
              if ( v43 == v44 )
                break;
              while ( v41 != *v43 )
              {
                if ( v44 == ++v43 )
                  goto LABEL_56;
              }
              v36 = *(_QWORD *)(v36 + 8);
              if ( v36 == v35 )
                break;
            }
            else
            {
              v57 = v33;
              v45 = sub_C8CA60(v33 + 568, v41);
              v33 = v57;
              if ( !v45 )
                break;
              v36 = *(_QWORD *)(v36 + 8);
              if ( v36 == v35 )
                break;
            }
          }
LABEL_56:
          v39 = v40;
        }
        v56 = v33;
        v37 = sub_FC3C00(v39, a2, v33, a4, a5, v30, v65, (unsigned int)v66);
        v38 = v34;
        v33 = v56;
        if ( (_BYTE)v37 )
          v38 = v37;
        v34 = v38;
      }
      while ( v36 != v35 );
      v31 = v35;
      v29 = v56;
      if ( !(_BYTE)v38 )
        break;
      v50 = v38;
      v32 = *(_QWORD *)(a1 + 80);
    }
    while ( v32 != v31 );
  }
  v46 = v65;
  v47 = &v65[3 * (unsigned int)v66];
  if ( v65 != v47 )
  {
    do
    {
      v48 = *(v47 - 1);
      v47 -= 3;
      if ( v48 != 0 && v48 != -4096 && v48 != -8192 )
        sub_BD60C0(v47);
    }
    while ( v46 != v47 );
    v47 = v65;
  }
  if ( v47 != (_QWORD *)v67 )
    _libc_free((unsigned __int64)v47);
  if ( !v63 )
    _libc_free((unsigned __int64)v60);
  if ( v68 != v70 )
    _libc_free((unsigned __int64)v68);
  return v50;
}
