// Function: sub_2E5C7D0
// Address: 0x2e5c7d0
//
__int64 __fastcall sub_2E5C7D0(__int64 a1, __int64 a2)
{
  __int64 (*v4)(void); // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 (*v8)(void); // rdx
  int v9; // eax
  __int64 v10; // r12
  __int64 (*v11)(void); // rax
  _QWORD *v12; // r15
  _QWORD *v13; // r14
  unsigned __int64 v14; // rsi
  _QWORD *v15; // rax
  _QWORD *v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rcx
  _QWORD *v20; // r8
  __int64 v21; // rax
  _QWORD *v22; // rdi
  __int64 v23; // rdx
  char v24; // al
  _QWORD *v25; // r15
  _QWORD *v26; // r14
  unsigned __int64 v27; // rsi
  _QWORD *v28; // rax
  _QWORD *v29; // rdi
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rax
  _QWORD *v36; // rdi
  __int64 (*v37)(); // rdx
  char v38; // al
  bool v39; // zf
  unsigned int v40; // ebx
  int v41; // eax
  int v42; // r12d
  int v43; // eax
  __int64 v44; // rdx
  _QWORD *v45; // rax
  _QWORD *i; // rdx
  int v47; // edx
  _QWORD *v48; // rdi
  __int64 v49; // rdx
  _QWORD *v50; // rax
  __int64 (*v52)(); // rdx
  char v53; // al
  unsigned int v54; // ecx
  __int64 v55; // r8
  __int64 v56; // rsi
  unsigned int v57; // edx
  unsigned int v58; // eax
  int v59; // eax
  unsigned __int64 v60; // rax
  __int64 v61; // rax
  int v62; // ecx
  __int64 v63; // r15
  unsigned int v64; // ecx
  unsigned int v65; // eax
  int v66; // eax
  unsigned __int64 v67; // rax
  __int64 v68; // rax
  int v69; // ecx
  __int64 v70; // r15
  int v71; // [rsp+Ch] [rbp-34h]
  int v72; // [rsp+Ch] [rbp-34h]

  v4 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  v5 = 0;
  if ( v4 != sub_2DAC790 )
    v5 = v4();
  *(_QWORD *)a1 = v5;
  v6 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v7 = *(_QWORD *)a1;
  *(_QWORD *)(a1 + 8) = v6;
  *(_QWORD *)(a1 + 24) = *(_QWORD *)(a2 + 32);
  v8 = *(__int64 (**)(void))(*(_QWORD *)v7 + 1272LL);
  v9 = 5;
  if ( v8 != sub_2E4EE70 )
    v9 = v8();
  *(_DWORD *)(a1 + 72) = v9;
  v10 = 0;
  v11 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 144LL);
  if ( v11 != sub_2C8F680 )
    v10 = v11();
  v12 = sub_C52410();
  v13 = v12 + 1;
  v14 = sub_C959E0();
  v15 = (_QWORD *)v12[2];
  if ( v15 )
  {
    v16 = v12 + 1;
    do
    {
      while ( 1 )
      {
        v17 = v15[2];
        v18 = v15[3];
        if ( v14 <= v15[4] )
          break;
        v15 = (_QWORD *)v15[3];
        if ( !v18 )
          goto LABEL_12;
      }
      v16 = v15;
      v15 = (_QWORD *)v15[2];
    }
    while ( v17 );
LABEL_12:
    if ( v13 != v16 && v14 >= v16[4] )
      v13 = v16;
  }
  if ( v13 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_57;
  v21 = v13[7];
  v20 = v13 + 6;
  if ( !v21 )
    goto LABEL_57;
  v14 = (unsigned int)dword_501FC88;
  v22 = v13 + 6;
  do
  {
    while ( 1 )
    {
      v19 = *(_QWORD *)(v21 + 16);
      v23 = *(_QWORD *)(v21 + 24);
      if ( *(_DWORD *)(v21 + 32) >= dword_501FC88 )
        break;
      v21 = *(_QWORD *)(v21 + 24);
      if ( !v23 )
        goto LABEL_21;
    }
    v22 = (_QWORD *)v21;
    v21 = *(_QWORD *)(v21 + 16);
  }
  while ( v19 );
LABEL_21:
  if ( v22 == v20 || dword_501FC88 < *((_DWORD *)v22 + 8) || (v24 = qword_501FD08, !*((_DWORD *)v22 + 9)) )
  {
LABEL_57:
    v52 = *(__int64 (**)())(*(_QWORD *)v10 + 1824LL);
    v24 = 0;
    if ( v52 != sub_2E4EE80 )
      v24 = ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64 (*)(), __int64, _QWORD *))v52)(
              v10,
              v14,
              v52,
              v19,
              v20);
  }
  *(_BYTE *)(a1 + 40) = v24;
  v25 = sub_C52410();
  v26 = v25 + 1;
  v27 = sub_C959E0();
  v28 = (_QWORD *)v25[2];
  if ( v28 )
  {
    v29 = v25 + 1;
    do
    {
      while ( 1 )
      {
        v30 = v28[2];
        v31 = v28[3];
        if ( v27 <= v28[4] )
          break;
        v28 = (_QWORD *)v28[3];
        if ( !v31 )
          goto LABEL_29;
      }
      v29 = v28;
      v28 = (_QWORD *)v28[2];
    }
    while ( v30 );
LABEL_29:
    if ( v29 != v26 && v27 >= v29[4] )
      v26 = v29;
  }
  if ( v26 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v35 = v26[7];
    v33 = (__int64)(v26 + 6);
    if ( v35 )
    {
      v27 = (unsigned int)dword_501FBA8;
      v36 = v26 + 6;
      do
      {
        while ( 1 )
        {
          v32 = *(_QWORD *)(v35 + 16);
          v37 = *(__int64 (**)())(v35 + 24);
          if ( *(_DWORD *)(v35 + 32) >= dword_501FBA8 )
            break;
          v35 = *(_QWORD *)(v35 + 24);
          if ( !v37 )
            goto LABEL_38;
        }
        v36 = (_QWORD *)v35;
        v35 = *(_QWORD *)(v35 + 16);
      }
      while ( v32 );
LABEL_38:
      if ( (_QWORD *)v33 != v36 && dword_501FBA8 >= *((_DWORD *)v36 + 8) )
      {
        v32 = *((unsigned int *)v36 + 9);
        v38 = qword_501FC28;
        if ( (_DWORD)v32 )
          goto LABEL_41;
      }
    }
  }
  v37 = *(__int64 (**)())(*(_QWORD *)v10 + 1832LL);
  v38 = 0;
  if ( v37 == sub_2E4EE90 )
  {
LABEL_41:
    v39 = *(_BYTE *)(a1 + 40) == 0;
    *(_BYTE *)(a1 + 41) = v38;
    if ( !v39 )
LABEL_61:
      sub_2E56F80(*(__int64 ****)(a1 + 48), *(_QWORD *)(a1 + 56));
  }
  else
  {
    v53 = ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64 (*)(), __int64, __int64))v37)(
            v10,
            v27,
            v37,
            v32,
            v33);
    v39 = *(_BYTE *)(a1 + 40) == 0;
    *(_BYTE *)(a1 + 41) = v53;
    if ( !v39 )
      goto LABEL_61;
  }
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a2 + 8) + 544LL) - 42) > 1
    || (v40 = (unsigned __int8)qword_501FA68, (_BYTE)qword_501FA68) )
  {
    v40 = sub_2E54D10((_QWORD *)a1, *(_QWORD *)(a1 + 16), (__int64)v37, v32, v33, v34);
  }
  v41 = sub_2E5A4E0(a1, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 96LL), (__int64)v37, v32, v33, v34);
  ++*(_QWORD *)(a1 + 80);
  v42 = v41;
  v43 = *(_DWORD *)(a1 + 96);
  if ( v43 )
  {
    v64 = 4 * v43;
    v44 = *(unsigned int *)(a1 + 104);
    if ( (unsigned int)(4 * v43) < 0x40 )
      v64 = 64;
    if ( v64 >= (unsigned int)v44 )
      goto LABEL_47;
    v65 = v43 - 1;
    if ( v65 )
    {
      _BitScanReverse(&v65, v65);
      v66 = 1 << (33 - (v65 ^ 0x1F));
      if ( v66 < 64 )
        v66 = 64;
      if ( v66 == (_DWORD)v44 )
        goto LABEL_82;
      v67 = (4 * v66 / 3u + 1) | ((unsigned __int64)(4 * v66 / 3u + 1) >> 1);
      v68 = ((((v67 >> 2) | v67 | (((v67 >> 2) | v67) >> 4)) >> 8)
           | (v67 >> 2)
           | v67
           | (((v67 >> 2) | v67) >> 4)
           | (((((v67 >> 2) | v67 | (((v67 >> 2) | v67) >> 4)) >> 8) | (v67 >> 2) | v67 | (((v67 >> 2) | v67) >> 4)) >> 16))
          + 1;
      v69 = v68;
      v70 = 16 * v68;
    }
    else
    {
      v70 = 2048;
      v69 = 128;
    }
    v72 = v69;
    sub_C7D6A0(*(_QWORD *)(a1 + 88), 16LL * (unsigned int)v44, 8);
    *(_DWORD *)(a1 + 104) = v72;
    *(_QWORD *)(a1 + 88) = sub_C7D670(v70, 8);
LABEL_82:
    sub_2E51280(a1 + 80);
    goto LABEL_50;
  }
  if ( !*(_DWORD *)(a1 + 100) )
    goto LABEL_50;
  v44 = *(unsigned int *)(a1 + 104);
  if ( (unsigned int)v44 > 0x40 )
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 88), 16LL * (unsigned int)v44, 8);
    *(_QWORD *)(a1 + 88) = 0;
    *(_QWORD *)(a1 + 96) = 0;
    *(_DWORD *)(a1 + 104) = 0;
    goto LABEL_50;
  }
LABEL_47:
  v45 = *(_QWORD **)(a1 + 88);
  for ( i = &v45[2 * v44]; i != v45; v45 += 2 )
    *v45 = -4096;
  *(_QWORD *)(a1 + 96) = 0;
LABEL_50:
  v47 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  if ( v47 )
  {
    v54 = 4 * v47;
    v48 = *(_QWORD **)(a1 + 120);
    v55 = *(unsigned int *)(a1 + 136);
    v56 = 2 * v55;
    if ( (unsigned int)(4 * v47) < 0x40 )
      v54 = 64;
    v50 = &v48[v56];
    if ( v54 >= (unsigned int)v55 )
      goto LABEL_53;
    v57 = v47 - 1;
    if ( v57 )
    {
      _BitScanReverse(&v58, v57);
      v59 = 1 << (33 - (v58 ^ 0x1F));
      if ( v59 < 64 )
        v59 = 64;
      if ( (_DWORD)v55 == v59 )
        goto LABEL_72;
      v60 = (4 * v59 / 3u + 1) | ((unsigned __int64)(4 * v59 / 3u + 1) >> 1);
      v61 = ((((v60 >> 2) | v60 | (((v60 >> 2) | v60) >> 4)) >> 8)
           | (v60 >> 2)
           | v60
           | (((v60 >> 2) | v60) >> 4)
           | (((((v60 >> 2) | v60 | (((v60 >> 2) | v60) >> 4)) >> 8) | (v60 >> 2) | v60 | (((v60 >> 2) | v60) >> 4)) >> 16))
          + 1;
      v62 = v61;
      v63 = 16 * v61;
    }
    else
    {
      v63 = 2048;
      v62 = 128;
    }
    v71 = v62;
    sub_C7D6A0((__int64)v48, v56 * 8, 8);
    *(_DWORD *)(a1 + 136) = v71;
    *(_QWORD *)(a1 + 120) = sub_C7D670(v63, 8);
LABEL_72:
    sub_2E512C0(a1 + 112);
    goto LABEL_56;
  }
  if ( !*(_DWORD *)(a1 + 132) )
    goto LABEL_56;
  v48 = *(_QWORD **)(a1 + 120);
  v49 = *(unsigned int *)(a1 + 136);
  v50 = &v48[2 * v49];
  if ( (unsigned int)v49 > 0x40 )
  {
    sub_C7D6A0((__int64)v48, 16 * v49, 8);
    *(_QWORD *)(a1 + 120) = 0;
    *(_QWORD *)(a1 + 128) = 0;
    *(_DWORD *)(a1 + 136) = 0;
    goto LABEL_56;
  }
LABEL_53:
  while ( v48 != v50 )
  {
    *v48 = 0;
    v48 += 2;
  }
  *(_QWORD *)(a1 + 128) = 0;
LABEL_56:
  *(_DWORD *)(a1 + 296) = 0;
  return v42 | v40;
}
