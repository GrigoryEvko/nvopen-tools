// Function: sub_2B53530
// Address: 0x2b53530
//
__int64 *__fastcall sub_2B53530(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v7; // rax
  __int64 v8; // r13
  __int64 v9; // rbx
  __int64 v10; // rbx
  __int64 v11; // r12
  int v12; // edx
  __int64 v13; // rsi
  int v14; // edx
  unsigned int v15; // eax
  __int64 v16; // rcx
  int v17; // edi
  unsigned __int8 *v18; // r8
  __int64 v19; // rax
  __int64 v20; // rdx
  char v21; // al
  __int64 v22; // r10
  unsigned __int64 v23; // rdx
  __int64 v24; // rax
  unsigned int v25; // ebx
  __int64 v26; // rcx
  char v27; // al
  unsigned __int64 v28; // rax
  unsigned __int64 i; // rdx
  unsigned int v30; // ecx
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdi
  _QWORD *v34; // rbx
  _BYTE *v35; // r12
  __int64 v36; // rax
  _BYTE *v37; // r13
  unsigned int v38; // r14d
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rax
  unsigned int v42; // edx
  _BYTE *v43; // r12
  _QWORD *v44; // rbx
  unsigned __int64 v45; // rdi
  __int64 v46; // rax
  unsigned int v48; // eax
  _QWORD *v49; // rbx
  __int64 v50; // rax
  _QWORD *v51; // r10
  unsigned int v52; // eax
  unsigned int v53; // esi
  __int64 v54; // rcx
  __int64 v55; // rdx
  bool v56; // zf
  _QWORD *v57; // rax
  __int64 v58; // [rsp+8h] [rbp-358h]
  __int64 v59; // [rsp+8h] [rbp-358h]
  __int64 v62; // [rsp+28h] [rbp-338h]
  _QWORD *v63; // [rsp+30h] [rbp-330h] BYREF
  _QWORD *v64; // [rsp+38h] [rbp-328h] BYREF
  __int64 v65; // [rsp+40h] [rbp-320h] BYREF
  __int64 v66; // [rsp+48h] [rbp-318h]
  __int64 v67; // [rsp+50h] [rbp-310h]
  __int64 v68; // [rsp+60h] [rbp-300h] BYREF
  __int64 v69; // [rsp+68h] [rbp-2F8h]
  _QWORD *v70; // [rsp+70h] [rbp-2F0h] BYREF
  unsigned int v71; // [rsp+78h] [rbp-2E8h]
  __int64 v72; // [rsp+80h] [rbp-2E0h]
  _BYTE v73[48]; // [rsp+330h] [rbp-30h] BYREF

  v7 = &v70;
  v68 = 0;
  v69 = 1;
  do
  {
    *(_QWORD *)v7 = -4096;
    v7 += 88;
    *((_QWORD *)v7 - 10) = -4096;
    *((_QWORD *)v7 - 9) = -4096;
  }
  while ( v7 != v73 );
  v8 = 0;
  v62 = *(unsigned int *)(a3 + 8);
  if ( *(_DWORD *)(a3 + 8) )
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(*(_QWORD *)a3 + 8LL * (unsigned int)v8);
      if ( *(_BYTE *)v9 > 0x1Cu )
      {
        if ( (unsigned __int8)sub_BD3660(*(_QWORD *)(*(_QWORD *)a3 + 8LL * (unsigned int)v8), 64) )
          goto LABEL_31;
        v10 = *(_QWORD *)(v9 + 16);
        if ( v10 )
          break;
      }
LABEL_5:
      if ( v62 == ++v8 )
        goto LABEL_31;
    }
    while ( 1 )
    {
      v11 = *(_QWORD *)(v10 + 24);
      if ( *(_BYTE *)v11 != 62
        || sub_B46500(*(unsigned __int8 **)(v10 + 24))
        || (*(_BYTE *)(v11 + 2) & 1) != 0
        || *(_QWORD *)(a2 + 3280) != sub_B43CB0(v11)
        || !sub_2B08630(*(_QWORD *)(*(_QWORD *)(v11 - 64) + 8LL)) )
      {
        goto LABEL_10;
      }
      if ( (*(_BYTE *)(a2 + 88) & 1) != 0 )
      {
        v13 = a2 + 96;
        v14 = 3;
      }
      else
      {
        v12 = *(_DWORD *)(a2 + 104);
        v13 = *(_QWORD *)(a2 + 96);
        if ( !v12 )
          goto LABEL_22;
        v14 = v12 - 1;
      }
      v15 = v14 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v16 = *(_QWORD *)(v13 + 72LL * v15);
      if ( v11 == v16 )
        goto LABEL_10;
      v17 = 1;
      while ( v16 != -4096 )
      {
        a5 = (unsigned int)(v17 + 1);
        v15 = v14 & (v17 + v15);
        v16 = *(_QWORD *)(v13 + 72LL * v15);
        if ( v11 == v16 )
          goto LABEL_10;
        ++v17;
      }
LABEL_22:
      v18 = sub_98ACB0(*(unsigned __int8 **)(v11 - 32), qword_500FC48);
      v19 = *(_QWORD *)(v11 + 40);
      v20 = *(_QWORD *)(*(_QWORD *)(v11 - 64) + 8LL);
      v65 = (__int64)v18;
      v66 = v20;
      v67 = v19;
      v21 = sub_2B3EEF0((__int64)&v68, &v65, &v63);
      a5 = (__int64)&v65;
      if ( v21 )
      {
        v22 = (__int64)(v63 + 3);
        goto LABEL_24;
      }
      v51 = v63;
      ++v68;
      v64 = v63;
      v52 = ((unsigned int)v69 >> 1) + 1;
      if ( (v69 & 1) != 0 )
      {
        v54 = 24;
        v53 = 8;
      }
      else
      {
        v53 = v71;
        v54 = 3 * v71;
      }
      v55 = 4 * v52;
      if ( (unsigned int)v55 >= (unsigned int)v54 )
      {
        v53 *= 2;
      }
      else
      {
        v54 = v53 - (v52 + HIDWORD(v69));
        v55 = v53 >> 3;
        if ( (unsigned int)v54 > (unsigned int)v55 )
          goto LABEL_98;
      }
      sub_2B52FA0((__int64)&v68, v53, v55, v54, (__int64)&v65, a6);
      sub_2B3EEF0((__int64)&v68, &v65, &v64);
      v51 = v64;
      v52 = ((unsigned int)v69 >> 1) + 1;
LABEL_98:
      v56 = v51[2] == -4096;
      LODWORD(v69) = v69 & 1 | (2 * v52);
      if ( !v56 || v51[1] != -4096 || *v51 != -4096 )
        --HIDWORD(v69);
      v51[2] = v67;
      v51[1] = v66;
      *v51 = v65;
      v57 = v51 + 5;
      v22 = (__int64)(v51 + 3);
      *(_QWORD *)v22 = v57;
      *(_QWORD *)(v22 + 8) = 0x600000000LL;
LABEL_24:
      v23 = *(unsigned int *)(v22 + 8);
      if ( (unsigned int)v8 >= v23 )
      {
        if ( !(_DWORD)v23 )
        {
          v24 = 0;
          goto LABEL_28;
        }
        v58 = v22;
        v65 = sub_D35010(
                *(_QWORD *)(*(_QWORD *)(v11 - 64) + 8LL),
                *(_QWORD *)(v11 - 32),
                *(_QWORD *)(*(_QWORD *)(v11 - 64) + 8LL),
                *(_QWORD *)(**(_QWORD **)v22 - 32LL),
                *(_QWORD *)(a2 + 3344),
                *(_QWORD *)(a2 + 3288),
                1,
                1);
        if ( BYTE4(v65) )
        {
          v22 = v58;
          v24 = *(unsigned int *)(v58 + 8);
LABEL_28:
          if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(v22 + 12) )
          {
            v59 = v22;
            sub_C8D5F0(v22, (const void *)(v22 + 16), v24 + 1, 8u, a5, a6);
            v22 = v59;
            v24 = *(unsigned int *)(v59 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v22 + 8 * v24) = v11;
          ++*(_DWORD *)(v22 + 8);
        }
      }
LABEL_10:
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
        goto LABEL_5;
    }
  }
LABEL_31:
  v25 = (unsigned int)v69 >> 1;
  v26 = (__int64)(a1 + 2);
  a1[1] = 0x100000000LL;
  *a1 = (__int64)(a1 + 2);
  v27 = v69;
  if ( v25 )
  {
    v28 = (unsigned __int64)(a1 + 2);
    if ( v25 != 1 )
    {
      sub_2B423B0((__int64)a1, v25, 0x100000000LL, v26, a5, a6);
      v26 = *a1;
      v28 = *a1 + ((unsigned __int64)*((unsigned int *)a1 + 2) << 6);
    }
    for ( i = v26 + ((unsigned __int64)v25 << 6); i != v28; v28 += 64LL )
    {
      if ( v28 )
      {
        *(_DWORD *)(v28 + 8) = 0;
        *(_QWORD *)v28 = v28 + 16;
        *(_DWORD *)(v28 + 12) = 6;
      }
    }
    v30 = v69;
    *((_DWORD *)a1 + 2) = v25;
    v27 = v69;
    v31 = v69 & 1;
    v26 = v30 >> 1;
    if ( (_DWORD)v26 )
    {
      if ( (_BYTE)v31 )
      {
        v36 = v72;
        v34 = &v70;
        v35 = v73;
        if ( v72 == -4096 )
          goto LABEL_71;
        goto LABEL_42;
      }
      v32 = v71;
      v33 = (__int64)v70;
      v34 = v70;
      v26 = 11LL * v71;
      v35 = &v70[11 * v71];
      if ( v70 != (_QWORD *)v35 )
      {
        while ( 1 )
        {
          v36 = v34[2];
          if ( v36 == -4096 )
          {
LABEL_71:
            if ( v34[1] != -4096 || *v34 != -4096 )
              goto LABEL_43;
          }
          else
          {
LABEL_42:
            if ( v36 != -8192 || v34[1] != -8192 || *v34 != -8192 )
              goto LABEL_43;
          }
          v34 += 11;
          if ( v35 == (_BYTE *)v34 )
            goto LABEL_43;
        }
      }
LABEL_82:
      v31 = 5 * v32;
      v37 = (_BYTE *)(v33 + 88 * v32);
      if ( v34 == (_QWORD *)v37 )
        goto LABEL_56;
LABEL_45:
      v38 = 0;
      do
      {
LABEL_48:
        v39 = v38++;
        sub_2B3B230(*a1 + (v39 << 6), (__int64)(v34 + 3), v31, v26, a5, a6);
        do
        {
          while ( 1 )
          {
            v34 += 11;
            if ( v35 == (_BYTE *)v34 )
            {
LABEL_47:
              if ( v34 == (_QWORD *)v37 )
                goto LABEL_54;
              goto LABEL_48;
            }
            v40 = v34[2];
            if ( v40 == -4096 )
              break;
            if ( v40 != -8192 || v34[1] != -8192 || *v34 != -8192 )
              goto LABEL_47;
          }
          if ( v34[1] != -4096 )
            goto LABEL_47;
        }
        while ( *v34 == -4096 );
      }
      while ( v34 != (_QWORD *)v37 );
LABEL_54:
      if ( (v69 & 1) == 0 )
      {
        v33 = (__int64)v70;
LABEL_56:
        v41 = v71;
        v42 = v71;
        if ( !v71 )
        {
LABEL_86:
          sub_C7D6A0(v33, 88 * v41, 8);
          return a1;
        }
        v43 = (_BYTE *)(v33 + 88LL * v71);
        v44 = (_QWORD *)v33;
        if ( v43 != (_BYTE *)v33 )
          goto LABEL_63;
LABEL_85:
        v41 = v42;
        goto LABEL_86;
      }
      goto LABEL_87;
    }
  }
  v48 = v27 & 1;
  v31 = v48;
  if ( v48 )
  {
    v49 = &v70;
    v50 = 88;
  }
  else
  {
    v49 = v70;
    v26 = 5LL * v71;
    v50 = 11LL * v71;
  }
  v34 = &v49[v50];
  v35 = v34;
LABEL_43:
  if ( !(_BYTE)v31 )
  {
    v33 = (__int64)v70;
    v32 = v71;
    goto LABEL_82;
  }
  v37 = v73;
  if ( v34 != (_QWORD *)v73 )
    goto LABEL_45;
LABEL_87:
  v43 = v73;
  v44 = &v70;
  do
  {
LABEL_63:
    while ( 1 )
    {
      v46 = v44[2];
      if ( v46 == -4096 )
        break;
      if ( v46 != -8192 || v44[1] != -8192 || *v44 != -8192 )
      {
LABEL_60:
        v45 = v44[3];
        if ( (_QWORD *)v45 != v44 + 5 )
          _libc_free(v45);
      }
      v44 += 11;
      if ( v43 == (_BYTE *)v44 )
        goto LABEL_67;
    }
    if ( v44[1] != -4096 || *v44 != -4096 )
      goto LABEL_60;
    v44 += 11;
  }
  while ( v43 != (_BYTE *)v44 );
LABEL_67:
  if ( (v69 & 1) == 0 )
  {
    v42 = v71;
    v33 = (__int64)v70;
    goto LABEL_85;
  }
  return a1;
}
