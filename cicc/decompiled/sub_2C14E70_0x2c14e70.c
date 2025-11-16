// Function: sub_2C14E70
// Address: 0x2c14e70
//
unsigned __int64 __fastcall sub_2C14E70(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned int *v5; // rax
  __int64 v6; // rsi
  unsigned __int8 *v7; // r14
  unsigned int v8; // edx
  unsigned int v9; // r8d
  __int64 v10; // r10
  unsigned int v11; // r9d
  int i; // ecx
  unsigned __int8 *v13; // rax
  unsigned int v14; // edi
  int *v15; // rax
  int v16; // ebx
  int v17; // eax
  int v18; // r11d
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // r8
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 *v24; // rbx
  int v25; // r15d
  __int64 v26; // rax
  unsigned int *v27; // rsi
  __int64 v28; // r9
  __int64 v29; // r11
  int j; // ebx
  unsigned int v31; // edx
  __int64 v32; // r8
  int v33; // ecx
  unsigned int v34; // edx
  unsigned int v35; // edi
  __int64 v36; // rax
  int v37; // r10d
  _BYTE *v38; // r8
  __int64 v39; // rbx
  unsigned __int64 v40; // r13
  int v42; // eax
  signed __int64 v43; // rax
  int v44; // edx
  int v45; // esi
  __int64 v46; // rdx
  __int64 v47; // rdx
  __int64 v48; // rax
  bool v49; // of
  unsigned __int64 v50; // rbx
  __int64 v51; // rdx
  unsigned __int64 v52; // r13
  int v53; // [rsp+0h] [rbp-90h]
  __int64 v54; // [rsp+0h] [rbp-90h]
  __int64 v55; // [rsp+10h] [rbp-80h]
  __int64 v57; // [rsp+38h] [rbp-58h]
  _BYTE *v58; // [rsp+40h] [rbp-50h] BYREF
  __int64 v59; // [rsp+48h] [rbp-48h]
  _BYTE v60[64]; // [rsp+50h] [rbp-40h] BYREF

  v5 = (unsigned int *)a1[12];
  v6 = *v5;
  v7 = (unsigned __int8 *)*((_QWORD *)v5 + 6);
  if ( (_DWORD)v6 )
  {
    v8 = v5[10];
    v9 = v5[8];
    v6 = 0;
    v10 = *((_QWORD *)v5 + 2);
    v11 = v9 - 1;
    for ( i = 37 * v8; ; i += 37 )
    {
      if ( v9 )
      {
        v14 = i & v11;
        v15 = (int *)(v10 + 16LL * (i & v11));
        v16 = *v15;
        if ( *v15 == v8 )
        {
LABEL_3:
          v13 = (unsigned __int8 *)*((_QWORD *)v15 + 1);
          if ( v13 )
          {
            if ( v7 == v13 )
              break;
            v6 = (unsigned int)(v6 + 1);
          }
        }
        else
        {
          v17 = 1;
          while ( v16 != 0x7FFFFFFF )
          {
            v18 = v17 + 1;
            v14 = v11 & (v17 + v14);
            v15 = (int *)(v10 + 16LL * v14);
            v16 = *v15;
            if ( *v15 == v8 )
              goto LABEL_3;
            v17 = v18;
          }
        }
      }
      ++v8;
    }
  }
  v19 = a1[2];
  v20 = v19 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v19 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v21 = a1[2] & 0xFFFFFFFFFFFFFFF8LL;
    v22 = v19 >> 2;
    if ( (v19 & 4) != 0 )
    {
      if ( *(_DWORD *)(v20 + 8) )
      {
        v23 = (unsigned int)v6;
        if ( (v22 & 1) == 0 )
          goto LABEL_21;
        goto LABEL_17;
      }
    }
    else
    {
      if ( (v22 & 1) == 0 )
        goto LABEL_21;
      v23 = v6;
      if ( *(_DWORD *)(v20 + 8) )
      {
LABEL_17:
        v21 = *(_QWORD *)(*(_QWORD *)v20 + 8 * v23);
        goto LABEL_21;
      }
    }
  }
  v21 = *(_QWORD *)(a1[6] + 8 * v6 + 8);
LABEL_21:
  v24 = (__int64 *)sub_2BFD6A0(a3 + 16, v21);
  v55 = sub_2AAEDF0((__int64)v24, a2);
  v25 = *(_DWORD *)a1[12];
  LODWORD(v57) = v25 * a2;
  BYTE4(v57) = BYTE4(a2);
  v26 = sub_BCE1B0(v24, v57);
  v27 = (unsigned int *)a1[12];
  v28 = 0;
  v29 = v26;
  v58 = v60;
  v59 = 0x400000000LL;
  if ( v25 )
  {
    for ( j = 0; j != v25; ++j )
    {
      v31 = v27[8];
      v32 = *((_QWORD *)v27 + 2);
      v33 = j + v27[10];
      if ( v31 )
      {
        v34 = v31 - 1;
        v35 = v34 & (37 * v33);
        v36 = v32 + 16LL * v35;
        v37 = *(_DWORD *)v36;
        if ( *(_DWORD *)v36 == v33 )
        {
LABEL_25:
          if ( *(_QWORD *)(v36 + 8) )
          {
            if ( v28 + 1 > (unsigned __int64)HIDWORD(v59) )
            {
              v54 = v29;
              sub_C8D5F0((__int64)&v58, v60, v28 + 1, 4u, v32, v28);
              v28 = (unsigned int)v59;
              v29 = v54;
            }
            *(_DWORD *)&v58[4 * v28] = j;
            v27 = (unsigned int *)a1[12];
            v28 = (unsigned int)(v59 + 1);
            LODWORD(v59) = v59 + 1;
          }
        }
        else
        {
          v42 = 1;
          while ( v37 != 0x7FFFFFFF )
          {
            v35 = v34 & (v42 + v35);
            v53 = v42 + 1;
            v36 = v32 + 16LL * v35;
            v37 = *(_DWORD *)v36;
            if ( v33 == *(_DWORD *)v36 )
              goto LABEL_25;
            v42 = v53;
          }
        }
      }
    }
    v38 = v58;
  }
  else
  {
    v38 = v60;
  }
  v39 = sub_DFD610(*(__int64 **)a3, (unsigned int)*v7 - 29, v29, *v27, (__int64)v38);
  if ( !*(_BYTE *)(a1[12] + 4LL) )
  {
    v40 = v39;
    goto LABEL_33;
  }
  v43 = sub_DFBC30(*(__int64 **)a3, 1, v55, 0, 0, *(unsigned int *)(a3 + 176), 0, 0, 0, 0, 0);
  v45 = v44;
  v46 = *(unsigned int *)(a1[12] + 24LL);
  if ( v45 != 1 )
  {
    v47 = v43 * v46;
    if ( is_mul_ok(v43, *(unsigned int *)(a1[12] + 24LL)) )
    {
      v48 = v47;
LABEL_43:
      v49 = __OFADD__(v48, v39);
      v50 = v48 + v39;
      if ( v49 )
      {
        v50 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v48 <= 0 )
          v50 = 0x8000000000000000LL;
      }
      goto LABEL_44;
    }
    if ( v43 > 0 && *(_DWORD *)(a1[12] + 24LL) )
    {
      v52 = 0x7FFFFFFFFFFFFFFFLL;
      v49 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v39);
      v50 = v39 + 0x7FFFFFFFFFFFFFFFLL;
      if ( !v49 )
        goto LABEL_44;
    }
    else
    {
      v52 = 0x8000000000000000LL;
      v49 = __OFADD__(0x8000000000000000LL, v39);
      v50 = v39 + 0x8000000000000000LL;
      if ( !v49 )
        goto LABEL_44;
    }
    goto LABEL_51;
  }
  v51 = v43 * v46;
  if ( is_mul_ok(v43, *(unsigned int *)(a1[12] + 24LL)) )
  {
    v48 = v51;
    goto LABEL_43;
  }
  if ( v43 <= 0 || !*(_DWORD *)(a1[12] + 24LL) )
  {
    v52 = 0x8000000000000000LL;
    v49 = __OFADD__(0x8000000000000000LL, v39);
    v50 = v39 + 0x8000000000000000LL;
    if ( !v49 )
      goto LABEL_44;
    goto LABEL_51;
  }
  v52 = 0x7FFFFFFFFFFFFFFFLL;
  v49 = __OFADD__(0x7FFFFFFFFFFFFFFFLL, v39);
  v50 = v39 + 0x7FFFFFFFFFFFFFFFLL;
  if ( v49 )
LABEL_51:
    v50 = v52;
LABEL_44:
  v40 = v50;
LABEL_33:
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
  return v40;
}
