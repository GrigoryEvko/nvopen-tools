// Function: sub_1F10BF0
// Address: 0x1f10bf0
//
unsigned __int64 __fastcall sub_1F10BF0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 v4; // r11
  __int64 v6; // rax
  _QWORD *v7; // rbx
  __int64 v8; // r8
  unsigned int v9; // edi
  _QWORD *v10; // r13
  unsigned int v11; // r10d
  unsigned int v12; // edx
  _QWORD *v13; // rcx
  unsigned __int64 v14; // r14
  unsigned int v15; // r9d
  unsigned int v16; // edx
  unsigned __int64 v17; // rdx
  unsigned int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // rbx
  __int64 v21; // r15
  unsigned __int64 result; // rax
  unsigned int v23; // ecx
  __int64 v24; // rdi
  unsigned int v25; // esi
  __int64 *v26; // rdx
  __int64 v27; // r8
  __int64 v28; // r12
  unsigned __int64 v29; // rbx
  _QWORD *v30; // r9
  unsigned __int64 v31; // r15
  _QWORD *v32; // r12
  char v33; // r14
  __int64 v34; // rsi
  __int64 v35; // rdx
  __int64 v36; // rdi
  __int64 *v37; // rcx
  __int64 v38; // r8
  int v39; // r12d
  unsigned __int64 v40; // rcx
  __int64 v41; // rdx
  unsigned __int64 v42; // rdx
  int v43; // ecx
  unsigned __int64 v44; // rdx
  unsigned __int64 i; // rbx
  int v46; // ecx
  unsigned __int64 *v47; // rax
  unsigned __int64 *v48; // rdx
  unsigned __int64 j; // rbx
  unsigned __int64 v50; // r14
  __int64 v51; // rbx
  _QWORD *v52; // rax
  _QWORD *v53; // rdx
  __int64 v54; // rax
  unsigned __int64 v55; // rax
  __int64 v56; // rsi
  __int64 v57; // rdx
  __int64 v58; // rcx
  int v59; // edi
  unsigned int v60; // edx
  __int64 v61; // r8
  int v62; // eax
  int v63; // r9d
  int v64; // r8d
  __int64 v65; // rcx
  int v66; // edx
  int v67; // r9d
  int v68; // eax
  int v69; // r10d
  __int64 v70; // rax
  char v71; // [rsp+Fh] [rbp-51h]
  _QWORD *v72; // [rsp+10h] [rbp-50h]
  _QWORD *v73; // [rsp+18h] [rbp-48h]
  int v74; // [rsp+18h] [rbp-48h]
  unsigned int v75; // [rsp+20h] [rbp-40h]
  __int64 v76; // [rsp+20h] [rbp-40h]
  unsigned __int64 v77; // [rsp+28h] [rbp-38h]

  v4 = a1;
  v6 = a4;
  v7 = *(_QWORD **)(a2 + 32);
  v72 = v7;
  if ( a3 == v7 )
  {
LABEL_54:
    v77 = a2 + 24;
    if ( a4 != a2 + 24 )
    {
      v8 = *(_QWORD *)(v4 + 368);
      v9 = *(_DWORD *)(v4 + 384);
      v10 = *(_QWORD **)(a2 + 32);
      goto LABEL_6;
    }
    v10 = *(_QWORD **)(a2 + 32);
    v14 = a2 + 24;
LABEL_62:
    v71 = 1;
    v21 = *(_QWORD *)(*(_QWORD *)(v4 + 392) + 16LL * *(unsigned int *)(a2 + 48));
    if ( v77 != v14 )
      goto LABEL_15;
    goto LABEL_63;
  }
  v8 = *(_QWORD *)(a1 + 368);
  v9 = *(_DWORD *)(a1 + 384);
  v10 = a3;
  v11 = v9 - 1;
  while ( !v9 )
  {
LABEL_47:
    v40 = *v10 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v40 )
      BUG();
    v41 = *(_QWORD *)v40;
    v10 = (_QWORD *)(*v10 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (*(_QWORD *)v40 & 4) == 0 && (*(_BYTE *)(v40 + 46) & 4) != 0 )
    {
      while ( 1 )
      {
        v42 = v41 & 0xFFFFFFFFFFFFFFF8LL;
        v10 = (_QWORD *)v42;
        if ( (*(_BYTE *)(v42 + 46) & 4) == 0 )
          break;
        v41 = *(_QWORD *)v42;
      }
    }
    if ( v10 == v7 )
      goto LABEL_54;
  }
  v12 = v11 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
  v13 = *(_QWORD **)(v8 + 16LL * v12);
  if ( v10 != v13 )
  {
    v39 = 1;
    while ( v13 != (_QWORD *)-8LL )
    {
      v12 = v11 & (v39 + v12);
      v13 = *(_QWORD **)(v8 + 16LL * v12);
      if ( v13 == v10 )
        goto LABEL_5;
      ++v39;
    }
    goto LABEL_47;
  }
LABEL_5:
  v77 = a2 + 24;
  v14 = a2 + 24;
  if ( a4 == a2 + 24 )
  {
LABEL_9:
    if ( v10 == v7 )
      goto LABEL_62;
  }
  else
  {
LABEL_6:
    v15 = v9 - 1;
    do
    {
      if ( v9 )
      {
        v16 = v15 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v14 = *(_QWORD *)(v8 + 16LL * v16);
        if ( v6 == v14 )
          goto LABEL_9;
        v43 = 1;
        while ( v14 != -8 )
        {
          v16 = v15 & (v43 + v16);
          v14 = *(_QWORD *)(v8 + 16LL * v16);
          if ( v14 == v6 )
            goto LABEL_9;
          ++v43;
        }
      }
      if ( !v6 )
        BUG();
      if ( (*(_BYTE *)v6 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v6 + 46) & 8) != 0 )
          v6 = *(_QWORD *)(v6 + 8);
      }
      v6 = *(_QWORD *)(v6 + 8);
    }
    while ( v6 != v77 );
    v14 = v77;
    if ( v10 == v7 )
      goto LABEL_62;
  }
  v17 = (unsigned __int64)v10;
  if ( (*((_BYTE *)v10 + 46) & 4) != 0 )
  {
    do
      v17 = *(_QWORD *)v17 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v17 + 46) & 4) != 0 );
  }
  if ( v9 )
  {
    v18 = (v9 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
    v19 = (__int64 *)(v8 + 16LL * v18);
    v20 = *v19;
    if ( *v19 == v17 )
      goto LABEL_14;
    v68 = 1;
    while ( v20 != -8 )
    {
      v69 = v68 + 1;
      v70 = (v9 - 1) & (v18 + v68);
      v18 = v70;
      v19 = (__int64 *)(v8 + 16 * v70);
      v20 = *v19;
      if ( *v19 == v17 )
        goto LABEL_14;
      v68 = v69;
    }
  }
  v19 = (__int64 *)(v8 + 16LL * v9);
LABEL_14:
  v71 = 0;
  v21 = v19[1];
  if ( v77 == v14 )
  {
LABEL_63:
    result = *(_QWORD *)(v4 + 392) + 16LL * *(unsigned int *)(a2 + 48);
    v28 = *(_QWORD *)(result + 8);
    goto LABEL_20;
  }
LABEL_15:
  for ( result = v14; (*(_BYTE *)(result + 46) & 4) != 0; result = *(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v23 = *(_DWORD *)(v4 + 384);
  v24 = *(_QWORD *)(v4 + 368);
  if ( v23 )
  {
    v25 = (v23 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
    v26 = (__int64 *)(v24 + 16LL * v25);
    v27 = *v26;
    if ( *v26 == result )
      goto LABEL_19;
    v66 = 1;
    while ( v27 != -8 )
    {
      v67 = v66 + 1;
      v25 = (v23 - 1) & (v25 + v66);
      v26 = (__int64 *)(v24 + 16LL * v25);
      v27 = *v26;
      if ( *v26 == result )
        goto LABEL_19;
      v66 = v67;
    }
  }
  v26 = (__int64 *)(v24 + 16LL * v23);
LABEL_19:
  v28 = v26[1];
LABEL_20:
  v29 = v14;
  v30 = (_QWORD *)v14;
  v31 = v21 & 0xFFFFFFFFFFFFFFF8LL;
  v32 = (_QWORD *)(v28 & 0xFFFFFFFFFFFFFFF8LL);
  v33 = 0;
LABEL_21:
  if ( (_QWORD *)v31 != v32 )
  {
LABEL_22:
    v34 = v32[2];
    result = 0;
    if ( v77 != v29 && !v33 )
      result = v29;
    goto LABEL_25;
  }
LABEL_33:
  if ( (_QWORD *)v29 != v10 )
    goto LABEL_22;
  if ( !v33 && v71 )
  {
    result = 0;
    v34 = *(_QWORD *)(v31 + 16);
    if ( v77 != v29 )
      result = v29;
    v33 = 0;
    if ( (_QWORD *)v29 == v10 )
    {
LABEL_39:
      if ( v72 == (_QWORD *)v29 && !v33 && result == v34 )
      {
        result = *v32 & 0xFFFFFFFFFFFFFFF8LL;
        v32 = (_QWORD *)result;
        if ( v31 == result )
          goto LABEL_85;
        v34 = *(_QWORD *)(result + 16);
        goto LABEL_44;
      }
      goto LABEL_27;
    }
    while ( 1 )
    {
      if ( result == v34 )
      {
        v32 = (_QWORD *)(*v32 & 0xFFFFFFFFFFFFFFF8LL);
        v44 = *(_QWORD *)v29 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v44 )
          BUG();
        result = *(_QWORD *)v44;
        v29 = *(_QWORD *)v29 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_QWORD *)v44 & 4) == 0 && (*(_BYTE *)(v44 + 46) & 4) != 0 )
        {
          for ( i = *(_QWORD *)v44; ; i = *(_QWORD *)v29 )
          {
            v29 = i & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v29 + 46) & 4) == 0 )
              break;
          }
        }
        goto LABEL_21;
      }
LABEL_27:
      if ( !result )
        goto LABEL_31;
      v35 = *(unsigned int *)(v4 + 384);
      if ( (_DWORD)v35 )
      {
        v36 = *(_QWORD *)(v4 + 368);
        v75 = (v35 - 1) & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
        v37 = (__int64 *)(v36 + 16LL * v75);
        v38 = *v37;
        if ( result == *v37 )
        {
LABEL_30:
          if ( v37 != (__int64 *)(v36 + 16 * v35) )
          {
LABEL_31:
            v32 = (_QWORD *)(*v32 & 0xFFFFFFFFFFFFFFF8LL);
            if ( !v34 )
              goto LABEL_21;
            v73 = v30;
            v76 = v4;
            result = sub_1F10740(v4, v34);
            v4 = v76;
            v30 = v73;
            if ( (_QWORD *)v31 != v32 )
              goto LABEL_22;
            goto LABEL_33;
          }
        }
        else
        {
          v46 = 1;
          while ( v38 != -8 )
          {
            v64 = v46 + 1;
            v65 = ((_DWORD)v35 - 1) & (v75 + v46);
            v74 = v64;
            v75 = v65;
            v37 = (__int64 *)(v36 + 16 * v65);
            v38 = *v37;
            if ( result == *v37 )
              goto LABEL_30;
            v46 = v74;
          }
        }
      }
      if ( (_QWORD *)v29 != v10 )
      {
        v47 = (unsigned __int64 *)(*(_QWORD *)v29 & 0xFFFFFFFFFFFFFFF8LL);
        v48 = v47;
        if ( !v47 )
          BUG();
        v29 = *(_QWORD *)v29 & 0xFFFFFFFFFFFFFFF8LL;
        result = *v47;
        if ( (result & 4) == 0 && (*((_BYTE *)v48 + 46) & 4) != 0 )
        {
          for ( j = result; ; j = *(_QWORD *)v29 )
          {
            v29 = j & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v29 + 46) & 4) == 0 )
              break;
          }
        }
        goto LABEL_21;
      }
      if ( (_QWORD *)v31 == v32 )
        break;
      v34 = v32[2];
LABEL_44:
      v33 = 1;
      result = 0;
LABEL_25:
      if ( (_QWORD *)v29 == v10 )
        goto LABEL_39;
    }
  }
LABEL_85:
  v50 = (unsigned __int64)v30;
  v51 = v4;
  if ( v30 != v10 )
  {
    while ( 2 )
    {
      v52 = (_QWORD *)(*(_QWORD *)v50 & 0xFFFFFFFFFFFFFFF8LL);
      v53 = v52;
      if ( !v52 )
        BUG();
      v50 = *(_QWORD *)v50 & 0xFFFFFFFFFFFFFFF8LL;
      v54 = *v52;
      if ( (v54 & 4) == 0 && (*((_BYTE *)v53 + 46) & 4) != 0 )
      {
        while ( 1 )
        {
          v55 = v54 & 0xFFFFFFFFFFFFFFF8LL;
          v50 = v55;
          if ( (*(_BYTE *)(v55 + 46) & 4) == 0 )
            break;
          v54 = *(_QWORD *)v55;
        }
      }
      result = (unsigned int)**(unsigned __int16 **)(v50 + 16) - 12;
      if ( (unsigned __int16)(**(_WORD **)(v50 + 16) - 12) > 1u )
      {
        v56 = *(_QWORD *)(v51 + 368);
        v57 = *(unsigned int *)(v51 + 384);
        v58 = v56 + 16 * v57;
        if ( (_DWORD)v57 )
        {
          v59 = v57 - 1;
          v60 = (v57 - 1) & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
          result = v56 + 16LL * v60;
          v61 = *(_QWORD *)result;
          if ( *(_QWORD *)result == v50 )
          {
LABEL_99:
            if ( v58 != result )
              goto LABEL_93;
          }
          else
          {
            v62 = 1;
            while ( v61 != -8 )
            {
              v63 = v62 + 1;
              v60 = v59 & (v62 + v60);
              result = v56 + 16LL * v60;
              v61 = *(_QWORD *)result;
              if ( v50 == *(_QWORD *)result )
                goto LABEL_99;
              v62 = v63;
            }
          }
        }
        result = sub_1DC1550(v51, v50, 0);
      }
LABEL_93:
      if ( (_QWORD *)v50 == v10 )
        return result;
      continue;
    }
  }
  return result;
}
