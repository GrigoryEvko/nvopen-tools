// Function: sub_B2AF20
// Address: 0xb2af20
//
__int64 __fastcall sub_B2AF20(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdi
  __int64 v4; // r15
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 *v7; // rax
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned int v11; // ecx
  unsigned int v12; // esi
  __int64 v13; // r8
  int v14; // r10d
  __int64 *v15; // rdx
  unsigned int v16; // edi
  _QWORD *v17; // rax
  __int64 v18; // rcx
  unsigned __int64 *v19; // r8
  unsigned __int64 v20; // r14
  _QWORD *v21; // rdx
  __int64 v22; // rcx
  _QWORD *v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rcx
  __int64 v26; // rsi
  int v28; // eax
  int v29; // ecx
  __int64 v30; // rax
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // r14
  int v33; // ecx
  __int64 v34; // rax
  int v35; // eax
  unsigned int v36; // r13d
  int v37; // r12d
  __int64 v38; // rax
  __int64 *v39; // r15
  int v40; // ecx
  __int64 v41; // rax
  __int64 *v42; // r8
  __int64 v43; // rdx
  unsigned __int64 v44; // rax
  __int64 v45; // rdx
  int v46; // eax
  int v47; // edi
  __int64 v48; // rsi
  unsigned int v49; // eax
  __int64 v50; // r8
  int v51; // r10d
  __int64 *v52; // r9
  int v53; // eax
  int v54; // eax
  __int64 v55; // rdi
  __int64 *v56; // r8
  unsigned int v57; // r14d
  int v58; // r9d
  __int64 v59; // rsi
  __int64 v60; // [rsp+8h] [rbp-168h]
  __int64 v61; // [rsp+18h] [rbp-158h]
  __int64 *v62; // [rsp+18h] [rbp-158h]
  __int64 v63; // [rsp+28h] [rbp-148h]
  __int64 *v64; // [rsp+28h] [rbp-148h]
  unsigned __int64 v65; // [rsp+28h] [rbp-148h]
  _QWORD *v66; // [rsp+30h] [rbp-140h] BYREF
  unsigned int v67; // [rsp+38h] [rbp-138h]
  unsigned int v68; // [rsp+3Ch] [rbp-134h]
  _QWORD v69[38]; // [rsp+40h] [rbp-130h] BYREF

  v3 = v69;
  LODWORD(v4) = 1;
  v5 = *(_QWORD *)(a2 + 80);
  v66 = v69;
  v6 = v5 - 24;
  v68 = 16;
  if ( !v5 )
    v6 = 0;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  v60 = v6;
  v69[0] = v6;
  v69[1] = v6;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  v67 = 1;
  do
  {
    v7 = &v3[2 * (unsigned int)v4 - 2];
    v8 = *v7;
    v9 = v7[1];
    v67 = v4 - 1;
    v10 = sub_AA4FF0(v8);
    if ( !v10 )
      BUG();
    v11 = *(unsigned __int8 *)(v10 - 24) - 39;
    if ( v11 <= 0x38 && ((1LL << v11) & 0x100060000000001LL) != 0 )
      v9 = v8;
    v12 = *(_DWORD *)(a1 + 24);
    if ( !v12 )
    {
      ++*(_QWORD *)a1;
      goto LABEL_79;
    }
    v13 = *(_QWORD *)(a1 + 8);
    v14 = 1;
    v15 = 0;
    v16 = (v12 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v17 = (_QWORD *)(v13 + 16LL * v16);
    v18 = *v17;
    if ( v8 != *v17 )
    {
      while ( v18 != -4096 )
      {
        if ( v18 == -8192 && !v15 )
          v15 = v17;
        v16 = (v12 - 1) & (v14 + v16);
        v17 = (_QWORD *)(v13 + 16LL * v16);
        v18 = *v17;
        if ( v8 == *v17 )
          goto LABEL_10;
        ++v14;
      }
      if ( !v15 )
        v15 = v17;
      v28 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v29 = v28 + 1;
      if ( 4 * (v28 + 1) < 3 * v12 )
      {
        if ( v12 - *(_DWORD *)(a1 + 20) - v29 <= v12 >> 3 )
        {
          sub_B2ACE0(a1, v12);
          v53 = *(_DWORD *)(a1 + 24);
          if ( !v53 )
          {
LABEL_107:
            ++*(_DWORD *)(a1 + 16);
            BUG();
          }
          v54 = v53 - 1;
          v55 = *(_QWORD *)(a1 + 8);
          v56 = 0;
          v57 = v54 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v58 = 1;
          v29 = *(_DWORD *)(a1 + 16) + 1;
          v15 = (__int64 *)(v55 + 16LL * v57);
          v59 = *v15;
          if ( v8 != *v15 )
          {
            while ( v59 != -4096 )
            {
              if ( v59 == -8192 && !v56 )
                v56 = v15;
              v57 = v54 & (v58 + v57);
              v15 = (__int64 *)(v55 + 16LL * v57);
              v59 = *v15;
              if ( v8 == *v15 )
                goto LABEL_33;
              ++v58;
            }
            if ( v56 )
              v15 = v56;
          }
        }
        goto LABEL_33;
      }
LABEL_79:
      sub_B2ACE0(a1, 2 * v12);
      v46 = *(_DWORD *)(a1 + 24);
      if ( !v46 )
        goto LABEL_107;
      v47 = v46 - 1;
      v48 = *(_QWORD *)(a1 + 8);
      v49 = (v46 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v29 = *(_DWORD *)(a1 + 16) + 1;
      v15 = (__int64 *)(v48 + 16LL * v49);
      v50 = *v15;
      if ( *v15 != v8 )
      {
        v51 = 1;
        v52 = 0;
        while ( v50 != -4096 )
        {
          if ( !v52 && v50 == -8192 )
            v52 = v15;
          v49 = v47 & (v51 + v49);
          v15 = (__int64 *)(v48 + 16LL * v49);
          v50 = *v15;
          if ( v8 == *v15 )
            goto LABEL_33;
          ++v51;
        }
        if ( v52 )
          v15 = v52;
      }
LABEL_33:
      *(_DWORD *)(a1 + 16) = v29;
      if ( *v15 != -4096 )
        --*(_DWORD *)(a1 + 20);
      v19 = (unsigned __int64 *)(v15 + 1);
      *v15 = v8;
      v15[1] = 0;
      v23 = v15 + 1;
      v21 = v15 + 1;
      goto LABEL_36;
    }
LABEL_10:
    v19 = v17 + 1;
    v20 = v17[1] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v17[1] & 4) == 0 )
    {
      v21 = v17 + 1;
      if ( v20 )
      {
        v23 = v17 + 2;
        v40 = 0;
        goto LABEL_57;
      }
      v23 = v17 + 1;
LABEL_36:
      v20 = 0;
      v26 = 0;
      goto LABEL_37;
    }
    v21 = *(_QWORD **)v20;
    v22 = 8LL * *(unsigned int *)(v20 + 8);
    v23 = (_QWORD *)(*(_QWORD *)v20 + v22);
    v24 = v22 >> 3;
    v25 = v22 >> 5;
    if ( v25 )
    {
      while ( v9 != *v21 )
      {
        if ( v9 == v21[1] )
        {
          ++v21;
          goto LABEL_18;
        }
        if ( v9 == v21[2] )
        {
          v21 += 2;
          goto LABEL_18;
        }
        if ( v9 == v21[3] )
        {
          v21 += 3;
          goto LABEL_18;
        }
        v21 += 4;
        if ( !--v25 )
        {
          v24 = v23 - v21;
          goto LABEL_65;
        }
      }
      goto LABEL_18;
    }
LABEL_65:
    if ( v24 != 2 )
    {
      if ( v24 != 3 )
      {
        if ( v24 != 1 )
        {
          v21 = v23;
          goto LABEL_18;
        }
        v40 = 1;
        goto LABEL_57;
      }
      if ( v9 == *v21 )
        goto LABEL_18;
      ++v21;
    }
    if ( v9 == *v21 )
      goto LABEL_18;
    ++v21;
    v40 = 1;
LABEL_57:
    if ( v9 != *v21 )
      v21 = v23;
    if ( v40 != 1 )
    {
      v26 = 0;
LABEL_37:
      LODWORD(v4) = v67;
      if ( v23 != v21 )
        goto LABEL_19;
      goto LABEL_38;
    }
LABEL_18:
    LODWORD(v4) = v67;
    v26 = 1;
    if ( v23 != v21 )
      goto LABEL_19;
LABEL_38:
    if ( v20 )
    {
      if ( (_DWORD)v26 != 1 )
      {
        v64 = (__int64 *)v19;
        v41 = sub_22077B0(48);
        v42 = v64;
        if ( v41 )
        {
          *(_QWORD *)v41 = v41 + 16;
          *(_QWORD *)(v41 + 8) = 0x400000000LL;
        }
        v43 = v41;
        v44 = v41 & 0xFFFFFFFFFFFFFFF8LL;
        *v64 = v43 | 4;
        v45 = *(unsigned int *)(v44 + 8);
        if ( v45 + 1 > (unsigned __int64)*(unsigned int *)(v44 + 12) )
        {
          v26 = v44 + 16;
          v62 = v64;
          v65 = v44;
          sub_C8D5F0(v44, v44 + 16, v45 + 1, 8);
          v44 = v65;
          v42 = v62;
          v45 = *(unsigned int *)(v65 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v44 + 8 * v45) = v20;
        ++*(_DWORD *)(v44 + 8);
        v20 = *v42 & 0xFFFFFFFFFFFFFFF8LL;
      }
      v30 = *(unsigned int *)(v20 + 8);
      if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(v20 + 12) )
      {
        v26 = v20 + 16;
        sub_C8D5F0(v20, v20 + 16, v30 + 1, 8);
        v30 = *(unsigned int *)(v20 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v20 + 8 * v30) = v9;
      ++*(_DWORD *)(v20 + 8);
    }
    else
    {
      *v19 = v9 & 0xFFFFFFFFFFFFFFFBLL;
    }
    v31 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v31 == v8 + 48 )
      goto LABEL_72;
    if ( !v31 )
      BUG();
    v32 = v31 - 24;
    v33 = *(unsigned __int8 *)(v31 - 24);
    if ( (unsigned int)(v33 - 30) > 0xA )
LABEL_72:
      BUG();
    if ( (_BYTE)v33 == 38 )
    {
      v9 = v60;
      v34 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(v31 - 88) - 32LL) - 8LL);
      if ( *(_BYTE *)v34 != 21 )
        v9 = *(_QWORD *)(v34 + 40);
    }
    v4 = v67;
    v35 = sub_B46E30(v32);
    if ( v35 )
    {
      v63 = a1;
      v36 = 0;
      v37 = v35;
      do
      {
        v26 = v36;
        v38 = sub_B46EC0(v32, v36);
        if ( v4 + 1 > (unsigned __int64)v68 )
        {
          v26 = (__int64)v69;
          v61 = v38;
          sub_C8D5F0(&v66, v69, v4 + 1, 16);
          v4 = v67;
          v38 = v61;
        }
        v39 = &v66[2 * v4];
        ++v36;
        *v39 = v38;
        v39[1] = v9;
        v4 = ++v67;
      }
      while ( v37 != v36 );
      a1 = v63;
    }
LABEL_19:
    v3 = v66;
  }
  while ( (_DWORD)v4 );
  if ( v66 != v69 )
    _libc_free(v66, v26);
  return a1;
}
