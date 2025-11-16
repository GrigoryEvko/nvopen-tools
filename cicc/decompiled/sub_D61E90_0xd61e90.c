// Function: sub_D61E90
// Address: 0xd61e90
//
__int64 __fastcall sub_D61E90(__int64 a1, __int64 a2, _BYTE *a3)
{
  int v6; // eax
  __int64 v7; // r15
  char v8; // al
  __int64 v9; // rdi
  unsigned int v10; // esi
  int v11; // esi
  unsigned int v12; // edx
  __int64 v13; // rbx
  _BYTE *v14; // r8
  unsigned int v15; // eax
  unsigned int v16; // eax
  unsigned int v18; // edx
  __int64 v19; // rcx
  int v20; // edi
  unsigned int v21; // r8d
  unsigned int v22; // eax
  bool v23; // cc
  char v24; // cl
  __int64 v25; // rdi
  int v26; // esi
  unsigned int v27; // eax
  __int64 v28; // rdx
  __int64 v29; // r9
  __int64 v30; // rbx
  unsigned int v31; // edx
  _QWORD *v32; // rbx
  int v33; // eax
  unsigned int v34; // edi
  unsigned int v35; // esi
  int v36; // r9d
  __int64 v37; // rsi
  int v38; // edx
  unsigned int v39; // eax
  _BYTE *v40; // rdi
  __int64 v41; // rdi
  int v42; // esi
  unsigned int v43; // eax
  _BYTE *v44; // rdx
  int v45; // r9d
  __int64 v46; // r8
  int v47; // edx
  int v48; // esi
  int v49; // r10d
  __int64 v50; // rsi
  int v51; // edx
  unsigned int v52; // eax
  __int64 v53; // rcx
  __int64 v54; // rsi
  int v55; // edx
  unsigned int v56; // eax
  _BYTE *v57; // rcx
  int v58; // r9d
  _QWORD *v59; // rdi
  int v60; // edx
  int v61; // edx
  int v62; // r9d
  int v63; // r9d
  __int64 v64; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v65; // [rsp+18h] [rbp-48h]
  __int64 v66; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v67; // [rsp+28h] [rbp-38h]

  v6 = (unsigned __int8)*a3;
  if ( (unsigned __int8)v6 <= 0x1Cu )
  {
    switch ( (_BYTE)v6 )
    {
      case 0x16:
        sub_D5DD50(a1, a2, (__int64)a3);
        return a1;
      case 0x14:
        sub_D5E190(a1, a2, (__int64)a3);
        return a1;
      case 1:
        sub_D65DE0();
        return a1;
      case 3:
        sub_D5E330(a1, a2, (__int64)a3);
        return a1;
    }
    if ( (unsigned int)(v6 - 12) <= 1 )
    {
      sub_D5EA50(a1, a2);
      return a1;
    }
LABEL_25:
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 8) = 1;
    *(_DWORD *)(a1 + 24) = 1;
    return a1;
  }
  v7 = a2 + 56;
  v8 = *(_BYTE *)(a2 + 64) & 1;
  if ( v8 )
  {
    v9 = a2 + 72;
    v11 = 7;
  }
  else
  {
    v9 = *(_QWORD *)(a2 + 72);
    v10 = *(_DWORD *)(a2 + 80);
    if ( !v10 )
    {
      v18 = *(_DWORD *)(a2 + 64);
      ++*(_QWORD *)(a2 + 56);
      v19 = 0;
      v20 = (v18 >> 1) + 1;
LABEL_19:
      v21 = 3 * v10;
      goto LABEL_20;
    }
    v11 = v10 - 1;
  }
  v12 = v11 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v13 = v9 + 40LL * (v11 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)));
  v14 = *(_BYTE **)v13;
  if ( a3 == *(_BYTE **)v13 )
  {
LABEL_6:
    v15 = *(_DWORD *)(v13 + 16);
    *(_DWORD *)(a1 + 8) = v15;
    if ( v15 > 0x40 )
      sub_C43780(a1, (const void **)(v13 + 8));
    else
      *(_QWORD *)a1 = *(_QWORD *)(v13 + 8);
    v16 = *(_DWORD *)(v13 + 32);
    *(_DWORD *)(a1 + 24) = v16;
    if ( v16 > 0x40 )
      sub_C43780(a1 + 16, (const void **)(v13 + 24));
    else
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(v13 + 24);
    return a1;
  }
  v36 = 1;
  v19 = 0;
  while ( v14 != (_BYTE *)-4096LL )
  {
    if ( !v19 && v14 == (_BYTE *)-8192LL )
      v19 = v13;
    v12 = v11 & (v36 + v12);
    v13 = v9 + 40LL * v12;
    v14 = *(_BYTE **)v13;
    if ( a3 == *(_BYTE **)v13 )
      goto LABEL_6;
    ++v36;
  }
  v18 = *(_DWORD *)(a2 + 64);
  if ( !v19 )
    v19 = v13;
  ++*(_QWORD *)(a2 + 56);
  v20 = (v18 >> 1) + 1;
  if ( !v8 )
  {
    v10 = *(_DWORD *)(a2 + 80);
    goto LABEL_19;
  }
  v21 = 24;
  v10 = 8;
LABEL_20:
  if ( 4 * v20 < v21 )
  {
    if ( v10 - *(_DWORD *)(a2 + 68) - v20 > v10 >> 3 )
      goto LABEL_22;
    sub_D5FDE0(v7, v10);
    if ( (*(_BYTE *)(a2 + 64) & 1) != 0 )
    {
      v41 = a2 + 72;
      v42 = 7;
      goto LABEL_65;
    }
    v48 = *(_DWORD *)(a2 + 80);
    v41 = *(_QWORD *)(a2 + 72);
    if ( v48 )
    {
      v42 = v48 - 1;
LABEL_65:
      v43 = v42 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v19 = v41 + 40LL * v43;
      v44 = *(_BYTE **)v19;
      if ( a3 != *(_BYTE **)v19 )
      {
        v45 = 1;
        v46 = 0;
        while ( v44 != (_BYTE *)-4096LL )
        {
          if ( v44 == (_BYTE *)-8192LL && !v46 )
            v46 = v19;
          v43 = v42 & (v45 + v43);
          v19 = v41 + 40LL * v43;
          v44 = *(_BYTE **)v19;
          if ( a3 == *(_BYTE **)v19 )
            goto LABEL_62;
          ++v45;
        }
LABEL_68:
        if ( v46 )
          v19 = v46;
        goto LABEL_62;
      }
      goto LABEL_62;
    }
LABEL_130:
    *(_DWORD *)(a2 + 64) = (2 * (*(_DWORD *)(a2 + 64) >> 1) + 2) | *(_DWORD *)(a2 + 64) & 1;
    BUG();
  }
  sub_D5FDE0(v7, 2 * v10);
  if ( (*(_BYTE *)(a2 + 64) & 1) != 0 )
  {
    v37 = a2 + 72;
    v38 = 7;
  }
  else
  {
    v47 = *(_DWORD *)(a2 + 80);
    v37 = *(_QWORD *)(a2 + 72);
    if ( !v47 )
      goto LABEL_130;
    v38 = v47 - 1;
  }
  v39 = v38 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v19 = v37 + 40LL * v39;
  v40 = *(_BYTE **)v19;
  if ( a3 != *(_BYTE **)v19 )
  {
    v62 = 1;
    v46 = 0;
    while ( v40 != (_BYTE *)-4096LL )
    {
      if ( !v46 && v40 == (_BYTE *)-8192LL )
        v46 = v19;
      v39 = v38 & (v62 + v39);
      v19 = v37 + 40LL * v39;
      v40 = *(_BYTE **)v19;
      if ( a3 == *(_BYTE **)v19 )
        goto LABEL_62;
      ++v62;
    }
    goto LABEL_68;
  }
LABEL_62:
  v18 = *(_DWORD *)(a2 + 64);
LABEL_22:
  *(_DWORD *)(a2 + 64) = (2 * (v18 >> 1) + 2) | v18 & 1;
  if ( *(_QWORD *)v19 != -4096 )
    --*(_DWORD *)(a2 + 68);
  *(_QWORD *)v19 = a3;
  *(_DWORD *)(v19 + 16) = 1;
  *(_QWORD *)(v19 + 8) = 0;
  *(_DWORD *)(v19 + 32) = 1;
  *(_QWORD *)(v19 + 24) = 0;
  v22 = *(_DWORD *)(a2 + 392) + 1;
  v23 = v22 <= dword_4F877C8;
  *(_DWORD *)(a2 + 392) = v22;
  if ( !v23 )
    goto LABEL_25;
  sub_D61DC0((__int64)&v64, (__int64 *)a2, a3);
  v24 = *(_BYTE *)(a2 + 64) & 1;
  if ( v24 )
  {
    v25 = a2 + 72;
    v26 = 7;
  }
  else
  {
    v35 = *(_DWORD *)(a2 + 80);
    v25 = *(_QWORD *)(a2 + 72);
    if ( !v35 )
    {
      v31 = *(_DWORD *)(a2 + 64);
      ++*(_QWORD *)(a2 + 56);
      v32 = 0;
      v33 = (v31 >> 1) + 1;
      goto LABEL_36;
    }
    v26 = v35 - 1;
  }
  v27 = v26 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v28 = v25 + 40LL * v27;
  v29 = *(_QWORD *)v28;
  if ( a3 != *(_BYTE **)v28 )
  {
    v49 = 1;
    v32 = 0;
    while ( v29 != -4096 )
    {
      if ( v29 == -8192 && !v32 )
        v32 = (_QWORD *)v28;
      v27 = v26 & (v49 + v27);
      v28 = v25 + 40LL * v27;
      v29 = *(_QWORD *)v28;
      if ( a3 == *(_BYTE **)v28 )
        goto LABEL_29;
      ++v49;
    }
    v34 = 24;
    v35 = 8;
    if ( !v32 )
      v32 = (_QWORD *)v28;
    v31 = *(_DWORD *)(a2 + 64);
    ++*(_QWORD *)(a2 + 56);
    v33 = (v31 >> 1) + 1;
    if ( v24 )
    {
LABEL_37:
      if ( v34 > 4 * v33 )
      {
        if ( v35 - *(_DWORD *)(a2 + 68) - v33 > v35 >> 3 )
        {
LABEL_39:
          *(_DWORD *)(a2 + 64) = (2 * (v31 >> 1) + 2) | v31 & 1;
          if ( *v32 != -4096 )
            --*(_DWORD *)(a2 + 68);
          *v32 = a3;
          v30 = (__int64)(v32 + 1);
          *(_DWORD *)(v30 + 8) = 1;
          *(_QWORD *)v30 = 0;
          *(_DWORD *)(v30 + 24) = 1;
          *(_QWORD *)(v30 + 16) = 0;
          goto LABEL_42;
        }
        sub_D5FDE0(v7, v35);
        if ( (*(_BYTE *)(a2 + 64) & 1) != 0 )
        {
          v54 = a2 + 72;
          v55 = 7;
          goto LABEL_86;
        }
        v61 = *(_DWORD *)(a2 + 80);
        v54 = *(_QWORD *)(a2 + 72);
        if ( v61 )
        {
          v55 = v61 - 1;
LABEL_86:
          v56 = v55 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
          v32 = (_QWORD *)(v54 + 40LL * v56);
          v57 = (_BYTE *)*v32;
          if ( a3 != (_BYTE *)*v32 )
          {
            v58 = 1;
            v59 = 0;
            while ( v57 != (_BYTE *)-4096LL )
            {
              if ( !v59 && v57 == (_BYTE *)-8192LL )
                v59 = v32;
              v56 = v55 & (v58 + v56);
              v32 = (_QWORD *)(v54 + 40LL * v56);
              v57 = (_BYTE *)*v32;
              if ( a3 == (_BYTE *)*v32 )
                goto LABEL_83;
              ++v58;
            }
LABEL_89:
            if ( v59 )
              v32 = v59;
            goto LABEL_83;
          }
          goto LABEL_83;
        }
LABEL_131:
        *(_DWORD *)(a2 + 64) = (2 * (*(_DWORD *)(a2 + 64) >> 1) + 2) | *(_DWORD *)(a2 + 64) & 1;
        BUG();
      }
      sub_D5FDE0(v7, 2 * v35);
      if ( (*(_BYTE *)(a2 + 64) & 1) != 0 )
      {
        v50 = a2 + 72;
        v51 = 7;
      }
      else
      {
        v60 = *(_DWORD *)(a2 + 80);
        v50 = *(_QWORD *)(a2 + 72);
        if ( !v60 )
          goto LABEL_131;
        v51 = v60 - 1;
      }
      v52 = v51 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v32 = (_QWORD *)(v50 + 40LL * v52);
      v53 = *v32;
      if ( a3 != (_BYTE *)*v32 )
      {
        v63 = 1;
        v59 = 0;
        while ( v53 != -4096 )
        {
          if ( v53 == -8192 && !v59 )
            v59 = v32;
          v52 = v51 & (v63 + v52);
          v32 = (_QWORD *)(v50 + 40LL * v52);
          v53 = *v32;
          if ( a3 == (_BYTE *)*v32 )
            goto LABEL_83;
          ++v63;
        }
        goto LABEL_89;
      }
LABEL_83:
      v31 = *(_DWORD *)(a2 + 64);
      goto LABEL_39;
    }
    v35 = *(_DWORD *)(a2 + 80);
LABEL_36:
    v34 = 3 * v35;
    goto LABEL_37;
  }
LABEL_29:
  v30 = v28 + 8;
  if ( *(_DWORD *)(v28 + 16) > 0x40u )
  {
LABEL_30:
    sub_C43990(v30, (__int64)&v64);
    goto LABEL_31;
  }
LABEL_42:
  if ( v65 > 0x40 )
    goto LABEL_30;
  *(_QWORD *)v30 = v64;
  *(_DWORD *)(v30 + 8) = v65;
LABEL_31:
  if ( *(_DWORD *)(v30 + 24) <= 0x40u && v67 <= 0x40 )
  {
    *(_QWORD *)(v30 + 16) = v66;
    *(_DWORD *)(v30 + 24) = v67;
  }
  else
  {
    sub_C43990(v30 + 16, (__int64)&v66);
  }
  *(_DWORD *)(a1 + 8) = v65;
  *(_QWORD *)a1 = v64;
  *(_DWORD *)(a1 + 24) = v67;
  *(_QWORD *)(a1 + 16) = v66;
  return a1;
}
