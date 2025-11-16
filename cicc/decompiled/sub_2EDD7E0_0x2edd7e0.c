// Function: sub_2EDD7E0
// Address: 0x2edd7e0
//
char __fastcall sub_2EDD7E0(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rax
  __int64 v8; // rdi
  __int64 (__fastcall *v9)(__int64, __int64); // rax
  int v10; // eax
  _BYTE *v11; // r13
  _BYTE *v12; // r12
  _BYTE *v13; // rbx
  int v14; // r13d
  unsigned int v15; // esi
  unsigned __int64 v16; // r10
  __int64 v17; // r8
  int v18; // r11d
  int v19; // r15d
  unsigned int j; // ecx
  int v21; // edx
  int v22; // edx
  int v23; // edi
  __int64 v24; // rcx
  int v25; // r8d
  unsigned __int64 v26; // rsi
  unsigned int i; // edx
  int v28; // r9d
  int v29; // ecx
  _BYTE *v30; // r13
  __int64 v31; // rbx
  _BYTE *v32; // r12
  _BYTE *v33; // r13
  _BYTE *v34; // rbx
  unsigned int v35; // r13d
  _BYTE *v36; // r13
  __int64 v37; // rdi
  char v38; // r13
  char v39; // r12
  __int64 v40; // rbx
  __int64 v41; // r14
  int v42; // ecx
  int v43; // ecx
  int v44; // edi
  __int64 v45; // rsi
  int v46; // eax
  unsigned int v47; // ecx
  unsigned int v48; // edx
  int v49; // ecx
  int v50; // eax
  int v51; // ecx
  __int64 v52; // rdi
  int v53; // esi
  unsigned __int64 v54; // rdx
  unsigned int k; // r15d
  int v56; // r8d
  unsigned int v57; // r15d
  char v59; // [rsp+8h] [rbp-88h]
  __int64 v60; // [rsp+10h] [rbp-80h]
  unsigned __int128 v62; // [rsp+20h] [rbp-70h]
  unsigned __int128 v63; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int128 v64; // [rsp+40h] [rbp-50h] BYREF
  char v65; // [rsp+50h] [rbp-40h]

  v62 = __PAIR128__(a4, a3);
  v63 = __PAIR128__(a4, a3);
  v59 = a5;
  LOBYTE(v7) = sub_2ED9D80((__int64)&v64, a1 + 736, (__m128i *)&v63, a4, a5, a6);
  if ( !v65 )
    goto LABEL_53;
  if ( *(_WORD *)(a2 + 68) != 20 )
  {
    v8 = *(_QWORD *)(a1 + 8);
    v9 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v8 + 176LL);
    if ( v9 == sub_2E4F5F0 )
    {
      v10 = *(_DWORD *)(a2 + 44);
      if ( (v10 & 4) == 0 && (v10 & 8) != 0 )
        LOBYTE(v7) = sub_2E88A90(a2, 0x40000000, 2);
      else
        v7 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 30) & 1LL;
    }
    else
    {
      LOBYTE(v7) = v9(v8, a2);
    }
    if ( !(_BYTE)v7 )
      goto LABEL_53;
  }
  v11 = *(_BYTE **)(a2 + 32);
  v12 = &v11[40 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF)];
  if ( v11 == v12 )
    goto LABEL_40;
  while ( 1 )
  {
    v13 = v11;
    if ( sub_2DADC00(v11) )
      break;
    v11 += 40;
    if ( v12 == v11 )
      goto LABEL_40;
  }
  if ( v12 == v11 )
  {
LABEL_40:
    if ( sub_2E322C0(v62, *((__int64 *)&v62 + 1)) )
    {
      sub_F02DB0(&v64, qword_5022148, 0x64u);
      LODWORD(v7) = sub_2E441D0(*(_QWORD *)(a1 + 72), v62, *((__int64 *)&v62 + 1));
      if ( (unsigned int)v64 >= (unsigned int)v7 )
      {
LABEL_72:
        v39 = v59;
        v40 = 0;
LABEL_55:
        if ( (_BYTE)qword_5022308 == 1 && (_QWORD)v62 != *((_QWORD *)&v62 + 1) )
        {
          LOBYTE(v7) = sub_2E322C0(v62, *((__int64 *)&v62 + 1));
          if ( (_BYTE)v7 )
          {
            LOBYTE(v7) = sub_2ED13E0(a1, v62, *((__int64 *)&v62 + 1), v39);
            if ( (_BYTE)v7 )
            {
              v41 = a1 + 960;
              v64 = v62;
              LOBYTE(v7) = sub_2EDD470(v41, (const __m128i *)&v64);
              if ( v40 )
              {
                *(_QWORD *)&v64 = v40;
                *((_QWORD *)&v64 + 1) = *((_QWORD *)&v62 + 1);
                LOBYTE(v7) = sub_2EDD470(v41, (const __m128i *)&v64);
              }
            }
          }
        }
        return v7;
      }
    }
    v31 = *(_QWORD *)(a2 + 32);
    v32 = (_BYTE *)(v31 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF));
    v33 = (_BYTE *)(v31 + 40LL * (unsigned int)sub_2E88FE0(a2));
    if ( v32 != v33 )
    {
      while ( 1 )
      {
        v34 = v33;
        if ( (unsigned __int8)sub_2E2FA70(v33) )
          break;
        v33 += 40;
        if ( v32 == v33 )
          goto LABEL_51;
      }
      while ( v32 != v34 )
      {
        v35 = *((_DWORD *)v34 + 2);
        if ( v35 > 0x3FFFFFFF )
        {
          if ( (unsigned __int8)sub_2EBEF70(*(_QWORD *)(a1 + 24), v35) )
          {
            v7 = sub_2EBEE10(*(_QWORD *)(a1 + 24), v35);
            if ( *(_QWORD *)(v7 + 24) == *(_QWORD *)(a2 + 24) )
              goto LABEL_72;
          }
        }
        v36 = v34 + 40;
        if ( v34 + 40 == v32 )
          break;
        while ( 1 )
        {
          v34 = v36;
          if ( (unsigned __int8)sub_2E2FA70(v36) )
            break;
          v36 += 40;
          if ( v32 == v36 )
            goto LABEL_51;
        }
      }
    }
LABEL_51:
    v37 = *(_QWORD *)(a1 + 8);
    v7 = *(_QWORD *)(*(_QWORD *)v37 + 48LL);
    if ( (__int64 (*)())v7 == sub_2ED11B0 )
      return v7;
    LOBYTE(v7) = ((__int64 (__fastcall *)(__int64, __int64))v7)(v37, a2);
    if ( !(_BYTE)v7 )
      return v7;
LABEL_53:
    v38 = v59;
LABEL_54:
    v39 = v38;
    v40 = 0;
    goto LABEL_55;
  }
  v60 = (DWORD2(v62) >> 9) ^ (DWORD2(v62) >> 4);
  while ( 1 )
  {
    v14 = *((_DWORD *)v13 + 2);
    if ( v14 )
      break;
LABEL_35:
    v30 = v13 + 40;
    if ( v13 + 40 != v12 )
    {
      while ( 1 )
      {
        v13 = v30;
        if ( sub_2DADC00(v30) )
          break;
        v30 += 40;
        if ( v12 == v30 )
          goto LABEL_40;
      }
      if ( v12 != v30 )
        continue;
    }
    goto LABEL_40;
  }
  if ( v14 < 0 )
    v14 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 16) + 56LL))(
            *(_QWORD *)(a1 + 16),
            (unsigned int)v14,
            *(_QWORD *)(a1 + 24));
  v15 = *(_DWORD *)(a1 + 952);
  if ( !v15 )
  {
    ++*(_QWORD *)(a1 + 928);
LABEL_27:
    sub_2ED6570(a1 + 928, 2 * v15);
    v22 = *(_DWORD *)(a1 + 952);
    if ( !v22 )
      goto LABEL_115;
    v23 = v22 - 1;
    v25 = 1;
    v26 = 0;
    for ( i = (v22 - 1)
            & (((0xBF58476D1CE4E5B9LL * (v60 | ((unsigned __int64)(unsigned int)(37 * v14) << 32))) >> 31)
             ^ (484763065 * v60)); ; i = v23 & v48 )
    {
      v24 = *(_QWORD *)(a1 + 936);
      v7 = v24 + 24LL * i;
      v28 = *(_DWORD *)v7;
      if ( v14 == *(_DWORD *)v7 && *((_QWORD *)&v62 + 1) == *(_QWORD *)(v7 + 8) )
        break;
      if ( v28 == -1 )
      {
        if ( *(_QWORD *)(v7 + 8) == -4096 )
        {
          v29 = *(_DWORD *)(a1 + 944) + 1;
          if ( v26 )
            v7 = v26;
          goto LABEL_32;
        }
      }
      else if ( v28 == -2 && *(_QWORD *)(v7 + 8) == -8192 && !v26 )
      {
        v26 = v24 + 24LL * i;
      }
      v48 = v25 + i;
      ++v25;
    }
LABEL_31:
    v29 = *(_DWORD *)(a1 + 944) + 1;
LABEL_32:
    *(_DWORD *)(a1 + 944) = v29;
    if ( *(_DWORD *)v7 != -1 || *(_QWORD *)(v7 + 8) != -4096 )
      --*(_DWORD *)(a1 + 948);
    *(_DWORD *)v7 = v14;
    *(_QWORD *)(v7 + 8) = *((_QWORD *)&v62 + 1);
    *(_QWORD *)(v7 + 16) = v62;
    goto LABEL_35;
  }
  v16 = 0;
  v17 = *(_QWORD *)(a1 + 936);
  v18 = 1;
  v19 = ((0xBF58476D1CE4E5B9LL * (v60 | ((unsigned __int64)(unsigned int)(37 * v14) << 32))) >> 31) ^ (484763065 * v60);
  for ( j = v19 & (v15 - 1); ; j = (v15 - 1) & v47 )
  {
    v7 = v17 + 24LL * j;
    v21 = *(_DWORD *)v7;
    if ( v14 == *(_DWORD *)v7 && *((_QWORD *)&v62 + 1) == *(_QWORD *)(v7 + 8) )
      break;
    if ( v21 == -1 )
    {
      if ( *(_QWORD *)(v7 + 8) == -4096 )
      {
        v49 = *(_DWORD *)(a1 + 944);
        if ( v16 )
          v7 = v16;
        ++*(_QWORD *)(a1 + 928);
        v29 = v49 + 1;
        if ( 4 * v29 >= 3 * v15 )
          goto LABEL_27;
        if ( v15 - *(_DWORD *)(a1 + 948) - v29 <= v15 >> 3 )
        {
          sub_2ED6570(a1 + 928, v15);
          v50 = *(_DWORD *)(a1 + 952);
          if ( v50 )
          {
            v51 = v50 - 1;
            v53 = 1;
            v54 = 0;
            for ( k = (v50 - 1) & v19; ; k = v51 & v57 )
            {
              v52 = *(_QWORD *)(a1 + 936);
              v7 = v52 + 24LL * k;
              v56 = *(_DWORD *)v7;
              if ( v14 == *(_DWORD *)v7 && *((_QWORD *)&v62 + 1) == *(_QWORD *)(v7 + 8) )
                break;
              if ( v56 == -1 )
              {
                if ( *(_QWORD *)(v7 + 8) == -4096 )
                {
                  v29 = *(_DWORD *)(a1 + 944) + 1;
                  if ( v54 )
                    v7 = v54;
                  goto LABEL_32;
                }
              }
              else if ( v56 == -2 && *(_QWORD *)(v7 + 8) == -8192 && !v54 )
              {
                v54 = v52 + 24LL * k;
              }
              v57 = v53 + k;
              ++v53;
            }
            goto LABEL_31;
          }
LABEL_115:
          ++*(_DWORD *)(a1 + 944);
          BUG();
        }
        goto LABEL_32;
      }
    }
    else if ( v21 == -2 && *(_QWORD *)(v7 + 8) == -8192 && !v16 )
    {
      v16 = v17 + 24LL * j;
    }
    v47 = v18 + j;
    ++v18;
  }
  v38 = v59;
  v40 = *(_QWORD *)(v7 + 16);
  v39 = v59;
  if ( !v40 )
    goto LABEL_54;
  v42 = *(_DWORD *)(a1 + 984);
  if ( v42 )
  {
    v43 = v42 - 1;
    v44 = 1;
    for ( LODWORD(v7) = v43
                      & (((0xBF58476D1CE4E5B9LL
                         * (v60 | ((unsigned __int64)(((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4)) << 32))) >> 31)
                       ^ (484763065 * v60)); ; LODWORD(v7) = v43 & v46 )
    {
      v45 = *(_QWORD *)(a1 + 968) + 16LL * (unsigned int)v7;
      if ( __PAIR128__(*((unsigned __int64 *)&v62 + 1), v40) == *(_OWORD *)v45 )
        break;
      if ( *(_QWORD *)v45 == -4096 && *(_QWORD *)(v45 + 8) == -4096 )
        goto LABEL_78;
      v46 = v44 + v7;
      ++v44;
    }
    goto LABEL_55;
  }
LABEL_78:
  if ( (_BYTE)qword_5022308 == 1 && *((_QWORD *)&v62 + 1) != v40 )
  {
    LOBYTE(v7) = sub_2E322C0(v40, *((__int64 *)&v62 + 1));
    if ( (_BYTE)v7 )
    {
      LOBYTE(v7) = sub_2ED13E0(a1, v40, *((__int64 *)&v62 + 1), v59);
      if ( (_BYTE)v7 )
        goto LABEL_55;
    }
  }
  return v7;
}
