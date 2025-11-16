// Function: sub_27D2430
// Address: 0x27d2430
//
__int64 __fastcall sub_27D2430(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 i; // r14
  char v8; // al
  __int64 *v9; // r8
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // rdx
  __int64 *v14; // rdx
  __int64 v15; // rax
  unsigned __int64 v16; // rbx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _QWORD *v20; // rdi
  __int64 *v21; // rbx
  __int64 v22; // rsi
  unsigned __int64 *v24; // rax
  __int64 v25; // rdx
  bool v26; // zf
  __int64 v27; // rsi
  __int64 v28; // rcx
  int v29; // eax
  __int64 v30; // rax
  __int64 *v31; // rax
  __int64 v32; // rax
  unsigned int v33; // edx
  __int64 v34; // rsi
  __int64 v35; // rdi
  char v36; // al
  _QWORD *v37; // rdi
  __int64 v38; // rdi
  unsigned int *v39; // rbx
  unsigned int *v40; // r12
  __int64 v41; // rsi
  __int64 v42; // rdx
  __int64 *v43; // rax
  __int64 v44; // rdi
  __int64 *v45; // r12
  __int64 *v46; // rbx
  __int64 v47; // rcx
  int v48; // eax
  __int64 v49; // rax
  int v50; // eax
  __int64 v51; // rcx
  int v52; // eax
  __int64 v53; // rsi
  __int64 v54; // rax
  __int64 v55; // [rsp+10h] [rbp-F0h]
  __int64 v56; // [rsp+20h] [rbp-E0h]
  __int64 *v57; // [rsp+30h] [rbp-D0h]
  _BYTE **v58; // [rsp+40h] [rbp-C0h] BYREF
  __int64 *v59; // [rsp+48h] [rbp-B8h]
  __int64 v60; // [rsp+50h] [rbp-B0h]
  __int64 v61; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v62; // [rsp+68h] [rbp-98h]
  __int64 v63; // [rsp+70h] [rbp-90h]
  __int64 v64; // [rsp+78h] [rbp-88h]
  __int64 v65; // [rsp+80h] [rbp-80h] BYREF
  __int64 v66; // [rsp+88h] [rbp-78h]
  _QWORD v67[2]; // [rsp+90h] [rbp-70h] BYREF
  _BYTE *v68; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v69; // [rsp+A8h] [rbp-58h]
  _BYTE v70[80]; // [rsp+B0h] [rbp-50h] BYREF

  v3 = a3 + 72;
  v5 = a2;
  v6 = *(_QWORD *)(a3 + 80);
  v58 = &v68;
  v68 = v70;
  v69 = 0x400000000LL;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v59 = &v61;
  v60 = a2;
  if ( a3 + 72 == v6 )
  {
    i = 0;
  }
  else
  {
    if ( !v6 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v6 + 32);
      if ( i != v6 + 24 )
        break;
      v6 = *(_QWORD *)(v6 + 8);
      if ( v3 == v6 )
        break;
      if ( !v6 )
        BUG();
    }
  }
  while ( v6 != v3 )
  {
    if ( !i )
      BUG();
    v8 = *(_BYTE *)(i - 24);
    v9 = (__int64 *)(i - 24);
    if ( v8 == 63 )
    {
      sub_27D1F70(v60, *(_QWORD *)(i - 32LL * (*(_DWORD *)(i - 20) & 0x7FFFFFF) - 24), (__int64)v58, (__int64)v59);
      goto LABEL_11;
    }
    if ( v8 == 61 || v8 == 62 )
    {
LABEL_19:
      sub_27D1F70(v60, *(_QWORD *)(i - 56), (__int64)v58, (__int64)v59);
      goto LABEL_11;
    }
    if ( v8 == 66 )
    {
      sub_27D1F70(v60, *(_QWORD *)(i - 88), (__int64)v58, (__int64)v59);
      goto LABEL_11;
    }
    if ( v8 == 65 )
    {
      sub_27D1F70(v60, *(_QWORD *)(i - 120), (__int64)v58, (__int64)v59);
      goto LABEL_11;
    }
    if ( v8 == 85 )
    {
      v32 = *(_QWORD *)(i - 56);
      if ( !v32 )
        goto LABEL_11;
      if ( !*(_BYTE *)v32
        && *(_QWORD *)(v32 + 24) == *(_QWORD *)(i + 56)
        && (*(_BYTE *)(v32 + 33) & 0x20) != 0
        && (unsigned int)(*(_DWORD *)(v32 + 36) - 238) <= 7
        && ((1LL << (*(_BYTE *)(v32 + 36) + 18)) & 0xAD) != 0 )
      {
        sub_27D1F70(v60, v9[-4 * (*(_DWORD *)(i - 20) & 0x7FFFFFF)], (__int64)v58, (__int64)v59);
        v49 = *(_QWORD *)(i - 56);
        if ( !v49 || *(_BYTE *)v49 || *(_QWORD *)(v49 + 24) != *(_QWORD *)(i + 56) )
          BUG();
        v50 = *(_DWORD *)(v49 + 36);
        if ( v50 == 238 || (unsigned int)(v50 - 240) <= 1 )
          sub_27D1F70(
            v60,
            *(_QWORD *)(i - 24 + 32 * (1LL - (*(_DWORD *)(i - 20) & 0x7FFFFFF))),
            (__int64)v58,
            (__int64)v59);
        goto LABEL_11;
      }
      if ( *(_BYTE *)v32 || *(_QWORD *)(v32 + 24) != *(_QWORD *)(i + 56) || (*(_BYTE *)(v32 + 33) & 0x20) == 0 )
        goto LABEL_11;
      v33 = *(_DWORD *)(v32 + 36);
      if ( v33 > 0xE6 )
      {
        if ( v33 == 286 || v33 == 299 || v33 == 282 )
        {
LABEL_82:
          v34 = *(_QWORD *)(i - 32LL * (*(_DWORD *)(i - 20) & 0x7FFFFFF) - 24);
LABEL_83:
          sub_27D1F70(v5, v34, (__int64)&v68, (__int64)&v61);
          goto LABEL_11;
        }
        goto LABEL_87;
      }
      if ( v33 > 0xE4 )
      {
        sub_27D1F70(
          v5,
          *(_QWORD *)(i + 32 * (1LL - (*(_DWORD *)(i - 20) & 0x7FFFFFF)) - 24),
          (__int64)&v68,
          (__int64)&v61);
        goto LABEL_11;
      }
      if ( v33 == 206 )
      {
        v34 = *(_QWORD *)(i - 32LL * (*(_DWORD *)(i - 20) & 0x7FFFFFF) - 24);
        v51 = *(_QWORD *)(v34 + 8);
        v52 = *(unsigned __int8 *)(v51 + 8);
        if ( (unsigned int)(v52 - 17) <= 1 )
          LOBYTE(v52) = *(_BYTE *)(**(_QWORD **)(v51 + 16) + 8LL);
        if ( (_BYTE)v52 == 14 )
          goto LABEL_83;
      }
      else
      {
        if ( v33 > 0xCE )
        {
          if ( v33 - 227 <= 1 )
            goto LABEL_82;
LABEL_87:
          v35 = *(_QWORD *)(v5 + 24);
          v65 = (__int64)v67;
          v66 = 0x200000000LL;
          v36 = sub_DF9880(v35);
          v37 = (_QWORD *)v65;
          if ( v36 && v65 != v65 + 4LL * (unsigned int)v66 )
          {
            v38 = v5;
            v55 = a1;
            v39 = (unsigned int *)v65;
            v40 = (unsigned int *)(v65 + 4LL * (unsigned int)v66);
            do
            {
              v41 = *v39++;
              sub_27D1F70(
                v38,
                *(_QWORD *)(i - 24 + 32 * (v41 - (*(_DWORD *)(i - 20) & 0x7FFFFFF))),
                (__int64)&v68,
                (__int64)&v61);
            }
            while ( v40 != v39 );
            v5 = v38;
            v37 = (_QWORD *)v65;
            a1 = v55;
          }
          if ( v37 != v67 )
            _libc_free((unsigned __int64)v37);
          goto LABEL_11;
        }
        if ( v33 != 171 )
          goto LABEL_87;
        v42 = 32LL * (*(_DWORD *)(i - 20) & 0x7FFFFFF);
        if ( (*(_BYTE *)(i - 17) & 0x40) != 0 )
        {
          v43 = *(__int64 **)(i - 32);
          v9 = &v43[(unsigned __int64)v42 / 8];
        }
        else
        {
          v43 = &v9[v42 / 0xFFFFFFFFFFFFFFF8LL];
        }
        if ( v43 != v9 )
        {
          v56 = a1;
          v44 = v5;
          v45 = v9;
          v46 = v43;
          do
          {
            v47 = *(_QWORD *)(*v46 + 8);
            v48 = *(unsigned __int8 *)(v47 + 8);
            if ( (unsigned int)(v48 - 17) <= 1 )
              LOBYTE(v48) = *(_BYTE *)(**(_QWORD **)(v47 + 16) + 8LL);
            if ( (_BYTE)v48 == 14 )
              sub_27D1F70(v44, *v46, (__int64)&v68, (__int64)&v61);
            v46 += 4;
          }
          while ( v45 != v46 );
          a1 = v56;
          v5 = v44;
        }
      }
    }
    else if ( v8 == 82 )
    {
      v27 = *(_QWORD *)(i - 88);
      v28 = *(_QWORD *)(v27 + 8);
      v29 = *(unsigned __int8 *)(v28 + 8);
      if ( (unsigned int)(v29 - 17) <= 1 )
        LOBYTE(v29) = *(_BYTE *)(**(_QWORD **)(v28 + 16) + 8LL);
      if ( (_BYTE)v29 == 14 )
      {
        sub_27D2410((__int64 *)&v58, v27);
        sub_27D2410((__int64 *)&v58, *(_QWORD *)(i - 56));
      }
    }
    else
    {
      if ( v8 == 79 )
        goto LABEL_19;
      if ( v8 == 77 )
      {
        if ( sub_27CE300((unsigned __int8 *)(i - 24), *(_QWORD *)(v5 + 32), *(_QWORD *)(v5 + 24)) )
        {
          v30 = *(_QWORD *)(i - 56);
          if ( (*(_BYTE *)(v30 + 7) & 0x40) != 0 )
            v31 = *(__int64 **)(v30 - 8);
          else
            v31 = (__int64 *)(v30 - 32LL * (*(_DWORD *)(v30 + 4) & 0x7FFFFFF));
          sub_27D2410((__int64 *)&v58, *v31);
        }
      }
      else
      {
        if ( v8 == 30 )
        {
          if ( (*(_DWORD *)(i - 20) & 0x7FFFFFF) == 0 )
            goto LABEL_11;
          v53 = v9[-4 * (*(_DWORD *)(i - 20) & 0x7FFFFFF)];
          if ( !v53 )
            goto LABEL_11;
          v54 = *(_QWORD *)(v53 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v54 + 8) - 17 <= 1 )
            v54 = **(_QWORD **)(v54 + 16);
        }
        else
        {
          if ( v8 != 94 )
            goto LABEL_11;
          v53 = *(_QWORD *)(i - 56);
          v54 = *(_QWORD *)(v53 + 8);
        }
        if ( *(_BYTE *)(v54 + 8) == 14 )
          sub_27D2410((__int64 *)&v58, v53);
      }
    }
LABEL_11:
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v6 + 32) )
    {
      v10 = v6 - 24;
      if ( !v6 )
        v10 = 0;
      if ( i != v10 + 48 )
        break;
      v6 = *(_QWORD *)(v6 + 8);
      if ( v3 == v6 )
        break;
      if ( !v6 )
        BUG();
    }
  }
  v11 = (unsigned int)v69;
  v12 = v5;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  if ( (_DWORD)v11 )
  {
    while ( 1 )
    {
      v14 = (__int64 *)&v68[8 * v11 - 8];
      v15 = *v14;
      v16 = *v14 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*v14 & 4) != 0 )
        break;
      *v14 = v15 | 4;
      v19 = (unsigned int)sub_DF9B70(*(_QWORD *)(v12 + 24));
      v11 = (unsigned int)v69;
      if ( (_DWORD)v19 == -1 )
      {
        sub_27CE620(&v65, (unsigned __int8 *)v16, *(_QWORD *)(v12 + 24), v17, v18, v19);
        v20 = (_QWORD *)v65;
        v57 = (__int64 *)(v65 + 8LL * (unsigned int)v66);
        if ( v57 != (__int64 *)v65 )
        {
          v21 = (__int64 *)v65;
          do
          {
            v22 = *v21++;
            sub_27D1F70(v12, v22, (__int64)&v68, (__int64)&v61);
          }
          while ( v57 != v21 );
          v20 = (_QWORD *)v65;
        }
        if ( v20 != v67 )
          _libc_free((unsigned __int64)v20);
        v11 = (unsigned int)v69;
        if ( !(_DWORD)v69 )
          goto LABEL_36;
      }
      else
      {
LABEL_26:
        if ( !(_DWORD)v11 )
          goto LABEL_36;
      }
    }
    v13 = *(_QWORD *)(v16 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17 <= 1 )
      v13 = **(_QWORD **)(v13 + 16);
    if ( *(_DWORD *)(v12 + 40) == *(_DWORD *)(v13 + 8) >> 8 )
    {
      v65 = 6;
      v66 = 0;
      v67[0] = v16;
      if ( (v15 & 0xFFFFFFFFFFFFEFF8LL) != 0xFFFFFFFFFFFFE000LL && v16 )
        sub_BD73F0((__int64)&v65);
      v24 = *(unsigned __int64 **)(a1 + 8);
      if ( v24 == *(unsigned __int64 **)(a1 + 16) )
      {
        sub_F465C0((unsigned __int64 **)a1, *(char **)(a1 + 8), &v65);
      }
      else
      {
        if ( v24 )
        {
          *v24 = 6;
          v24[1] = 0;
          v25 = v67[0];
          v26 = v67[0] == -4096;
          v24[2] = v67[0];
          if ( v25 != 0 && !v26 && v25 != -8192 )
            sub_BD6050(v24, v65 & 0xFFFFFFFFFFFFFFF8LL);
          v24 = *(unsigned __int64 **)(a1 + 8);
        }
        *(_QWORD *)(a1 + 8) = v24 + 3;
      }
      if ( v67[0] != 0 && v67[0] != -4096 && v67[0] != -8192 )
        sub_BD60C0(&v65);
    }
    v11 = (unsigned int)(v69 - 1);
    LODWORD(v69) = v69 - 1;
    goto LABEL_26;
  }
LABEL_36:
  sub_C7D6A0(v62, 8LL * (unsigned int)v64, 8);
  if ( v68 != v70 )
    _libc_free((unsigned __int64)v68);
  return a1;
}
