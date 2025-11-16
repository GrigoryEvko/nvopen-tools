// Function: sub_21DB6B0
// Address: 0x21db6b0
//
__int64 __fastcall sub_21DB6B0(__int64 a1, __int64 a2)
{
  __int64 *v2; // r13
  __int64 v3; // rax
  __int64 (*v4)(); // rdx
  __int64 (*v5)(); // rax
  __int64 v6; // r14
  _DWORD *v7; // r13
  __int64 *v8; // rdi
  __int64 v9; // rax
  __int64 (*v10)(void); // rdx
  __int64 (*v11)(void); // rax
  int v12; // edi
  __int64 v13; // r12
  int v14; // eax
  __int64 v15; // rcx
  __int64 v16; // r9
  __int64 v17; // rdx
  signed __int64 v18; // rbx
  signed __int64 v19; // rax
  __int64 v20; // rax
  signed __int64 v21; // rax
  unsigned int v22; // r15d
  int v23; // r8d
  unsigned int v24; // r11d
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 (*v31)(); // rax
  __int64 v32; // rdx
  __int64 (__fastcall *v33)(__int64); // rax
  char v34; // al
  __int64 v35; // rdx
  unsigned int v36; // eax
  int v37; // r11d
  __int64 i; // r12
  __int64 v39; // r13
  __int64 j; // rbx
  __int64 v41; // rax
  unsigned int v42; // r15d
  __int64 v43; // rdx
  __int64 v44; // rbx
  unsigned __int64 v45; // rax
  __int64 *v46; // rdi
  __int64 v47; // rdx
  __int16 v48; // ax
  __int64 v49; // rax
  _QWORD *v51; // rax
  __int64 v52; // rax
  __int64 v53; // [rsp+8h] [rbp-78h]
  __int64 v54; // [rsp+10h] [rbp-70h]
  __int64 v55; // [rsp+18h] [rbp-68h]
  __int64 v56; // [rsp+20h] [rbp-60h]
  __int64 v57; // [rsp+30h] [rbp-50h]
  unsigned __int8 v58; // [rsp+30h] [rbp-50h]
  __int64 v59; // [rsp+38h] [rbp-48h]
  int v60; // [rsp+38h] [rbp-48h]
  int v61[13]; // [rsp+4Ch] [rbp-34h] BYREF

  v2 = *(__int64 **)(a2 + 16);
  v56 = a2;
  v53 = 0;
  v3 = *v2;
  v4 = *(__int64 (**)())(*v2 + 48);
  if ( v4 != sub_1D90020 )
  {
    v53 = ((__int64 (__fastcall *)(__int64 *))v4)(v2);
    v3 = *v2;
  }
  v5 = *(__int64 (**)())(v3 + 112);
  v6 = 0;
  if ( v5 != sub_1D00B10 )
    v6 = ((__int64 (__fastcall *)(__int64 *))v5)(v2);
  v7 = 0;
  v8 = *(__int64 **)(a2 + 16);
  v9 = *v8;
  v10 = *(__int64 (**)(void))(*v8 + 48);
  if ( v10 != sub_1D90020 )
  {
    v7 = (_DWORD *)v10();
    v9 = **(_QWORD **)(a2 + 16);
  }
  v11 = *(__int64 (**)(void))(v9 + 112);
  v59 = 0;
  if ( v11 != sub_1D00B10 )
    v59 = v11();
  v12 = v7[2];
  v13 = *(_QWORD *)(a2 + 56);
  v14 = v7[5];
  v15 = *(unsigned int *)(v13 + 32);
  v16 = *(_QWORD *)(v13 + 8);
  v17 = (unsigned int)-v14;
  if ( v12 == 1 )
    v14 = -v14;
  v18 = v14;
  v57 = v14;
  if ( (_DWORD)v15 )
  {
    LODWORD(v17) = 0;
    do
    {
      while ( 1 )
      {
        a2 = v16 + 40LL * (unsigned int)v17;
        v20 = *(_QWORD *)a2;
        if ( v12 != 1 )
          break;
        v21 = -v20;
        if ( v18 < v21 )
          v18 = v21;
        v17 = (unsigned int)(v17 + 1);
        if ( (_DWORD)v15 == (_DWORD)v17 )
          goto LABEL_20;
      }
      v19 = *(_QWORD *)(a2 + 8) + v20;
      if ( v18 < v19 )
        v18 = v19;
      v17 = (unsigned int)(v17 + 1);
    }
    while ( (_DWORD)v15 != (_DWORD)v17 );
  }
  else
  {
    v18 = v14;
  }
LABEL_20:
  v22 = *(_DWORD *)(v13 + 60);
  if ( *(_BYTE *)(v13 + 652) )
  {
    a2 = *(unsigned int *)(v13 + 648);
    v23 = *(_DWORD *)(v13 + 120);
    v24 = *(_DWORD *)(v13 + 648);
    v17 = (a2 + v18 - 1) % a2;
    v25 = a2 * ((a2 + v18 - 1) / a2);
    if ( v23 )
    {
      a2 = -v25;
      if ( v12 != 1 )
        a2 = v25;
      LODWORD(v17) = 0;
      while ( 1 )
      {
        v26 = (int)v17;
        v17 = (unsigned int)(v17 + 1);
        *(_QWORD *)(v16 + 40LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(v13 + 112) + 16 * v26) + v15)) = a2 + *(_QWORD *)(*(_QWORD *)(v13 + 112) + 16 * v26 + 8);
        if ( v23 == (_DWORD)v17 )
          break;
        LODWORD(v15) = *(_DWORD *)(v13 + 32);
        v16 = *(_QWORD *)(v13 + 8);
      }
      v16 = *(_QWORD *)(v13 + 8);
      v15 = *(unsigned int *)(v13 + 32);
    }
    v18 = *(_QWORD *)(v13 + 640) + v25;
    if ( v22 < v24 )
      v22 = v24;
  }
  v27 = -858993459 * (unsigned int)((*(_QWORD *)(v13 + 16) - v16) >> 3) - (unsigned int)v15;
  if ( (_DWORD)v27 )
  {
    LODWORD(a2) = 0;
    while ( 1 )
    {
      v28 = v16 + 40LL * (unsigned int)(a2 + v15);
      if ( *(_BYTE *)(v28 + 32) && *(_BYTE *)(v13 + 652) )
        goto LABEL_33;
      v29 = *(_QWORD *)(v28 + 8);
      if ( v29 == -1 )
        goto LABEL_33;
      v15 = *(unsigned int *)(v28 + 16);
      if ( v22 < *(_DWORD *)(v28 + 16) )
        v22 = *(_DWORD *)(v28 + 16);
      if ( v12 == 1 )
        break;
      *(_QWORD *)v28 = v15 * ((v15 + v18 - 1) / v15);
      v17 = v15 * ((v15 + v18 - 1) / v15);
      v15 = *(_QWORD *)(v13 + 8);
      v30 = (unsigned int)(*(_DWORD *)(v13 + 32) + a2);
      a2 = (unsigned int)(a2 + 1);
      v18 = v17 + *(_QWORD *)(v15 + 40 * v30 + 8);
      if ( (_DWORD)v27 == (_DWORD)a2 )
        goto LABEL_42;
LABEL_34:
      v15 = *(unsigned int *)(v13 + 32);
      v16 = *(_QWORD *)(v13 + 8);
    }
    v17 = (v15 + v18 + v29 - 1) % *(unsigned int *)(v28 + 16);
    v18 = *(unsigned int *)(v28 + 16) * ((v15 + v18 + v29 - 1) / *(unsigned int *)(v28 + 16));
    *(_QWORD *)v28 = -v18;
LABEL_33:
    a2 = (unsigned int)(a2 + 1);
    if ( (_DWORD)v27 == (_DWORD)a2 )
      goto LABEL_42;
    goto LABEL_34;
  }
LABEL_42:
  v31 = *(__int64 (**)())(*(_QWORD *)v7 + 48LL);
  if ( v31 == sub_1EAD5E0
    || !((unsigned __int8 (__fastcall *)(_DWORD *, __int64, __int64, __int64, __int64))v31)(v7, a2, v17, v15, v27) )
  {
    if ( *(_BYTE *)(v13 + 64) )
    {
      v32 = *(_QWORD *)v7;
      v33 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 152LL);
      if ( v33 == sub_1E08720 )
        v34 = (*(__int64 (__fastcall **)(_DWORD *, __int64, __int64, __int64, __int64))(v32 + 144))(
                v7,
                v56,
                v32,
                v15,
                v27)
            ^ 1;
      else
        v34 = ((__int64 (__fastcall *)(_DWORD *, __int64, __int64, __int64, __int64))v33)(v7, v56, v32, v15, v27);
      if ( v34 )
      {
        v35 = *(unsigned int *)(v13 + 76);
        if ( (_DWORD)v35 != -1 )
          v18 += v35;
      }
      if ( *(_BYTE *)(v13 + 64) )
        goto LABEL_50;
    }
    if ( *(_BYTE *)(v13 + 36)
      || sub_1F4B450(v59, v56)
      && -858993459 * (unsigned int)((__int64)(*(_QWORD *)(v13 + 16) - *(_QWORD *)(v13 + 8)) >> 3) != *(_DWORD *)(v13 + 32) )
    {
LABEL_50:
      v36 = v7[3];
    }
    else
    {
      v36 = v7[4];
    }
    if ( v36 < v22 )
      v36 = v22;
    v18 = ~(unsigned __int64)(v36 - 1) & (v36 - 1 + v18);
  }
  *(_QWORD *)(v13 + 48) = v18 - v57;
  v54 = v56 + 320;
  v55 = *(_QWORD *)(v56 + 328);
  if ( v55 == v56 + 320 )
  {
    v43 = v56 + 320;
    v42 = 0;
  }
  else
  {
    v37 = 0;
    do
    {
      for ( i = *(_QWORD *)(v55 + 32); v55 + 24 != i; i = *(_QWORD *)(i + 8) )
      {
        v39 = *(unsigned int *)(i + 40);
        if ( (_DWORD)v39 )
        {
          for ( j = 0; j != v39; ++j )
          {
            v41 = *(_QWORD *)(i + 32);
            if ( *(_BYTE *)(v41 + 40 * j) == 5 )
            {
              if ( **(_WORD **)(i + 16) == 12 )
              {
                v58 = v37;
                v60 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, int *))(*(_QWORD *)v53 + 176LL))(
                        v53,
                        v56,
                        *(unsigned int *)(v41 + 24),
                        v61);
                sub_1E31400(*(char **)(i + 32), v61[0], 0, 0, 0, 0, 0, 0);
                *(_BYTE *)(*(_QWORD *)(i + 32) + 4LL) |= 8u;
                v51 = (_QWORD *)sub_1E16510(i);
                v52 = sub_15C48E0(v51, 0, v60, 0, 0);
                v37 = v58;
                *(_QWORD *)(*(_QWORD *)(i + 32) + 144LL) = v52;
              }
              else
              {
                (*(void (__fastcall **)(__int64, __int64, _QWORD, _QWORD, _QWORD))(*(_QWORD *)v6 + 392LL))(
                  v6,
                  i,
                  0,
                  (unsigned int)j,
                  0);
                v37 = 1;
              }
            }
          }
        }
        if ( (*(_BYTE *)i & 4) == 0 )
        {
          while ( (*(_BYTE *)(i + 46) & 8) != 0 )
            i = *(_QWORD *)(i + 8);
        }
      }
      v55 = *(_QWORD *)(v55 + 8);
    }
    while ( v54 != v55 );
    v42 = v37;
    v43 = *(_QWORD *)(v56 + 328);
  }
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v53 + 80LL))(v53, v56, v43);
  v44 = *(_QWORD *)(v56 + 328);
  if ( v54 != v44 )
  {
    while ( 1 )
    {
      v45 = *(_QWORD *)(v44 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      v46 = (__int64 *)v45;
      if ( v45 != v44 + 24 )
        break;
LABEL_79:
      v44 = *(_QWORD *)(v44 + 8);
      if ( v54 == v44 )
        return v42;
    }
    if ( !v45 )
      BUG();
    v47 = *(_QWORD *)v45;
    v48 = *(_WORD *)(v45 + 46);
    if ( (v47 & 4) != 0 )
    {
      if ( (v48 & 4) != 0 )
        goto LABEL_90;
    }
    else if ( (v48 & 4) != 0 )
    {
      while ( 1 )
      {
        v46 = (__int64 *)(v47 & 0xFFFFFFFFFFFFFFF8LL);
        v48 = *(_WORD *)((v47 & 0xFFFFFFFFFFFFFFF8LL) + 46);
        if ( (v48 & 4) == 0 )
          break;
        v47 = *v46;
      }
    }
    if ( (v48 & 8) != 0 )
    {
      LOBYTE(v49) = sub_1E15D00((__int64)v46, 8u, 1);
      goto LABEL_77;
    }
LABEL_90:
    v49 = (*(_QWORD *)(v46[2] + 8) >> 3) & 1LL;
LABEL_77:
    if ( (_BYTE)v49 )
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v53 + 88LL))(v53, v56, v44);
    goto LABEL_79;
  }
  return v42;
}
