// Function: sub_1802D00
// Address: 0x1802d00
//
__int64 __fastcall sub_1802D00(__int64 a1, _DWORD *a2, int a3, char a4)
{
  int v6; // r9d
  int v7; // r14d
  bool v8; // r10
  bool v9; // r12
  bool v10; // bl
  unsigned __int64 v11; // r8
  _QWORD *v12; // rax
  _DWORD *v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rax
  _DWORD *v17; // r8
  __int64 v18; // rcx
  __int64 v19; // rdx
  unsigned __int64 v20; // rsi
  _QWORD *v21; // rax
  _DWORD *v22; // r8
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rax
  _DWORD *v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v30; // rax
  bool v32; // [rsp+Ah] [rbp-56h]
  bool v33; // [rsp+Bh] [rbp-55h]
  int v34; // [rsp+Ch] [rbp-54h]
  bool v36; // [rsp+14h] [rbp-4Ch]
  char v37; // [rsp+15h] [rbp-4Bh]
  bool v38; // [rsp+16h] [rbp-4Ah]
  bool v39; // [rsp+17h] [rbp-49h]
  int v40; // [rsp+18h] [rbp-48h]
  int v41; // [rsp+1Ch] [rbp-44h]
  unsigned int v42; // [rsp+24h] [rbp-3Ch] BYREF
  int v43; // [rsp+28h] [rbp-38h] BYREF
  _DWORD v44[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v6 = a2[11];
  v7 = a2[8];
  v40 = a2[12];
  v32 = v6 == 7 || (unsigned int)(v6 - 29) <= 1;
  v41 = a2[10];
  if ( v7 == 32 )
  {
    v36 = 0;
    v33 = 0;
    v37 = 1;
    v8 = 0;
    v39 = a2[10] == 3 && v6 == 27;
    goto LABEL_4;
  }
  if ( v7 == 17 )
  {
    v39 = 0;
    v8 = 1;
    v37 = 0;
    v36 = 0;
    v33 = 0;
LABEL_4:
    v9 = v7 == 2 || (unsigned int)(v7 - 29) <= 1;
    v10 = v7 != 3 && !v8;
    goto LABEL_5;
  }
  v8 = v7 == 18;
  if ( v7 == 10 )
  {
    v39 = 0;
    v8 = 0;
    v37 = 0;
    v36 = 0;
    v33 = 1;
    goto LABEL_4;
  }
  v33 = v7 == 11;
  if ( v7 == 12 )
  {
    v39 = 0;
    v37 = 0;
    v36 = 1;
    goto LABEL_4;
  }
  v39 = 0;
  v37 = 0;
  v10 = v7 != 18 && v7 != 3;
  v36 = v7 == 13;
  if ( v7 != 1 )
    goto LABEL_4;
  v9 = 1;
LABEL_5:
  v38 = v8;
  v34 = a2[11];
  *(_DWORD *)a1 = 2 * (v41 == 12) + 3;
  v11 = sub_16D5D50();
  v12 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v13 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v14 = v12[2];
        v15 = v12[3];
        if ( v11 <= v12[4] )
          break;
        v12 = (_QWORD *)v12[3];
        if ( !v15 )
          goto LABEL_10;
      }
      v13 = v12;
      v12 = (_QWORD *)v12[2];
    }
    while ( v14 );
LABEL_10:
    if ( v13 != dword_4FA0208 && v11 >= *((_QWORD *)v13 + 4) )
    {
      v16 = *((_QWORD *)v13 + 7);
      if ( v16 )
      {
        v17 = v13 + 12;
        do
        {
          while ( 1 )
          {
            v18 = *(_QWORD *)(v16 + 16);
            v19 = *(_QWORD *)(v16 + 24);
            if ( *(_DWORD *)(v16 + 32) >= dword_4FA7948 )
              break;
            v16 = *(_QWORD *)(v16 + 24);
            if ( !v19 )
              goto LABEL_17;
          }
          v17 = (_DWORD *)v16;
          v16 = *(_QWORD *)(v16 + 16);
        }
        while ( v18 );
LABEL_17:
        if ( v13 + 12 != v17 && dword_4FA7948 >= v17[8] && (int)v17[9] > 0 )
          *(_DWORD *)a1 = dword_4FA79E0;
      }
    }
  }
  if ( a3 == 32 )
  {
    if ( v40 != 10 )
    {
      if ( v33 )
      {
        *(_QWORD *)(a1 + 8) = 178913280;
      }
      else if ( v34 == 5 || v34 == 12 || v32 )
      {
        *(_QWORD *)(a1 + 8) = 0x40000000;
      }
      else if ( v34 == 15 )
      {
        *(_QWORD *)(a1 + 8) = 805306368;
      }
      else if ( v41 == 12 )
      {
        *(_QWORD *)(a1 + 8) = 2684354560u - ((0x80000000uLL >> *(_DWORD *)a1) + (0x20000000uLL >> *(_DWORD *)a1));
      }
      else
      {
        *(_QWORD *)(a1 + 8) = 0x20000000;
      }
      goto LABEL_25;
    }
LABEL_81:
    *(_QWORD *)(a1 + 8) = -1;
    goto LABEL_25;
  }
  if ( v34 == 6 )
  {
    *(_QWORD *)(a1 + 8) = 0;
    goto LABEL_25;
  }
  if ( v38 )
  {
LABEL_24:
    *(_QWORD *)(a1 + 8) = 0x100000000000LL;
    goto LABEL_25;
  }
  if ( v7 == 26 )
  {
    *(_QWORD *)(a1 + 8) = 0x10000000000000LL;
    goto LABEL_25;
  }
  if ( v34 == 5 || v34 == 12 )
  {
    *(_QWORD *)(a1 + 8) = 0x400000000000LL;
    goto LABEL_25;
  }
  if ( v39 )
  {
    *(_QWORD *)(a1 + 8) = 0x10000000000LL;
    goto LABEL_25;
  }
  if ( v34 == 9 && v37 )
  {
    if ( a4 )
      *(_QWORD *)(a1 + 8) = 0xDFFFFC0000000000LL;
    else
      *(_QWORD *)(a1 + 8) = (-4096LL << *(_DWORD *)a1) & 0x7FFFFFFF;
    goto LABEL_25;
  }
  if ( v34 == 15 && v37 )
    goto LABEL_81;
  if ( v36 )
  {
    *(_QWORD *)(a1 + 8) = 0x2000000000LL;
  }
  else if ( v32 )
  {
    v30 = 0x100000000000LL;
    if ( v7 != 32 )
      v30 = -1;
    *(_QWORD *)(a1 + 8) = v30;
  }
  else
  {
    if ( v7 != 3 )
      goto LABEL_24;
    *(_QWORD *)(a1 + 8) = 0x1000000000LL;
  }
LABEL_25:
  if ( byte_4FA8900 )
    *(_QWORD *)(a1 + 8) = -1;
  v20 = sub_16D5D50();
  v21 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v22 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v23 = v21[2];
        v24 = v21[3];
        if ( v20 <= v21[4] )
          break;
        v21 = (_QWORD *)v21[3];
        if ( !v24 )
          goto LABEL_32;
      }
      v22 = v21;
      v21 = (_QWORD *)v21[2];
    }
    while ( v23 );
LABEL_32:
    if ( v22 != dword_4FA0208 && v20 >= *((_QWORD *)v22 + 4) )
    {
      v25 = *((_QWORD *)v22 + 7);
      if ( v25 )
      {
        v26 = v22 + 12;
        do
        {
          while ( 1 )
          {
            v27 = *(_QWORD *)(v25 + 16);
            v28 = *(_QWORD *)(v25 + 24);
            if ( *(_DWORD *)(v25 + 32) >= dword_4FA7868 )
              break;
            v25 = *(_QWORD *)(v25 + 24);
            if ( !v28 )
              goto LABEL_39;
          }
          v26 = (_DWORD *)v25;
          v25 = *(_QWORD *)(v25 + 16);
        }
        while ( v27 );
LABEL_39:
        if ( v26 != v22 + 12 && dword_4FA7868 >= v26[8] && (int)v26[9] > 0 )
          *(_QWORD *)(a1 + 8) = qword_4FA7900;
      }
    }
  }
  if ( v10 )
  {
    v10 = !v39 && v7 != 26;
    if ( v10 )
      v10 = *(_QWORD *)(a1 + 8) != -1 && (*(_QWORD *)(a1 + 8) & (*(_QWORD *)(a1 + 8) - 1LL)) == 0;
  }
  *(_BYTE *)(a1 + 16) = v10;
  if ( v40 != 10
    || (sub_16E22F0((__int64)a2, &v42, &v43, v44), !sub_16E2900((__int64)a2)) && v42 <= 0x14
    || !byte_4FA8820 )
  {
    v9 = 0;
  }
  *(_BYTE *)(a1 + 17) = v9;
  return a1;
}
