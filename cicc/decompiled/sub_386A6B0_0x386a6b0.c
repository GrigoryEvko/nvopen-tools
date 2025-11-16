// Function: sub_386A6B0
// Address: 0x386a6b0
//
__int64 __fastcall sub_386A6B0(__int64 a1, unsigned __int8 *a2, __m128i a3, __m128i a4)
{
  unsigned __int8 *v6; // r14
  unsigned __int8 *v7; // r15
  __int64 v8; // rax
  __int64 v9; // rdx
  char v10; // al
  int v11; // edi
  int v12; // ecx
  _QWORD *v13; // r14
  __int64 v14; // r13
  unsigned int v15; // esi
  __int64 v16; // rdi
  unsigned int v17; // ecx
  unsigned __int8 **v18; // rax
  unsigned __int8 *v19; // rdx
  __int64 v21; // rax
  __int64 v22; // rcx
  int v23; // eax
  int v24; // eax
  unsigned int v25; // esi
  unsigned __int8 **v26; // rdx
  unsigned __int8 *v27; // rdi
  __int64 v28; // rdi
  int v29; // eax
  unsigned int v30; // esi
  unsigned __int8 **v31; // rdx
  unsigned __int8 *v32; // r8
  unsigned __int8 *v33; // rdx
  int v34; // eax
  int v35; // esi
  __int64 v36; // r8
  unsigned int v37; // edx
  int v38; // ecx
  unsigned __int8 *v39; // rdi
  int v40; // edx
  int v41; // edx
  int v42; // r8d
  int v43; // r10d
  unsigned __int8 **v44; // r9
  int v45; // edx
  int v46; // eax
  int v47; // edx
  __int64 v48; // rdi
  unsigned __int8 **v49; // r8
  unsigned int v50; // r15d
  int v51; // r9d
  unsigned __int8 *v52; // rsi
  int v53; // r9d
  int v54; // r10d
  unsigned __int8 **v55; // r9
  _QWORD v56[10]; // [rsp+0h] [rbp-50h] BYREF

  v6 = (unsigned __int8 *)*((_QWORD *)a2 - 6);
  v7 = (unsigned __int8 *)*((_QWORD *)a2 - 3);
  if ( v6[16] <= 0x10u )
  {
    if ( v7[16] <= 0x10u )
      goto LABEL_3;
    v21 = *(_QWORD *)(a1 + 40);
    v22 = *(_QWORD *)(v21 + 8);
    v23 = *(_DWORD *)(v21 + 24);
    if ( !v23 )
      goto LABEL_3;
    goto LABEL_18;
  }
  v28 = *(_QWORD *)(a1 + 40);
  v29 = *(_DWORD *)(v28 + 24);
  if ( !v29 )
    goto LABEL_3;
  v24 = v29 - 1;
  v22 = *(_QWORD *)(v28 + 8);
  v30 = v24 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v31 = (unsigned __int8 **)(v22 + 16LL * v30);
  v32 = *v31;
  if ( v6 != *v31 )
  {
    v40 = 1;
    while ( v32 != (unsigned __int8 *)-8LL )
    {
      v53 = v40 + 1;
      v30 = v24 & (v40 + v30);
      v31 = (unsigned __int8 **)(v22 + 16LL * v30);
      v32 = *v31;
      if ( v6 == *v31 )
        goto LABEL_25;
      v40 = v53;
    }
    v33 = v6;
LABEL_26:
    v6 = v33;
    if ( v7[16] <= 0x10u )
      goto LABEL_3;
    goto LABEL_19;
  }
LABEL_25:
  v33 = v31[1];
  if ( v33 )
    goto LABEL_26;
  if ( v7[16] <= 0x10u )
    goto LABEL_3;
  v22 = *(_QWORD *)(v28 + 8);
  v23 = *(_DWORD *)(v28 + 24);
LABEL_18:
  v24 = v23 - 1;
LABEL_19:
  v25 = v24 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v26 = (unsigned __int8 **)(v22 + 16LL * v25);
  v27 = *v26;
  if ( v7 == *v26 )
  {
LABEL_20:
    if ( v26[1] )
      v7 = v26[1];
  }
  else
  {
    v41 = 1;
    while ( v27 != (unsigned __int8 *)-8LL )
    {
      v42 = v41 + 1;
      v25 = v24 & (v41 + v25);
      v26 = (unsigned __int8 **)(v22 + 16LL * v25);
      v27 = *v26;
      if ( v7 == *v26 )
        goto LABEL_20;
      v41 = v42;
    }
  }
LABEL_3:
  v8 = sub_15F2050((__int64)a2);
  v9 = sub_1632FA0(v8);
  v10 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  if ( v10 == 16 )
    v10 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
  v56[0] = v9;
  v11 = a2[16];
  memset(&v56[1], 0, 32);
  if ( (unsigned __int8)(v10 - 1) <= 5u || (_BYTE)v11 == 76 )
  {
    v12 = a2[17] >> 1;
    if ( v12 == 127 )
      LOBYTE(v12) = -1;
    v13 = sub_13E1150(v11 - 24, v6, v7, v12, v56);
    if ( !v13 )
      return sub_386A280(a1, (__int64 *)a2, a3, a4);
LABEL_9:
    if ( *((_BYTE *)v13 + 16) > 0x10u )
      return 1;
    v14 = *(_QWORD *)(a1 + 40);
    v15 = *(_DWORD *)(v14 + 24);
    if ( v15 )
    {
      v16 = *(_QWORD *)(v14 + 8);
      v17 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v18 = (unsigned __int8 **)(v16 + 16LL * v17);
      v19 = *v18;
      if ( a2 == *v18 )
      {
LABEL_12:
        v18[1] = (unsigned __int8 *)v13;
        return 1;
      }
      v43 = 1;
      v44 = 0;
      while ( v19 != (unsigned __int8 *)-8LL )
      {
        if ( !v44 && v19 == (unsigned __int8 *)-16LL )
          v44 = v18;
        v17 = (v15 - 1) & (v43 + v17);
        v18 = (unsigned __int8 **)(v16 + 16LL * v17);
        v19 = *v18;
        if ( a2 == *v18 )
          goto LABEL_12;
        ++v43;
      }
      v45 = *(_DWORD *)(v14 + 16);
      if ( v44 )
        v18 = v44;
      ++*(_QWORD *)v14;
      v38 = v45 + 1;
      if ( 4 * (v45 + 1) < 3 * v15 )
      {
        if ( v15 - *(_DWORD *)(v14 + 20) - v38 > v15 >> 3 )
        {
LABEL_31:
          *(_DWORD *)(v14 + 16) = v38;
          if ( *v18 != (unsigned __int8 *)-8LL )
            --*(_DWORD *)(v14 + 20);
          *v18 = a2;
          v18[1] = 0;
          goto LABEL_12;
        }
        sub_19B8820(v14, v15);
        v46 = *(_DWORD *)(v14 + 24);
        if ( v46 )
        {
          v47 = v46 - 1;
          v48 = *(_QWORD *)(v14 + 8);
          v49 = 0;
          v50 = (v46 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v51 = 1;
          v38 = *(_DWORD *)(v14 + 16) + 1;
          v18 = (unsigned __int8 **)(v48 + 16LL * v50);
          v52 = *v18;
          if ( a2 != *v18 )
          {
            while ( v52 != (unsigned __int8 *)-8LL )
            {
              if ( !v49 && v52 == (unsigned __int8 *)-16LL )
                v49 = v18;
              v50 = v47 & (v51 + v50);
              v18 = (unsigned __int8 **)(v48 + 16LL * v50);
              v52 = *v18;
              if ( a2 == *v18 )
                goto LABEL_31;
              ++v51;
            }
            if ( v49 )
              v18 = v49;
          }
          goto LABEL_31;
        }
LABEL_77:
        ++*(_DWORD *)(v14 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)v14;
    }
    sub_19B8820(v14, 2 * v15);
    v34 = *(_DWORD *)(v14 + 24);
    if ( v34 )
    {
      v35 = v34 - 1;
      v36 = *(_QWORD *)(v14 + 8);
      v37 = (v34 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v38 = *(_DWORD *)(v14 + 16) + 1;
      v18 = (unsigned __int8 **)(v36 + 16LL * v37);
      v39 = *v18;
      if ( a2 != *v18 )
      {
        v54 = 1;
        v55 = 0;
        while ( v39 != (unsigned __int8 *)-8LL )
        {
          if ( !v55 && v39 == (unsigned __int8 *)-16LL )
            v55 = v18;
          v37 = v35 & (v54 + v37);
          v18 = (unsigned __int8 **)(v36 + 16LL * v37);
          v39 = *v18;
          if ( a2 == *v18 )
            goto LABEL_31;
          ++v54;
        }
        if ( v55 )
          v18 = v55;
      }
      goto LABEL_31;
    }
    goto LABEL_77;
  }
  v13 = sub_13E1140(v11 - 24, v6, v7, v56);
  if ( v13 )
    goto LABEL_9;
  return sub_386A280(a1, (__int64 *)a2, a3, a4);
}
