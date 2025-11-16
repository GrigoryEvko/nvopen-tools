// Function: sub_1EB8380
// Address: 0x1eb8380
//
__int64 __fastcall sub_1EB8380(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  _QWORD *v5; // r13
  __int64 *v6; // r14
  __int64 (*v7)(); // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  int v12; // ecx
  unsigned int v13; // eax
  int v14; // r8d
  int v15; // r9d
  unsigned __int16 v16; // dx
  __int64 v17; // rbx
  unsigned __int16 *v18; // rdi
  unsigned __int16 *v19; // r8
  unsigned __int16 *v20; // r11
  unsigned __int16 *v21; // rbx
  __int64 v22; // rax
  _DWORD *v23; // r14
  unsigned __int16 v24; // r13
  __int64 v25; // rdx
  unsigned int v26; // ecx
  __int16 v27; // ax
  _WORD *v28; // rcx
  __int16 *v29; // rdx
  unsigned __int16 v30; // r10
  __int16 *v31; // r9
  __int64 v32; // rcx
  unsigned int v33; // eax
  __int64 v34; // rsi
  _DWORD *v35; // rdx
  __int16 v36; // ax
  unsigned int v38; // r14d
  unsigned int v39; // r15d
  unsigned int v40; // eax
  int v41; // r8d
  int v42; // r9d
  int v43; // r8d
  int v44; // r9d
  unsigned __int16 v45; // r8
  __int64 v46; // rsi
  __int64 v47; // rdi
  unsigned int v48; // eax
  int v49; // edx
  __int64 v50; // [rsp+18h] [rbp-58h]
  __int64 v52; // [rsp+28h] [rbp-48h]
  int v53; // [rsp+30h] [rbp-40h]
  unsigned int v54; // [rsp+34h] [rbp-3Ch]
  unsigned __int16 *v55; // [rsp+38h] [rbp-38h]

  v5 = *(_QWORD **)(a1 + 240);
  v52 = a3;
  v53 = *(_DWORD *)(a3 + 8);
  v50 = v53 & 0x7FFFFFFF;
  v6 = (__int64 *)(*(_QWORD *)(v5[3] + 16 * v50) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (int)a4 > 0 )
  {
    v7 = *(__int64 (**)())(**(_QWORD **)(*v5 + 16LL) + 112LL);
    if ( v7 == sub_1D00B10 )
      BUG();
    v9 = *(_QWORD *)(v7() + 232);
    v10 = *v6;
    if ( !*(_BYTE *)(v9 + 8LL * a4 + 4) )
      goto LABEL_12;
    if ( (*(_QWORD *)(v5[38] + 8LL * (a4 >> 6)) & (1LL << a4)) != 0 )
      goto LABEL_12;
    v11 = a4 >> 3;
    if ( (unsigned int)v11 >= *(unsigned __int16 *)(v10 + 22) )
      goto LABEL_12;
    v12 = *(unsigned __int8 *)(*(_QWORD *)(v10 + 8) + v11);
    if ( !_bittest(&v12, a4 & 7) )
      goto LABEL_12;
    v13 = sub_1EB7500(a1, a4);
    if ( v13 <= 0x63 )
    {
      if ( v13 )
        sub_1EB8000(a1, (__int64 *)a2, (unsigned __int16)a4, 1, v14, v15);
      v16 = a4;
      return sub_1EB6B30(a1, v53, v16);
    }
  }
  v10 = *v6;
LABEL_12:
  v17 = *(_QWORD *)(a1 + 264) + 24LL * *(unsigned __int16 *)(v10 + 24);
  if ( *(_DWORD *)(a1 + 272) != *(_DWORD *)v17 )
    sub_1ED7890(a1 + 264);
  v18 = *(unsigned __int16 **)(v17 + 16);
  v19 = &v18[*(unsigned int *)(v17 + 4)];
  v20 = v18;
  v21 = v18;
  if ( v19 != v18 )
  {
    do
    {
      v22 = *v20;
      v23 = (_DWORD *)(*(_QWORD *)(a1 + 648) + 4 * v22);
      v24 = *v20;
      if ( *v23 == 1 )
      {
        v25 = *(_QWORD *)(a1 + 248);
        if ( !v25 )
          BUG();
        v26 = *(_DWORD *)(*(_QWORD *)(v25 + 8) + 24 * v22 + 16);
        v27 = v26 & 0xF;
        v28 = (_WORD *)(*(_QWORD *)(v25 + 56) + 2LL * (v26 >> 4));
        v29 = v28 + 1;
        v30 = *v28 + v24 * v27;
LABEL_20:
        v31 = v29;
        if ( !v29 )
        {
LABEL_28:
          *v23 = *(_DWORD *)(v52 + 8);
          *(_WORD *)(v52 + 12) = v24;
          return v52;
        }
        while ( 1 )
        {
          v32 = *(unsigned int *)(a1 + 1032);
          v33 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 1072) + v30);
          if ( v33 < (unsigned int)v32 )
          {
            v34 = *(_QWORD *)(a1 + 1024);
            while ( 1 )
            {
              v35 = (_DWORD *)(v34 + 4LL * v33);
              if ( v30 == *v35 )
                break;
              v33 += 256;
              if ( (unsigned int)v32 <= v33 )
                goto LABEL_26;
            }
            if ( v35 != (_DWORD *)(v34 + 4 * v32) )
              break;
          }
LABEL_26:
          v36 = *v31;
          v29 = 0;
          ++v31;
          v30 += v36;
          if ( !v36 )
            goto LABEL_20;
          if ( !v31 )
            goto LABEL_28;
        }
      }
      ++v20;
    }
    while ( v19 != v20 );
    v54 = 0;
    v38 = -1;
    v55 = v19;
    do
    {
      v39 = *v21;
      v40 = sub_1EB7500(a1, *v21);
      if ( !v40 )
      {
        *(_DWORD *)(*(_QWORD *)(a1 + 648) + 4LL * (unsigned __int16)v39) = *(_DWORD *)(v52 + 8);
        *(_WORD *)(v52 + 12) = v39;
        return v52;
      }
      if ( v38 > v40 )
      {
        v54 = v39;
        v38 = v40;
      }
      ++v21;
    }
    while ( v55 != v21 );
    if ( v54 )
    {
      sub_1EB8000(a1, (__int64 *)a2, v54, 1, v41, v42);
      v16 = v54;
      return sub_1EB6B30(a1, v53, v16);
    }
  }
  if ( **(_WORD **)(a2 + 16) == 1 )
    sub_1E1A6B0(a2, "inline assembly requires more registers than available", 54);
  else
    sub_1E1A6B0(a2, "ran out of registers during register allocation", 47);
  sub_1EB8000(a1, (__int64 *)a2, *v18, 1, v43, v44);
  v45 = *v18;
  v46 = *(unsigned int *)(a1 + 400);
  v47 = *(_QWORD *)(a1 + 392);
  v48 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 600) + v50);
  if ( v48 >= (unsigned int)v46 )
  {
LABEL_46:
    v52 = v47 + 24 * v46;
    v49 = *(_DWORD *)(v52 + 8);
  }
  else
  {
    while ( 1 )
    {
      v49 = *(_DWORD *)(v47 + 24LL * v48 + 8);
      if ( (v53 & 0x7FFFFFFF) == (v49 & 0x7FFFFFFF) )
        break;
      v48 += 256;
      if ( (unsigned int)v46 <= v48 )
        goto LABEL_46;
    }
    v52 = v47 + 24LL * v48;
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 648) + 4LL * v45) = v49;
  *(_WORD *)(v52 + 12) = v45;
  return v52;
}
