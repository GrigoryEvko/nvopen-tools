// Function: sub_1986060
// Address: 0x1986060
//
void __fastcall sub_1986060(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v6; // r14
  unsigned int i; // eax
  _QWORD *v9; // rdx
  unsigned int v10; // esi
  __int64 v11; // rbx
  __int64 v12; // r9
  unsigned int v13; // r8d
  __int64 *v14; // rdi
  __int64 v15; // rcx
  __int64 *v16; // r11
  int v17; // eax
  int v18; // eax
  __int64 v19; // rax
  __int64 *v20; // r13
  __int64 v21; // r14
  __int64 v22; // rax
  int v23; // r8d
  int v24; // r9d
  __int64 v25; // rax
  __int64 v26; // rbx
  _QWORD *v27; // rax
  __int64 v28; // rdi
  __int64 v29; // r13
  _QWORD *v30; // rdx
  int v31; // r8d
  int v32; // r9d
  __int64 v33; // rax
  int v34; // eax
  int v35; // ecx
  __int64 v36; // rdi
  unsigned int v37; // edx
  __int64 v38; // rsi
  int v39; // r9d
  __int64 *v40; // r8
  int v41; // eax
  int v42; // edx
  __int64 v43; // rsi
  int v44; // r8d
  __int64 *v45; // rdi
  unsigned int v46; // r13d
  __int64 v47; // rcx
  int v48; // [rsp+8h] [rbp-E8h]
  __int64 v49; // [rsp+8h] [rbp-E8h]
  _QWORD *v52; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v53; // [rsp+38h] [rbp-B8h]
  _QWORD v54[22]; // [rsp+40h] [rbp-B0h] BYREF

  v6 = v54;
  v52 = v54;
  v54[0] = a2;
  v53 = 0x1000000001LL;
  for ( i = 1; ; i = v53 )
  {
    v9 = &v6[i];
    if ( !i )
      break;
    while ( 1 )
    {
      v10 = *(_DWORD *)(a5 + 24);
      --i;
      v11 = *(v9 - 1);
      LODWORD(v53) = i;
      if ( !v10 )
      {
        ++*(_QWORD *)a5;
        goto LABEL_49;
      }
      v12 = *(_QWORD *)(a5 + 8);
      v13 = (v10 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v14 = (__int64 *)(v12 + 8LL * v13);
      v15 = *v14;
      if ( v11 != *v14 )
        break;
LABEL_5:
      --v9;
      if ( !i )
        goto LABEL_6;
    }
    v48 = 1;
    v16 = 0;
    while ( v15 != -8 )
    {
      if ( v16 || v15 != -16 )
        v14 = v16;
      v13 = (v10 - 1) & (v48 + v13);
      v15 = *(_QWORD *)(v12 + 8LL * v13);
      if ( v11 == v15 )
        goto LABEL_5;
      ++v48;
      v16 = v14;
      v14 = (__int64 *)(v12 + 8LL * v13);
    }
    v17 = *(_DWORD *)(a5 + 16);
    if ( !v16 )
      v16 = v14;
    ++*(_QWORD *)a5;
    v18 = v17 + 1;
    if ( 4 * v18 < 3 * v10 )
    {
      if ( v10 - (v18 + *(_DWORD *)(a5 + 20)) > v10 >> 3 )
        goto LABEL_15;
      sub_1467110(a5, v10);
      v41 = *(_DWORD *)(a5 + 24);
      if ( v41 )
      {
        v42 = v41 - 1;
        v43 = *(_QWORD *)(a5 + 8);
        v44 = 1;
        v45 = 0;
        v46 = (v41 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v16 = (__int64 *)(v43 + 8LL * v46);
        v47 = *v16;
        v18 = *(_DWORD *)(a5 + 16) + 1;
        if ( v11 != *v16 )
        {
          while ( v47 != -8 )
          {
            if ( !v45 && v47 == -16 )
              v45 = v16;
            v46 = v42 & (v44 + v46);
            v16 = (__int64 *)(v43 + 8LL * v46);
            v47 = *v16;
            if ( v11 == *v16 )
              goto LABEL_15;
            ++v44;
          }
          if ( v45 )
            v16 = v45;
        }
        goto LABEL_15;
      }
LABEL_77:
      ++*(_DWORD *)(a5 + 16);
      BUG();
    }
LABEL_49:
    sub_1467110(a5, 2 * v10);
    v34 = *(_DWORD *)(a5 + 24);
    if ( !v34 )
      goto LABEL_77;
    v35 = v34 - 1;
    v36 = *(_QWORD *)(a5 + 8);
    v37 = (v34 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v16 = (__int64 *)(v36 + 8LL * v37);
    v38 = *v16;
    v18 = *(_DWORD *)(a5 + 16) + 1;
    if ( v11 != *v16 )
    {
      v39 = 1;
      v40 = 0;
      while ( v38 != -8 )
      {
        if ( v38 == -16 && !v40 )
          v40 = v16;
        v37 = v35 & (v39 + v37);
        v16 = (__int64 *)(v36 + 8LL * v37);
        v38 = *v16;
        if ( v11 == *v16 )
          goto LABEL_15;
        ++v39;
      }
      if ( v40 )
        v16 = v40;
    }
LABEL_15:
    *(_DWORD *)(a5 + 16) = v18;
    if ( *v16 != -8 )
      --*(_DWORD *)(a5 + 20);
    *v16 = v11;
    if ( !sub_13A0E30(a4, v11) && *(_QWORD *)(v11 + 8) )
    {
      v49 = v11;
      v26 = *(_QWORD *)(v11 + 8);
      do
      {
        while ( 1 )
        {
          v27 = sub_1648700(v26);
          v28 = *(_QWORD *)(a1 + 8);
          v29 = (__int64)v27;
          if ( *((_BYTE *)v27 + 16) != 77
            || ((*((_BYTE *)v27 + 23) & 0x40) == 0
              ? (v30 = &v27[-3 * (*((_DWORD *)v27 + 5) & 0xFFFFFFF)])
              : (v30 = (_QWORD *)*(v27 - 1)),
                **(_QWORD **)(v28 + 32) != v30[3 * *((unsigned int *)v27 + 14)
                                             + 1
                                             + -1431655765 * (unsigned int)((v26 - (__int64)v30) >> 3)]) )
          {
            if ( sub_1377F70(v28 + 56, v27[5]) && !sub_13A0E30(a3, v29) )
              break;
          }
          v26 = *(_QWORD *)(v26 + 8);
          if ( !v26 )
            goto LABEL_46;
        }
        v33 = (unsigned int)v53;
        if ( (unsigned int)v53 >= HIDWORD(v53) )
        {
          sub_16CD150((__int64)&v52, v54, 0, 8, v31, v32);
          v33 = (unsigned int)v53;
        }
        v52[v33] = v29;
        LODWORD(v53) = v53 + 1;
        v26 = *(_QWORD *)(v26 + 8);
      }
      while ( v26 );
LABEL_46:
      v11 = v49;
    }
    v19 = 3LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v11 + 23) & 0x40) != 0 )
    {
      v20 = *(__int64 **)(v11 - 8);
      v11 = (__int64)&v20[v19];
    }
    else
    {
      v20 = (__int64 *)(v11 - v19 * 8);
    }
    if ( (__int64 *)v11 != v20 )
    {
      do
      {
        while ( 1 )
        {
          v21 = *v20;
          if ( *(_BYTE *)(*v20 + 16) > 0x17u )
          {
            v22 = *(_QWORD *)(v21 + 8);
            if ( v22 )
            {
              if ( !*(_QWORD *)(v22 + 8)
                && sub_1377F70(*(_QWORD *)(a1 + 8) + 56LL, *(_QWORD *)(v21 + 40))
                && !sub_13A0E30(a3, v21)
                && !sub_13A0E30(a4, v21) )
              {
                break;
              }
            }
          }
          v20 += 3;
          if ( v20 == (__int64 *)v11 )
            goto LABEL_32;
        }
        v25 = (unsigned int)v53;
        if ( (unsigned int)v53 >= HIDWORD(v53) )
        {
          sub_16CD150((__int64)&v52, v54, 0, 8, v23, v24);
          v25 = (unsigned int)v53;
        }
        v20 += 3;
        v52[v25] = v21;
        LODWORD(v53) = v53 + 1;
      }
      while ( v20 != (__int64 *)v11 );
    }
LABEL_32:
    v6 = v52;
  }
LABEL_6:
  if ( v6 != v54 )
    _libc_free((unsigned __int64)v6);
}
