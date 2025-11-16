// Function: sub_2F51790
// Address: 0x2f51790
//
__int64 __fastcall sub_2F51790(_QWORD *a1, _QWORD *a2, unsigned int *a3, __int64 a4)
{
  unsigned int *v6; // r13
  __int64 v7; // rbx
  unsigned int v8; // edx
  int *v9; // rax
  __int64 v10; // r12
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r9
  unsigned __int64 v15; // rbx
  _QWORD *v16; // r10
  __int64 v17; // rdi
  __int64 v18; // r8
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rsi
  int v21; // edi
  __int64 k; // rcx
  __int64 m; // rdi
  __int16 v24; // dx
  __int64 v25; // r10
  unsigned int v26; // ecx
  unsigned int v27; // r11d
  __int64 *v28; // rdx
  __int64 v29; // rsi
  unsigned __int64 v30; // rdx
  __int64 i; // rdi
  __int64 j; // rsi
  __int16 v33; // ax
  __int64 v34; // rcx
  __int64 v35; // rsi
  unsigned int v36; // ebx
  __int64 *v37; // rax
  __int64 v38; // rdi
  __int64 v39; // rcx
  _QWORD *v40; // rdi
  __int64 v41; // rsi
  __int64 *v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // r12
  int v47; // eax
  int v48; // edx
  int v49; // r11d
  int v50; // edi
  __int64 v51; // [rsp+8h] [rbp-B8h]
  _QWORD *v52; // [rsp+10h] [rbp-B0h]
  unsigned int v53; // [rsp+18h] [rbp-A8h]
  __int64 v54; // [rsp+18h] [rbp-A8h]
  __int64 v55; // [rsp+18h] [rbp-A8h]
  unsigned int *v56; // [rsp+20h] [rbp-A0h]
  unsigned int v57; // [rsp+28h] [rbp-98h]
  unsigned int v58; // [rsp+2Ch] [rbp-94h]
  _DWORD v59[8]; // [rsp+30h] [rbp-90h] BYREF
  _DWORD v60[28]; // [rsp+50h] [rbp-70h] BYREF

  v56 = &a3[a4];
  if ( a3 != v56 )
  {
    v57 = 0;
    v6 = a3;
    v58 = 0;
    while ( 1 )
    {
      v7 = *a2;
      v8 = *v6;
      v9 = &dword_503BD90;
      if ( *a2 )
      {
        v10 = 24LL * v8;
        v9 = (int *)(v10 + *(_QWORD *)(v7 + 512));
        if ( *v9 != *(_DWORD *)(v7 + 4) )
        {
          v53 = *v6;
          sub_3501C20(*a2);
          v8 = v53;
          v9 = (int *)(v10 + *(_QWORD *)(v7 + 512));
        }
      }
      a2[1] = v9;
      if ( (*((_QWORD *)v9 + 1) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      {
        v59[v57++] = v8;
        if ( v57 == 8 )
        {
          sub_2FAF8F0(a1[104], v59, 8);
          v57 = 0;
        }
        goto LABEL_5;
      }
      v11 = v8;
      v12 = *(_QWORD *)(a1[96] + 96LL);
      v60[2 * v58] = v8;
      v54 = *(_QWORD *)(v12 + 8LL * v8);
      v13 = sub_2E319B0(v54, 1);
      v14 = v58;
      v15 = v13;
      if ( v13 != v54 + 48 )
        break;
LABEL_35:
      v39 = a2[1];
      v40 = (_QWORD *)a1[124];
      v41 = *(_QWORD *)(*(_QWORD *)(a1[98] + 152LL) + 16 * v11);
      LOBYTE(v60[2 * v14 + 1]) = (*(_DWORD *)((v41 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v41 >> 1) & 3) < (*(_DWORD *)((*(_QWORD *)(v39 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)(v39 + 8) >> 1) & 3)
                               ? 2
                               : 4;
      v42 = (__int64 *)(v40[7] + 16LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(*v40 + 96LL) + 8 * v11) + 24LL));
      v43 = *v42;
      if ( (*v42 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v42[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v55 = v14;
        v43 = sub_2FB0650(v40 + 6, v40[5], *(_QWORD *)(*(_QWORD *)(*v40 + 96LL) + 8 * v11));
        v39 = a2[1];
        v14 = v55;
      }
      ++v58;
      BYTE1(v60[2 * v14 + 1]) = (*(_DWORD *)((*(_QWORD *)(v39 + 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                               | (unsigned int)(*(__int64 *)(v39 + 16) >> 1) & 3) < (*(_DWORD *)((v43
                                                                                                & 0xFFFFFFFFFFFFFFF8LL)
                                                                                               + 24)
                                                                                   | (unsigned int)(v43 >> 1) & 3)
                              ? 2
                              : 4;
      if ( v58 == 8 )
      {
        ++v6;
        sub_2FAF660(a1[104], v60, 8);
        v58 = 0;
        if ( v56 == v6 )
        {
LABEL_40:
          v44 = v58;
          v45 = v57;
          goto LABEL_41;
        }
      }
      else
      {
LABEL_5:
        if ( v56 == ++v6 )
          goto LABEL_40;
      }
    }
    v16 = (_QWORD *)a1[124];
    v52 = v16;
    v17 = *(_QWORD *)(*(_QWORD *)(*v16 + 96LL) + 8 * v11);
    v18 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v16[6] + 32LL) + 152LL) + 16LL * *(unsigned int *)(v17 + 24));
    if ( v17 + 48 == (*(_QWORD *)(v17 + 48) & 0xFFFFFFFFFFFFFFF8LL)
      || (v51 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v16[6] + 32LL) + 152LL) + 16LL * *(unsigned int *)(v17 + 24)),
          v19 = sub_2E312E0(v17, *(_QWORD *)(v17 + 56), 0, 1),
          v18 = v51,
          v14 = v58,
          v20 = v19,
          v17 + 48 == v19) )
    {
LABEL_24:
      v30 = v15;
      for ( i = *(_QWORD *)(a1[4] + 32LL); (*(_BYTE *)(v30 + 44) & 4) != 0; v30 = *(_QWORD *)v30 & 0xFFFFFFFFFFFFFFF8LL )
        ;
      for ( ; (*(_BYTE *)(v15 + 44) & 8) != 0; v15 = *(_QWORD *)(v15 + 8) )
        ;
      for ( j = *(_QWORD *)(v15 + 8); j != v30; v30 = *(_QWORD *)(v30 + 8) )
      {
        v33 = *(_WORD *)(v30 + 68);
        if ( (unsigned __int16)(v33 - 14) > 4u && v33 != 24 )
          break;
      }
      v34 = *(unsigned int *)(i + 144);
      v35 = *(_QWORD *)(i + 128);
      if ( (_DWORD)v34 )
      {
        v36 = (v34 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
        v37 = (__int64 *)(v35 + 16LL * v36);
        v38 = *v37;
        if ( v30 == *v37 )
          goto LABEL_34;
        v47 = 1;
        while ( v38 != -4096 )
        {
          v49 = v47 + 1;
          v36 = (v34 - 1) & (v47 + v36);
          v37 = (__int64 *)(v35 + 16LL * v36);
          v38 = *v37;
          if ( v30 == *v37 )
            goto LABEL_34;
          v47 = v49;
        }
      }
      v37 = (__int64 *)(v35 + 16 * v34);
LABEL_34:
      if ( *(_DWORD *)((v37[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) < *(_DWORD *)((v18 & 0xFFFFFFFFFFFFFFF8LL) + 24) )
        return 0;
      goto LABEL_35;
    }
    v21 = *(_DWORD *)(v19 + 44) & 0xFFFFFF;
    for ( k = *(_QWORD *)(v52[6] + 32LL); (*(_BYTE *)(v19 + 44) & 4) != 0; v19 = *(_QWORD *)v19 & 0xFFFFFFFFFFFFFFF8LL )
      ;
    if ( (v21 & 8) != 0 )
    {
      do
        v20 = *(_QWORD *)(v20 + 8);
      while ( (*(_BYTE *)(v20 + 44) & 8) != 0 );
    }
    for ( m = *(_QWORD *)(v20 + 8); m != v19; v19 = *(_QWORD *)(v19 + 8) )
    {
      v24 = *(_WORD *)(v19 + 68);
      if ( (unsigned __int16)(v24 - 14) > 4u && v24 != 24 )
        break;
    }
    v25 = *(_QWORD *)(k + 128);
    v26 = *(_DWORD *)(k + 144);
    if ( v26 )
    {
      v27 = (v26 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v28 = (__int64 *)(v25 + 16LL * v27);
      v29 = *v28;
      if ( *v28 == v19 )
      {
LABEL_23:
        v18 = v28[1];
        goto LABEL_24;
      }
      v48 = 1;
      while ( v29 != -4096 )
      {
        v50 = v48 + 1;
        v27 = (v26 - 1) & (v48 + v27);
        v28 = (__int64 *)(v25 + 16LL * v27);
        v29 = *v28;
        if ( *v28 == v19 )
          goto LABEL_23;
        v48 = v50;
      }
    }
    v28 = (__int64 *)(v25 + 16LL * v26);
    goto LABEL_23;
  }
  v45 = 0;
  v44 = 0;
LABEL_41:
  sub_2FAF660(a1[104], v60, v44);
  sub_2FAF8F0(a1[104], v59, v45);
  return 1;
}
