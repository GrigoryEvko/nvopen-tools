// Function: sub_2F51220
// Address: 0x2f51220
//
__int64 __fastcall sub_2F51220(_QWORD *a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r15
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 i; // rdx
  __int64 v14; // rbx
  _QWORD *v15; // rdi
  __int64 v16; // rax
  int v17; // r15d
  _QWORD *v18; // r8
  __int64 v19; // r12
  unsigned int *v20; // r13
  int *v21; // rdx
  unsigned int v22; // esi
  __int64 v23; // rax
  bool v24; // al
  __int64 v25; // rdx
  unsigned int v26; // eax
  __int64 v27; // rsi
  _QWORD *v28; // r10
  __int64 v29; // rdi
  __int64 v30; // r9
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rsi
  int v33; // edi
  __int64 j; // rcx
  __int64 k; // rdi
  __int16 v36; // dx
  __int64 v37; // rsi
  unsigned int v38; // ecx
  unsigned int v39; // r11d
  __int64 *v40; // rdx
  __int64 v41; // rdi
  _BOOL4 v42; // ecx
  __int64 v43; // rax
  bool v44; // cf
  _QWORD *v46; // rdi
  __int64 *v47; // rdx
  __int64 v48; // rax
  unsigned int v49; // edx
  int v50; // edx
  int v51; // r9d
  __int64 v53; // [rsp+10h] [rbp-60h]
  _QWORD *v54; // [rsp+18h] [rbp-58h]
  _QWORD *v55; // [rsp+18h] [rbp-58h]
  __int64 v56; // [rsp+20h] [rbp-50h]
  __int64 v57; // [rsp+20h] [rbp-50h]
  _QWORD *v58; // [rsp+20h] [rbp-50h]
  _BOOL4 v59; // [rsp+28h] [rbp-48h]
  __int64 v60; // [rsp+30h] [rbp-40h]
  unsigned __int64 v61; // [rsp+38h] [rbp-38h]

  v6 = a1;
  v8 = a1[124];
  v9 = *(_QWORD *)(v8 + 280);
  v61 = *(unsigned int *)(v8 + 288);
  v10 = *((unsigned int *)v6 + 6026);
  v60 = v9;
  if ( v61 != v10 )
  {
    if ( v61 >= v10 )
    {
      if ( v61 > *((unsigned int *)v6 + 6027) )
      {
        sub_C8D5F0((__int64)(v6 + 3012), v6 + 3014, v61, 8u, a5, a6);
        v10 = *((unsigned int *)v6 + 6026);
      }
      v11 = v6[3012];
      v12 = v11 + 8 * v10;
      for ( i = v11 + 8 * v61; i != v12; v12 += 8 )
      {
        if ( v12 )
        {
          *(_DWORD *)v12 = 0;
          *(_WORD *)(v12 + 4) = 0;
          *(_BYTE *)(v12 + 6) = 0;
        }
      }
    }
    *((_DWORD *)v6 + 6026) = v61;
  }
  v14 = 0;
  if ( !v61 )
  {
LABEL_46:
    *a3 = v14;
    sub_2FAF660(v6[104], v6[3012], *((unsigned int *)v6 + 6026));
    return sub_2FAFD80(v6[104]);
  }
  v15 = v6;
  v16 = 0;
  v17 = 0;
  v18 = v15;
  while ( 1 )
  {
    v19 = v60 + 40 * v16;
    v20 = (unsigned int *)(v18[3012] + 8 * v16);
    v21 = &dword_503BD90;
    v22 = *(_DWORD *)(*(_QWORD *)v19 + 24LL);
    *v20 = v22;
    v23 = *a2;
    if ( *a2 )
    {
      v21 = (int *)(*(_QWORD *)(v23 + 512) + 24LL * v22);
      if ( *v21 != *(_DWORD *)(v23 + 4) )
      {
        v54 = v18;
        v56 = *a2;
        sub_3501C20(v23);
        v18 = v54;
        v21 = (int *)(*(_QWORD *)(v56 + 512) + 24LL * v22);
      }
    }
    a2[1] = (__int64)v21;
    *((_BYTE *)v20 + 4) = *(_BYTE *)(v19 + 32);
    v24 = 0;
    if ( *(_BYTE *)(v19 + 33) )
      v24 = *(_WORD *)(*(_QWORD *)((*(_QWORD *)(v19 + 16) & 0xFFFFFFFFFFFFFFF8LL) + 16) + 68LL) != 10;
    *((_BYTE *)v20 + 5) = v24;
    *((_BYTE *)v20 + 6) = (*(_QWORD *)(v19 + 24) & 0xFFFFFFFFFFFFFFF8LL) != 0;
    if ( (*(_QWORD *)(a2[1] + 8) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_44;
    if ( !*(_BYTE *)(v19 + 32) )
    {
      v42 = 0;
      goto LABEL_36;
    }
    v25 = *v20;
    v26 = *(_DWORD *)((*(_QWORD *)(a2[1] + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(a2[1] + 8) >> 1) & 3;
    v27 = *(_QWORD *)(*(_QWORD *)(v18[98] + 152LL) + 16 * v25);
    if ( v26 <= (*(_DWORD *)((v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v27 >> 1) & 3) )
    {
      *((_BYTE *)v20 + 4) = 4;
      goto LABEL_21;
    }
    if ( v26 < (*(_DWORD *)((*(_QWORD *)(v19 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
              | (unsigned int)(*(__int64 *)(v19 + 8) >> 1) & 3) )
      break;
    v42 = v26 < (*(_DWORD *)((*(_QWORD *)(v19 + 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
               | (unsigned int)(*(__int64 *)(v19 + 16) >> 1) & 3);
LABEL_36:
    if ( *(_BYTE *)(v19 + 33) )
    {
      v46 = (_QWORD *)v18[124];
      v47 = (__int64 *)(v46[7] + 16LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(*v46 + 96LL) + 8LL * *v20) + 24LL));
      v48 = *v47;
      if ( (*v47 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v47[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v58 = v18;
        v59 = v42;
        v48 = sub_2FB0650(v46 + 6, v46[5], *(_QWORD *)(*(_QWORD *)(*v46 + 96LL) + 8LL * *v20));
        v18 = v58;
        v42 = v59;
      }
      v49 = *(_DWORD *)((*(_QWORD *)(a2[1] + 16) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(a2[1] + 16) >> 1) & 3;
      if ( v49 >= (*(_DWORD *)((v48 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v48 >> 1) & 3) )
      {
        *((_BYTE *)v20 + 5) = 4;
        goto LABEL_39;
      }
      if ( v49 > (*(_DWORD *)((*(_QWORD *)(v19 + 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                | (unsigned int)(*(__int64 *)(v19 + 16) >> 1) & 3) )
      {
        *((_BYTE *)v20 + 5) = 2;
LABEL_39:
        v43 = *(_QWORD *)(*(_QWORD *)(v18[104] + 136LL) + 8LL * *v20);
        while ( 1 )
        {
          v44 = __CFADD__(v43, v14);
          v14 += v43;
          if ( v44 )
            v14 = -1;
          if ( !v42 )
            break;
          v42 = 0;
        }
        goto LABEL_44;
      }
      if ( v49 > (*(_DWORD *)((*(_QWORD *)(v19 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                | (unsigned int)(*(__int64 *)(v19 + 8) >> 1) & 3) )
        goto LABEL_39;
    }
    if ( v42 )
    {
      v42 = 0;
      goto LABEL_39;
    }
LABEL_44:
    v16 = (unsigned int)++v17;
    if ( v17 == v61 )
    {
      v6 = v18;
      goto LABEL_46;
    }
  }
  *((_BYTE *)v20 + 4) = 2;
LABEL_21:
  v28 = (_QWORD *)v18[124];
  v29 = *(_QWORD *)(*(_QWORD *)(*v28 + 96LL) + 8 * v25);
  v30 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v28[6] + 32LL) + 152LL) + 16LL * *(unsigned int *)(v29 + 24));
  if ( v29 + 48 != (*(_QWORD *)(v29 + 48) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v55 = v18;
    v57 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v28[6] + 32LL) + 152LL) + 16LL * *(unsigned int *)(v29 + 24));
    v53 = v18[124];
    v31 = sub_2E312E0(v29, *(_QWORD *)(v29 + 56), 0, 1);
    v30 = v57;
    v18 = v55;
    v32 = v31;
    if ( v29 + 48 != v31 )
    {
      v33 = *(_DWORD *)(v31 + 44) & 0xFFFFFF;
      for ( j = *(_QWORD *)(*(_QWORD *)(v53 + 48) + 32LL);
            (*(_BYTE *)(v31 + 44) & 4) != 0;
            v31 = *(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL )
      {
        ;
      }
      if ( (v33 & 8) != 0 )
      {
        do
          v32 = *(_QWORD *)(v32 + 8);
        while ( (*(_BYTE *)(v32 + 44) & 8) != 0 );
      }
      for ( k = *(_QWORD *)(v32 + 8); k != v31; v31 = *(_QWORD *)(v31 + 8) )
      {
        v36 = *(_WORD *)(v31 + 68);
        if ( (unsigned __int16)(v36 - 14) > 4u && v36 != 24 )
          break;
      }
      v37 = *(_QWORD *)(j + 128);
      v38 = *(_DWORD *)(j + 144);
      if ( v38 )
      {
        v39 = (v38 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
        v40 = (__int64 *)(v37 + 16LL * v39);
        v41 = *v40;
        if ( *v40 == v31 )
          goto LABEL_33;
        v50 = 1;
        while ( v41 != -4096 )
        {
          v51 = v50 + 1;
          v39 = (v38 - 1) & (v50 + v39);
          v40 = (__int64 *)(v37 + 16LL * v39);
          v41 = *v40;
          if ( *v40 == v31 )
            goto LABEL_33;
          v50 = v51;
        }
      }
      v40 = (__int64 *)(v37 + 16LL * v38);
LABEL_33:
      v30 = v40[1];
    }
  }
  if ( *(_DWORD *)((*(_QWORD *)(v19 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24) >= *(_DWORD *)((v30 & 0xFFFFFFFFFFFFFFF8LL) + 24) )
  {
    v42 = 1;
    goto LABEL_36;
  }
  return 0;
}
