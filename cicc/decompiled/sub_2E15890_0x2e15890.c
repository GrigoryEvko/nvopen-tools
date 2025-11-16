// Function: sub_2E15890
// Address: 0x2e15890
//
void __fastcall sub_2E15890(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r9
  __int64 v7; // r15
  char v8; // al
  unsigned int v9; // r14d
  __int64 v10; // r13
  unsigned int v11; // eax
  unsigned __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r10
  _QWORD *v15; // r12
  char v16; // si
  _QWORD *v17; // rax
  _QWORD *v18; // r12
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rax
  unsigned int v23; // r14d
  __int16 *v24; // r13
  __int64 v25; // r12
  _QWORD *v26; // rax
  _QWORD *v27; // rax
  _QWORD *v28; // r8
  char v29; // dl
  unsigned int v30; // eax
  __int64 v31; // rcx
  __int64 *v32; // r12
  __int64 v33; // rax
  unsigned __int16 v34; // ax
  __int64 v35; // rdx
  __int64 v36; // rbx
  __int64 *v37; // rax
  __int64 v38; // r14
  __int64 v39; // r13
  unsigned __int16 v40; // ax
  __int64 *v41; // rax
  __int64 v42; // r13
  unsigned __int64 v43; // rax
  __int64 v44; // r15
  _QWORD *v45; // rbx
  unsigned __int64 v46; // r12
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // r10
  __int64 v50; // rdx
  _QWORD *v51; // rax
  __int64 v52; // rdi
  __int64 v53; // r10
  __int64 v54; // rdx
  unsigned __int64 *v55; // r8
  __int64 v56; // rcx
  __int64 *v57; // rsi
  __int64 v58; // r10
  __int64 v59; // r9
  _QWORD *v60; // rax
  _QWORD *v61; // rsi
  __int64 v62; // rax
  __int64 v63; // [rsp+8h] [rbp-68h]
  __int64 v64; // [rsp+10h] [rbp-60h]
  _QWORD *v65; // [rsp+18h] [rbp-58h]
  __int64 v66; // [rsp+18h] [rbp-58h]
  char v67; // [rsp+20h] [rbp-50h]
  __int64 v68; // [rsp+20h] [rbp-50h]
  __int64 v69; // [rsp+20h] [rbp-50h]
  __int64 v70; // [rsp+20h] [rbp-50h]
  __int64 v71; // [rsp+20h] [rbp-50h]
  char v72; // [rsp+2Fh] [rbp-41h]
  __int64 v73; // [rsp+30h] [rbp-40h]
  __int64 v74; // [rsp+30h] [rbp-40h]
  __int64 v75; // [rsp+30h] [rbp-40h]
  _QWORD **v76; // [rsp+30h] [rbp-40h]
  __int64 v77; // [rsp+30h] [rbp-40h]
  __int64 v78; // [rsp+30h] [rbp-40h]
  __int64 v79; // [rsp+30h] [rbp-40h]
  __int64 v80; // [rsp+38h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 32);
  v72 = 0;
  v80 = v5 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  if ( v80 == v5 )
    return;
  v7 = *(_QWORD *)(a2 + 32);
  do
  {
    if ( *(_BYTE *)v7 == 12 )
    {
      v72 = 1;
      goto LABEL_21;
    }
    if ( *(_BYTE *)v7 )
      goto LABEL_21;
    v8 = *(_BYTE *)(v7 + 3);
    if ( (v8 & 0x10) == 0 )
    {
      if ( (*(_BYTE *)(v7 + 4) & 1) != 0 || (*(_BYTE *)(v7 + 4) & 2) != 0 )
        goto LABEL_21;
      *(_BYTE *)(v7 + 3) = v8 & 0xBF;
    }
    v9 = *(_DWORD *)(v7 + 8);
    if ( !v9 )
      goto LABEL_21;
    if ( (v9 & 0x80000000) == 0 )
    {
      v19 = *(_QWORD *)(a1 + 16);
      v20 = *(_QWORD *)(v19 + 8);
      v21 = *(_QWORD *)(v19 + 56);
      v22 = *(_DWORD *)(v20 + 24LL * v9 + 16) >> 12;
      v23 = *(_DWORD *)(v20 + 24LL * v9 + 16) & 0xFFF;
      v24 = (__int16 *)(v21 + 2 * v22);
      while ( 1 )
      {
        if ( !v24 )
          goto LABEL_21;
        if ( !*(_BYTE *)(a1 + 136) || (unsigned __int8)sub_2EBFC90(*(_QWORD *)(a1 + 8), v23) )
        {
          v25 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 424LL) + 8LL * v23);
        }
        else
        {
          a5 = *(_QWORD *)a1;
          v25 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 424LL) + 8LL * v23);
          if ( v25 )
            goto LABEL_30;
          v65 = *(_QWORD **)a1;
          v67 = qword_501EA48[8];
          v27 = (_QWORD *)sub_22077B0(0x68u);
          v28 = v65;
          v25 = (__int64)v27;
          if ( v27 )
          {
            *v27 = v27 + 2;
            v27[1] = 0x200000000LL;
            v27[8] = v27 + 10;
            v27[9] = 0x200000000LL;
            if ( v67 )
            {
              v62 = sub_22077B0(0x30u);
              v28 = v65;
              if ( v62 )
              {
                *(_DWORD *)(v62 + 8) = 0;
                *(_QWORD *)(v62 + 16) = 0;
                *(_QWORD *)(v62 + 24) = v62 + 8;
                *(_QWORD *)(v62 + 32) = v62 + 8;
                *(_QWORD *)(v62 + 40) = 0;
              }
              *(_QWORD *)(v25 + 96) = v62;
            }
            else
            {
              v27[12] = 0;
            }
          }
          *(_QWORD *)(v28[53] + 8LL * v23) = v25;
          sub_2E11710(v28, v25, v23);
        }
        if ( !v25 )
          goto LABEL_35;
LABEL_30:
        if ( *(_BYTE *)(a1 + 68) )
        {
          v26 = *(_QWORD **)(a1 + 48);
          v20 = *(unsigned int *)(a1 + 60);
          v21 = (__int64)&v26[v20];
          if ( v26 != (_QWORD *)v21 )
          {
            while ( *v26 != v25 )
            {
              if ( (_QWORD *)v21 == ++v26 )
                goto LABEL_46;
            }
            goto LABEL_35;
          }
LABEL_46:
          if ( (unsigned int)v20 < *(_DWORD *)(a1 + 56) )
          {
            *(_DWORD *)(a1 + 60) = v20 + 1;
            *(_QWORD *)v21 = v25;
            ++*(_QWORD *)(a1 + 40);
LABEL_44:
            if ( *(_DWORD *)((*(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 24) >= *(_DWORD *)((*(_QWORD *)(a1 + 32)
                                                                                               & 0xFFFFFFFFFFFFFFF8LL)
                                                                                              + 24) )
              sub_2E14970((_QWORD *)a1, v25, v23, 0, 0);
            else
              sub_2E13DC0(a1, v25);
            goto LABEL_35;
          }
        }
        sub_C8CC70(a1 + 40, v25, v21, v20, a5, v5);
        if ( v29 )
          goto LABEL_44;
LABEL_35:
        v21 = (unsigned int)*v24++;
        v23 += v21;
        if ( !(_WORD)v21 )
          goto LABEL_21;
      }
    }
    v10 = *(_QWORD *)a1;
    v11 = v9 & 0x7FFFFFFF;
    v12 = *(unsigned int *)(*(_QWORD *)a1 + 160LL);
    v13 = v9 & 0x7FFFFFFF;
    if ( (v9 & 0x7FFFFFFF) < (unsigned int)v12 )
    {
      v14 = *(_QWORD *)(*(_QWORD *)(v10 + 152) + 8LL * v11);
      if ( v14 )
        goto LABEL_13;
    }
    v30 = v11 + 1;
    if ( (unsigned int)v12 < v30 && v30 != v12 )
    {
      if ( v30 >= v12 )
      {
        v58 = *(_QWORD *)(v10 + 168);
        v59 = v30 - v12;
        if ( v30 > (unsigned __int64)*(unsigned int *)(v10 + 164) )
        {
          v71 = v30 - v12;
          v79 = *(_QWORD *)(v10 + 168);
          sub_C8D5F0(v10 + 152, (const void *)(v10 + 168), v30, 8u, v30, v59);
          v12 = *(unsigned int *)(v10 + 160);
          v59 = v71;
          v58 = v79;
        }
        v31 = *(_QWORD *)(v10 + 152);
        v60 = (_QWORD *)(v31 + 8 * v12);
        v61 = &v60[v59];
        if ( v60 != v61 )
        {
          do
            *v60++ = v58;
          while ( v61 != v60 );
          LODWORD(v12) = *(_DWORD *)(v10 + 160);
          v31 = *(_QWORD *)(v10 + 152);
        }
        *(_DWORD *)(v10 + 160) = v59 + v12;
        goto LABEL_51;
      }
      *(_DWORD *)(v10 + 160) = v30;
    }
    v31 = *(_QWORD *)(v10 + 152);
LABEL_51:
    v32 = (__int64 *)(v31 + 8LL * (v9 & 0x7FFFFFFF));
    v33 = sub_2E10F30(v9);
    *v32 = v33;
    v74 = v33;
    sub_2E11E80((_QWORD *)v10, v33);
    v14 = v74;
LABEL_13:
    v15 = *(_QWORD **)(v14 + 104);
    v73 = a1 + 40;
    if ( !v15 )
      goto LABEL_14;
    v40 = (*(_DWORD *)v7 >> 8) & 0xFFF;
    if ( v40 )
    {
      v41 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL) + 16LL * v40);
      v42 = *v41;
      v43 = v41[1];
    }
    else
    {
      v69 = v14;
      v47 = sub_2EBF1E0(*(_QWORD *)(a1 + 8), v9, v12, v13, a5, v5);
      v14 = v69;
      v15 = *(_QWORD **)(v69 + 104);
      if ( !v15 )
      {
LABEL_14:
        v16 = *(_BYTE *)(a1 + 68);
        goto LABEL_15;
      }
      v42 = v47;
      v43 = v12;
    }
    v63 = v14;
    v16 = *(_BYTE *)(a1 + 68);
    v64 = v7;
    v44 = a1;
    v45 = v15;
    v46 = v43;
    do
    {
      v12 = v46 & v45[15] | v42 & v45[14];
      if ( !v12 )
        goto LABEL_65;
      v13 = v45[14];
      a5 = v45[15];
      if ( !v16 )
        goto LABEL_82;
      v51 = *(_QWORD **)(v44 + 48);
      v52 = *(unsigned int *)(v44 + 60);
      v12 = (unsigned __int64)&v51[v52];
      if ( v51 == (_QWORD *)v12 )
      {
LABEL_77:
        if ( (unsigned int)v52 >= *(_DWORD *)(v44 + 56) )
        {
LABEL_82:
          v66 = v45[15];
          v70 = v45[14];
          sub_C8CC70(v73, (__int64)v45, v12, v13, a5, v5);
          v16 = *(_BYTE *)(v44 + 68);
          v13 = v70;
          a5 = v66;
          if ( !(_BYTE)v12 )
            goto LABEL_65;
        }
        else
        {
          *(_DWORD *)(v44 + 60) = v52 + 1;
          *(_QWORD *)v12 = v45;
          ++*(_QWORD *)(v44 + 40);
        }
        if ( *(_DWORD *)((*(_QWORD *)(v44 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 24) >= *(_DWORD *)((*(_QWORD *)(v44 + 32)
                                                                                            & 0xFFFFFFFFFFFFFFF8LL)
                                                                                           + 24) )
          sub_2E14970((_QWORD *)v44, (__int64)v45, v9, v13, a5);
        else
          sub_2E13DC0(v44, (__int64)v45);
        v16 = *(_BYTE *)(v44 + 68);
        goto LABEL_65;
      }
      while ( v45 != (_QWORD *)*v51 )
      {
        if ( (_QWORD *)v12 == ++v51 )
          goto LABEL_77;
      }
LABEL_65:
      v45 = (_QWORD *)v45[13];
    }
    while ( v45 );
    a1 = v44;
    v14 = v63;
    v7 = v64;
LABEL_15:
    if ( v16 )
    {
      v17 = *(_QWORD **)(a1 + 48);
      v13 = *(unsigned int *)(a1 + 60);
      v12 = (unsigned __int64)&v17[v13];
      if ( v17 == (_QWORD *)v12 )
      {
LABEL_52:
        if ( (unsigned int)v13 >= *(_DWORD *)(a1 + 56) )
          goto LABEL_53;
        *(_DWORD *)(a1 + 60) = v13 + 1;
        *(_QWORD *)v12 = v14;
        ++*(_QWORD *)(a1 + 40);
LABEL_54:
        if ( *(_DWORD *)((*(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 24) < *(_DWORD *)((*(_QWORD *)(a1 + 32)
                                                                                          & 0xFFFFFFFFFFFFFFF8LL)
                                                                                         + 24) )
        {
          v75 = v14;
          sub_2E13DC0(a1, v14);
          v14 = v75;
          v18 = *(_QWORD **)(v75 + 104);
          if ( v18 )
            goto LABEL_56;
          goto LABEL_21;
        }
        v77 = v14;
        sub_2E14970((_QWORD *)a1, v14, v9, 0, 0);
        v14 = v77;
      }
      else
      {
        while ( v14 != *v17 )
        {
          if ( (_QWORD *)v12 == ++v17 )
            goto LABEL_52;
        }
      }
    }
    else
    {
LABEL_53:
      v68 = v14;
      sub_C8CC70(v73, v14, v12, v13, a5, v5);
      v14 = v68;
      if ( (_BYTE)v12 )
        goto LABEL_54;
    }
    v18 = *(_QWORD **)(v14 + 104);
    if ( v18 )
    {
LABEL_56:
      v34 = (*(_DWORD *)v7 >> 8) & 0xFFF;
      if ( v34 )
      {
        v35 = *(_QWORD *)(a1 + 16);
        v76 = (_QWORD **)a1;
        v36 = v14;
        v37 = (__int64 *)(*(_QWORD *)(v35 + 272) + 16LL * v34);
        v38 = *v37;
        v39 = v37[1];
LABEL_58:
        while ( !(v38 & v18[14] | v39 & v18[15]) || sub_2E0A1A0(v36, (__int64)v18) )
        {
          v18 = (_QWORD *)v18[13];
          if ( !v18 )
          {
            a1 = (__int64)v76;
            goto LABEL_21;
          }
        }
        v53 = v36;
        a1 = (__int64)v76;
        *(_DWORD *)(v53 + 72) = 0;
        *(_DWORD *)(v53 + 8) = 0;
        sub_2E15850(*v76, v53);
      }
      else
      {
        v78 = v14;
        v48 = sub_2EBF1E0(*(_QWORD *)(a1 + 8), v9, v12, v13, a5, v5);
        v49 = v78;
        v38 = v48;
        v39 = v50;
        v18 = *(_QWORD **)(v78 + 104);
        if ( v18 )
        {
          v76 = (_QWORD **)a1;
          v36 = v49;
          goto LABEL_58;
        }
      }
    }
LABEL_21:
    v7 += 40;
  }
  while ( v80 != v7 );
  if ( v72 )
  {
    v54 = *(unsigned int *)(*(_QWORD *)a1 + 192LL);
    v55 = *(unsigned __int64 **)(*(_QWORD *)a1 + 184LL);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 192LL) )
    {
      do
      {
        while ( 1 )
        {
          v56 = v54 >> 1;
          v57 = (__int64 *)&v55[v54 >> 1];
          if ( (*(_DWORD *)((*v57 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v57 >> 1) & 3) >= (*(_DWORD *)((*(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*(__int64 *)(a1 + 24) >> 1) & 3)) )
            break;
          v55 = (unsigned __int64 *)(v57 + 1);
          v54 = v54 - v56 - 1;
          if ( v54 <= 0 )
            goto LABEL_92;
        }
        v54 >>= 1;
      }
      while ( v56 > 0 );
    }
LABEL_92:
    *v55 = *(_QWORD *)(a1 + 32) & 0xFFFFFFFFFFFFFFF8LL | 4;
  }
}
