// Function: sub_2FBCF60
// Address: 0x2fbcf60
//
void __fastcall sub_2FBCF60(unsigned int *a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v4; // r8
  __int64 *v6; // r13
  __int64 v7; // r12
  unsigned int v8; // edx
  __int64 v9; // rcx
  __int64 v10; // rsi
  int v11; // ebx
  __int64 v12; // r15
  unsigned int v13; // eax
  bool v14; // bl
  int v15; // esi
  __int64 v16; // rbx
  __int64 v17; // rdx
  unsigned __int64 *v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rcx
  __int64 v25; // rbx
  __int64 v26; // rax
  unsigned int v27; // r9d
  __int64 v28; // r15
  unsigned int v29; // eax
  unsigned int v30; // ecx
  __int64 v31; // rdx
  unsigned int v32; // r10d
  __int64 v33; // rax
  __int64 v34; // r8
  int v35; // r10d
  unsigned int v36; // r9d
  char v37; // dl
  unsigned __int64 v38; // rax
  int v39; // edx
  __int64 v40; // rcx
  __int64 v41; // rax
  __int64 v42; // rax
  _QWORD *v43; // rdx
  _QWORD *v44; // rax
  unsigned int v45; // r14d
  __int64 v46; // rbx
  int v47; // r9d
  __int64 v48; // rdi
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rax
  unsigned __int64 *v52; // rax
  unsigned int v53; // r12d
  __int64 v54; // r8
  __int64 v55; // rax
  unsigned int v56; // edx
  __int64 v57; // rax
  __int64 v58; // rbx
  bool v59; // al
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // rdx
  __int64 v63; // rsi
  unsigned __int64 v64; // rcx
  __int64 v65; // rax
  int v66; // [rsp+4h] [rbp-BCh]
  __int64 v67; // [rsp+10h] [rbp-B0h]
  unsigned int v68; // [rsp+18h] [rbp-A8h]
  char v69; // [rsp+18h] [rbp-A8h]
  __int64 v70; // [rsp+18h] [rbp-A8h]
  unsigned int v71; // [rsp+20h] [rbp-A0h]
  int v72; // [rsp+20h] [rbp-A0h]
  int v73; // [rsp+28h] [rbp-98h]
  unsigned int v74; // [rsp+28h] [rbp-98h]
  __int64 v75; // [rsp+30h] [rbp-90h]
  unsigned int v76; // [rsp+30h] [rbp-90h]
  __int64 v77; // [rsp+30h] [rbp-90h]
  unsigned int v78; // [rsp+38h] [rbp-88h]
  __int64 v79; // [rsp+38h] [rbp-88h]
  unsigned int v80; // [rsp+38h] [rbp-88h]
  unsigned int v81; // [rsp+38h] [rbp-88h]
  __int64 v82; // [rsp+38h] [rbp-88h]
  __int64 *v83; // [rsp+38h] [rbp-88h]
  __int64 v84; // [rsp+40h] [rbp-80h]
  unsigned __int64 v85; // [rsp+40h] [rbp-80h]
  __int64 v86; // [rsp+48h] [rbp-78h] BYREF
  unsigned int v87[4]; // [rsp+50h] [rbp-70h] BYREF
  _DWORD v88[4]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v89[10]; // [rsp+70h] [rbp-50h] BYREF

  v4 = a2;
  v6 = (__int64 *)(a1 + 2);
  v7 = (__int64)a1;
  v86 = a3;
  v8 = a1[4];
  v84 = a2;
  if ( !v8 || (v9 = *((_QWORD *)a1 + 1), *(_DWORD *)(v9 + 12) >= *(_DWORD *)(v9 + 8)) )
  {
    v16 = 16LL * *(unsigned int *)(*(_QWORD *)a1 + 184LL);
    sub_F03AD0(a1 + 2, *(_DWORD *)(*(_QWORD *)a1 + 184LL));
    v4 = a2;
    ++*(_DWORD *)(*((_QWORD *)a1 + 1) + v16 + 12);
    v8 = a1[4];
    v9 = *((_QWORD *)a1 + 1);
  }
  v10 = v9 + 16LL * v8 - 16;
  v11 = *(_DWORD *)(v10 + 12);
  v12 = *(_QWORD *)v10;
  if ( !v11
    && (*(_DWORD *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v4 >> 1) & 3) < (*(_DWORD *)((*(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL)
                                                                                                  + 24)
                                                                                      | (unsigned int)(*(__int64 *)v12 >> 1)
                                                                                      & 3) )
  {
    v79 = v4;
    v19 = sub_F03A30(v6, v8 - 1);
    if ( v19 )
    {
      v20 = v19;
      v21 = v19 & 0x3F;
      v22 = v20 & 0xFFFFFFFFFFFFFFC0LL;
      v23 = a1[4];
      v10 = *(_QWORD *)(v7 + 8) + 16 * v23 - 16;
      v24 = (unsigned int)v21;
      v12 = *(_QWORD *)v10;
      if ( *(_DWORD *)(v22 + 4 * v21 + 144) != a4 || (v57 = 16 * v21, v58 = v57, *(_QWORD *)(v22 + v57 + 8) != v79) )
      {
        v11 = *(_DWORD *)(v10 + 12);
        goto LABEL_5;
      }
      v85 = v22;
      v77 = v24;
      v83 = (__int64 *)(v22 + v57 + 8);
      sub_F03AD0((unsigned int *)v6, v23 - 1);
      v59 = sub_2DF8330(&v86, *(_QWORD *)v12);
      v62 = v85;
      if ( v59 )
      {
        v60 = v77;
        if ( *(_DWORD *)(v12 + 144) != a4 || *(_QWORD *)v12 != v86 )
        {
          *(_QWORD *)(v85 + 16 * v77 + 8) = v86;
          sub_2FB6C00(v7, *(_DWORD *)(v7 + 16) - 1, *v83);
          return;
        }
      }
      v84 = *(_QWORD *)(v85 + v58);
      sub_2FB6E70(v7, 0, v62, v60, v61);
    }
    else
    {
      **(_QWORD **)a1 = v79;
    }
    v10 = *(_QWORD *)(v7 + 8) + 16LL * *(unsigned int *)(v7 + 16) - 16;
    v11 = *(_DWORD *)(v10 + 12);
    v12 = *(_QWORD *)v10;
  }
LABEL_5:
  v78 = *(_DWORD *)(v10 + 8);
  v13 = sub_2FB3800(v12, (unsigned int *)(v10 + 12), v78, v84, v86, a4);
  v14 = v78 == v11;
  if ( v13 <= 9 )
    goto LABEL_6;
  v25 = (unsigned int)(*(_DWORD *)(v7 + 16) - 1);
  v80 = *(_DWORD *)(*(_QWORD *)(v7 + 8) + 16 * v25 + 12);
  v26 = sub_F03A30(v6, *(_DWORD *)(v7 + 16) - 1);
  v27 = v80;
  v28 = v26;
  if ( v26 )
  {
    v81 = 2;
    v29 = (v26 & 0x3F) + 1;
    v89[0] = v28 & 0xFFFFFFFFFFFFFFC0LL;
    v30 = 1;
    v87[0] = v29;
    v27 += v29;
  }
  else
  {
    v81 = 1;
    v29 = 0;
    v30 = 0;
  }
  v31 = *(_QWORD *)(v7 + 8) + 16 * v25;
  v32 = *(_DWORD *)(v31 + 8);
  v68 = v27;
  v71 = v30;
  v87[v30] = v32;
  v73 = v29 + v32;
  v89[v30] = *(_QWORD *)v31;
  v75 = v30;
  v33 = sub_F03C90(v6, v25);
  v34 = v75;
  v35 = v73;
  v36 = v68;
  if ( v33 )
  {
    v37 = v33;
    v34 = v81;
    v38 = v33 & 0xFFFFFFFFFFFFFFC0LL;
    v76 = v71 + 2;
    v39 = (v37 & 0x3F) + 1;
    v89[v81] = v38;
    v35 = v39 + v73;
    v87[v81] = v39;
    if ( v39 + v73 + 1 > 9 * v71 + 18 )
    {
LABEL_19:
      v40 = v76++;
      goto LABEL_20;
    }
LABEL_44:
    v69 = 0;
    v81 = 0;
    goto LABEL_25;
  }
  if ( v73 + 1 <= 9 * (unsigned int)(v81 != 1) + 9 )
  {
    v76 = v81;
    goto LABEL_44;
  }
  if ( v81 != 1 )
  {
    v39 = v87[v75];
    v38 = v89[v75];
    v81 = v71;
    v76 = 2;
    goto LABEL_19;
  }
  v39 = v87[1];
  v38 = v89[1];
  v40 = 1;
  v76 = 2;
  v34 = 1;
LABEL_20:
  v89[v40] = v38;
  v41 = *(_QWORD *)v7;
  v87[v40] = v39;
  v87[v34] = 0;
  v42 = *(_QWORD *)(v41 + 192);
  v43 = *(_QWORD **)v42;
  if ( *(_QWORD *)v42 )
  {
    *(_QWORD *)v42 = *v43;
    goto LABEL_22;
  }
  v63 = *(_QWORD *)(v42 + 8);
  *(_QWORD *)(v42 + 88) += 192LL;
  v64 = (v63 + 63) & 0xFFFFFFFFFFFFFFC0LL;
  if ( *(_QWORD *)(v42 + 16) < v64 + 192 || !v63 )
  {
    v70 = v34;
    v72 = v35;
    v74 = v36;
    v65 = sub_9D1E70(v42 + 8, 192, 192, 6);
    v34 = v70;
    v35 = v72;
    v36 = v74;
    v43 = (_QWORD *)v65;
LABEL_22:
    memset(v43, 0, 0xB8u);
    v44 = v43;
    do
    {
      *v44 = 0;
      v44 += 2;
      *(v44 - 1) = 0;
    }
    while ( v43 + 18 != v44 );
    goto LABEL_24;
  }
  *(_QWORD *)(v42 + 8) = v64 + 192;
  if ( v64 )
  {
    v43 = (_QWORD *)((v63 + 63) & 0xFFFFFFFFFFFFFFC0LL);
    goto LABEL_22;
  }
LABEL_24:
  v89[v34] = v43;
  v69 = 1;
LABEL_25:
  v67 = sub_F03E60(v76, v35, 9, (__int64)v87, (__int64)v88, v36, 1u);
  sub_2FB89C0((__int64)v89, v76, v87, (__int64)v88);
  if ( v28 )
    sub_F03AD0((unsigned int *)v6, v25);
  v66 = a4;
  v45 = v25;
  v46 = 0;
  while ( 1 )
  {
    v47 = v88[v46];
    v48 = (unsigned int)(v47 - 1);
    v49 = v89[v46];
    v50 = *(_QWORD *)(v49 + 16 * v48 + 8);
    if ( v81 != (_DWORD)v46 || !v69 )
      break;
    ++v46;
    v45 += (unsigned __int8)sub_2FBCCC0(v7, v45, v48 | v49 & 0xFFFFFFFFFFFFFFC0LL, v50);
    if ( v76 == v46 )
      goto LABEL_38;
LABEL_31:
    sub_F03D40(v6, v45);
  }
  *(_DWORD *)(*(_QWORD *)(v7 + 8) + 16LL * v45 + 8) = v47;
  if ( v45 )
  {
    v51 = *(_QWORD *)(v7 + 8) + 16LL * (v45 - 1);
    v52 = (unsigned __int64 *)(*(_QWORD *)v51 + 8LL * *(unsigned int *)(v51 + 12));
    *v52 = v48 | *v52 & 0xFFFFFFFFFFFFFFC0LL;
  }
  ++v46;
  sub_2FB6C00(v7, v45, v50);
  if ( v76 != v46 )
    goto LABEL_31;
LABEL_38:
  if ( v76 - 1 != (_DWORD)v67 )
  {
    v82 = v7;
    v53 = v76 - 1;
    do
    {
      --v53;
      sub_F03AD0((unsigned int *)v6, v45);
    }
    while ( v53 != (_DWORD)v67 );
    v7 = v82;
  }
  v54 = v86;
  *(_DWORD *)(*(_QWORD *)(v7 + 8) + 16LL * v45 + 12) = HIDWORD(v67);
  v55 = *(_QWORD *)(v7 + 8) + 16LL * *(unsigned int *)(v7 + 16) - 16;
  v56 = *(_DWORD *)(v55 + 8);
  v14 = *(_DWORD *)(v55 + 12) == v56;
  v13 = sub_2FB3800(*(_QWORD *)v55, (unsigned int *)(v55 + 12), v56, v84, v54, v66);
LABEL_6:
  v15 = *(_DWORD *)(v7 + 16);
  *(_DWORD *)(*(_QWORD *)(v7 + 8) + 16LL * (unsigned int)(v15 - 1) + 8) = v13;
  if ( v15 == 1 )
  {
    if ( !v14 )
      return;
LABEL_11:
    sub_2FB6C00(v7, *(_DWORD *)(v7 + 16) - 1, v86);
    return;
  }
  v17 = *(_QWORD *)(v7 + 8) + 16LL * (unsigned int)(v15 - 2);
  v18 = (unsigned __int64 *)(*(_QWORD *)v17 + 8LL * *(unsigned int *)(v17 + 12));
  *v18 = *v18 & 0xFFFFFFFFFFFFFFC0LL | (v13 - 1);
  if ( v14 )
    goto LABEL_11;
}
