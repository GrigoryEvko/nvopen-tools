// Function: sub_20FEBD0
// Address: 0x20febd0
//
void __fastcall sub_20FEBD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  unsigned int v7; // edx
  __int64 v8; // rcx
  __int64 v9; // rsi
  int v10; // r13d
  __int64 *v11; // r15
  unsigned int v12; // r14d
  unsigned int v13; // eax
  bool v14; // r13
  int v15; // esi
  __int64 v16; // r13
  __int64 v17; // rdx
  unsigned __int64 *v18; // rcx
  __int64 v19; // rdx
  int v20; // esi
  __int64 v21; // rax
  __int64 v22; // r13
  __int64 v23; // rax
  unsigned __int64 v24; // r13
  __int64 v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // r15
  __int64 v28; // rax
  unsigned int v29; // r13d
  unsigned int v30; // edx
  __int64 v31; // r14
  unsigned int v32; // eax
  unsigned int v33; // r13d
  __int64 v34; // rax
  unsigned int v35; // edx
  char v36; // si
  __int64 v37; // r14
  unsigned __int64 v38; // rax
  int v39; // esi
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // r8
  _QWORD *v43; // rdx
  unsigned int v44; // r13d
  __int64 v45; // r15
  __int64 v46; // rbx
  int v47; // r9d
  __int64 v48; // rdi
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rax
  unsigned __int64 *v52; // rax
  unsigned int i; // r14d
  __int64 v54; // rax
  unsigned int v55; // edx
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // rcx
  unsigned __int64 v61; // r9
  __int64 v62; // rax
  unsigned __int64 v63; // r9
  __int64 v64; // rdx
  __int64 v65; // r8
  __int64 v66; // [rsp+0h] [rbp-C0h]
  __int64 v67; // [rsp+8h] [rbp-B8h]
  __int64 v68; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v69; // [rsp+8h] [rbp-B8h]
  char v70; // [rsp+10h] [rbp-B0h]
  unsigned int v71; // [rsp+10h] [rbp-B0h]
  __int64 v72; // [rsp+10h] [rbp-B0h]
  unsigned int v73; // [rsp+1Ch] [rbp-A4h]
  unsigned int v74; // [rsp+1Ch] [rbp-A4h]
  __int64 v75; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v76; // [rsp+20h] [rbp-A0h]
  __int64 v77; // [rsp+20h] [rbp-A0h]
  int v78; // [rsp+28h] [rbp-98h]
  __int64 v79; // [rsp+28h] [rbp-98h]
  unsigned int v80; // [rsp+30h] [rbp-90h]
  __int64 *v81; // [rsp+30h] [rbp-90h]
  __int64 v82; // [rsp+38h] [rbp-88h]
  __int64 v83; // [rsp+38h] [rbp-88h]
  unsigned int v86[4]; // [rsp+50h] [rbp-70h] BYREF
  _DWORD v87[4]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v88[10]; // [rsp+70h] [rbp-50h] BYREF

  v6 = a1 + 8;
  v7 = *(_DWORD *)(a1 + 16);
  v82 = a2;
  if ( !v7 || (v8 = *(_QWORD *)(a1 + 8), *(_DWORD *)(v8 + 12) >= *(_DWORD *)(v8 + 8)) )
  {
    v16 = 16LL * *(unsigned int *)(*(_QWORD *)a1 + 192LL);
    sub_3945E40(a1 + 8, *(unsigned int *)(*(_QWORD *)a1 + 192LL));
    ++*(_DWORD *)(*(_QWORD *)(a1 + 8) + v16 + 12);
    v7 = *(_DWORD *)(a1 + 16);
    v8 = *(_QWORD *)(a1 + 8);
  }
  v9 = v8 + 16LL * v7 - 16;
  v10 = *(_DWORD *)(v9 + 12);
  v11 = *(__int64 **)v9;
  if ( !v10
    && (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2 >> 1) & 3) < (*(_DWORD *)((*v11 & 0xFFFFFFFFFFFFFFF8LL)
                                                                                                  + 24)
                                                                                      | (unsigned int)(*v11 >> 1) & 3) )
  {
    v21 = sub_3945DA0(v6, v7 - 1);
    if ( v21 )
    {
      v22 = v21;
      v23 = v21 & 0x3F;
      v24 = v22 & 0xFFFFFFFFFFFFFFC0LL;
      v25 = *(unsigned int *)(a1 + 16);
      v9 = *(_QWORD *)(a1 + 8) + 16 * v25 - 16;
      v26 = (unsigned int)v23;
      v11 = *(__int64 **)v9;
      if ( *(_QWORD *)(v24 + 8 * v23 + 128) != a4 || (v56 = 16 * v23, *(_QWORD *)(v24 + v56 + 8) != a2) )
      {
        v10 = *(_DWORD *)(v9 + 12);
        goto LABEL_5;
      }
      v83 = v56;
      v79 = v26;
      v81 = (__int64 *)(v24 + v56 + 8);
      sub_3945E40(v6, (unsigned int)(v25 - 1));
      v59 = *v11;
      if ( (*(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a3 >> 1) & 3) <= (*(_DWORD *)((*v11 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                           | (unsigned int)(*v11 >> 1)
                                                                                           & 3) )
      {
        v58 = (__int64)v81;
        v57 = v79;
        if ( v11[16] != a4 || a3 != v59 )
        {
          *(_QWORD *)(v24 + 16 * v79 + 8) = a3;
          v19 = *v81;
          v20 = *(_DWORD *)(a1 + 16) - 1;
          goto LABEL_12;
        }
      }
      v82 = *(_QWORD *)(v24 + v83);
      sub_20FD7B0(a1, 0, v57, v58, v59);
    }
    else
    {
      **(_QWORD **)a1 = a2;
    }
    v9 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
    v10 = *(_DWORD *)(v9 + 12);
    v11 = *(__int64 **)v9;
  }
LABEL_5:
  v12 = *(_DWORD *)(v9 + 8);
  v13 = sub_20FCAC0((__int64)v11, (unsigned int *)(v9 + 12), v12, v82, a3, a4);
  v14 = v12 == v10;
  if ( v13 <= 8 )
    goto LABEL_6;
  v27 = (unsigned int)(*(_DWORD *)(a1 + 16) - 1);
  v78 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16 * v27 + 12);
  v28 = sub_3945DA0(v6, v27);
  v66 = v28;
  if ( v28 )
  {
    v30 = 1;
    v80 = 2;
    v88[0] = v28 & 0xFFFFFFFFFFFFFFC0LL;
    v29 = (v28 & 0x3F) + 1;
    v78 += v29;
    v86[0] = v29;
  }
  else
  {
    v80 = 1;
    v29 = 0;
    v30 = 0;
  }
  v31 = *(_QWORD *)(a1 + 8) + 16 * v27;
  v32 = *(_DWORD *)(v31 + 8);
  v73 = v30;
  v75 = v30;
  v86[v30] = v32;
  v33 = v32 + v29;
  v88[v30] = *(_QWORD *)v31;
  v34 = sub_3945FF0(v6, (unsigned int)v27);
  v35 = v73;
  if ( v34 )
  {
    v36 = v34;
    v37 = v80;
    v38 = v34 & 0xFFFFFFFFFFFFFFC0LL;
    v74 = v73 + 2;
    v39 = (v36 & 0x3F) + 1;
    v88[v80] = v38;
    v33 += v39;
    v86[v80] = v39;
    if ( v33 + 1 > 8 * v35 + 16 )
    {
LABEL_20:
      v40 = v74++;
      goto LABEL_21;
    }
LABEL_42:
    v70 = 0;
    v80 = 0;
    goto LABEL_25;
  }
  if ( v33 + 1 <= 8 * (unsigned int)(v80 != 1) + 8 )
  {
    v74 = v80;
    goto LABEL_42;
  }
  if ( v80 != 1 )
  {
    v80 = v73;
    v39 = v86[v75];
    v37 = v75;
    v74 = 2;
    v38 = v88[v75];
    goto LABEL_20;
  }
  v39 = v86[1];
  v38 = v88[1];
  v40 = 1;
  v74 = 2;
  v37 = 1;
LABEL_21:
  v88[v40] = v38;
  v41 = *(_QWORD *)a1;
  v86[v40] = v39;
  v86[v37] = 0;
  v42 = *(_QWORD *)(v41 + 200);
  v43 = *(_QWORD **)v42;
  if ( *(_QWORD *)v42 )
  {
    *(_QWORD *)v42 = *v43;
  }
  else
  {
    v60 = *(_QWORD *)(v42 + 8);
    *(_QWORD *)(v42 + 88) += 192LL;
    if ( ((v60 + 63) & 0xFFFFFFFFFFFFFFC0LL) - v60 + 192 <= *(_QWORD *)(v42 + 16) - v60 )
    {
      v43 = (_QWORD *)((v60 + 63) & 0xFFFFFFFFFFFFFFC0LL);
      *(_QWORD *)(v42 + 8) = v43 + 24;
    }
    else
    {
      v61 = 0x40000000000LL;
      v68 = v42;
      v71 = *(_DWORD *)(v42 + 32);
      if ( v71 >> 7 < 0x1E )
        v61 = 4096LL << (v71 >> 7);
      v76 = v61;
      v62 = malloc(v61);
      v63 = v76;
      v64 = v71;
      v65 = v68;
      if ( !v62 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v65 = v68;
        v63 = v76;
        v62 = 0;
        v64 = *(unsigned int *)(v68 + 32);
      }
      if ( *(_DWORD *)(v65 + 36) <= (unsigned int)v64 )
      {
        v69 = v63;
        v72 = v62;
        v77 = v65;
        sub_16CD150(v65 + 24, (const void *)(v65 + 40), 0, 8, v65, v63);
        v65 = v77;
        v63 = v69;
        v62 = v72;
        v64 = *(unsigned int *)(v77 + 32);
      }
      *(_QWORD *)(*(_QWORD *)(v65 + 24) + 8 * v64) = v62;
      v43 = (_QWORD *)((v62 + 63) & 0xFFFFFFFFFFFFFFC0LL);
      ++*(_DWORD *)(v65 + 32);
      *(_QWORD *)(v65 + 16) = v62 + v63;
      *(_QWORD *)(v65 + 8) = v43 + 24;
    }
    if ( !v43 )
      goto LABEL_24;
  }
  memset(v43, 0, 0xC0u);
LABEL_24:
  v88[v37] = v43;
  v70 = 1;
LABEL_25:
  v67 = sub_39461C0(v74, v33, 8, (unsigned int)v86, (unsigned int)v87, v78, 1);
  sub_20FDD30((__int64)v88, v74, v86, (__int64)v87);
  if ( v66 )
    sub_3945E40(v6, (unsigned int)v27);
  v44 = v27;
  v45 = v6;
  v46 = 0;
  while ( 1 )
  {
    v47 = v87[v46];
    v48 = (unsigned int)(v47 - 1);
    v49 = v88[v46];
    v50 = *(_QWORD *)(v49 + 16 * v48 + 8);
    if ( v80 != (_DWORD)v46 || !v70 )
      break;
    ++v46;
    v44 += (unsigned __int8)sub_20FE8C0(a1, v44, v48 | v49 & 0xFFFFFFFFFFFFFFC0LL, v50);
    if ( v74 == v46 )
      goto LABEL_38;
LABEL_31:
    sub_39460A0(v45, v44);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * v44 + 8) = v47;
  if ( v44 )
  {
    v51 = *(_QWORD *)(a1 + 8) + 16LL * (v44 - 1);
    v52 = (unsigned __int64 *)(*(_QWORD *)v51 + 8LL * *(unsigned int *)(v51 + 12));
    *v52 = v48 | *v52 & 0xFFFFFFFFFFFFFFC0LL;
  }
  ++v46;
  sub_20FCF40(a1, v44, v50);
  if ( v74 != v46 )
    goto LABEL_31;
LABEL_38:
  for ( i = v74 - 1; i != (_DWORD)v67; --i )
    sub_3945E40(v45, v44);
  *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * v44 + 12) = HIDWORD(v67);
  v54 = *(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16;
  v55 = *(_DWORD *)(v54 + 8);
  v14 = *(_DWORD *)(v54 + 12) == v55;
  v13 = sub_20FCAC0(*(_QWORD *)v54, (unsigned int *)(v54 + 12), v55, v82, a3, a4);
LABEL_6:
  v15 = *(_DWORD *)(a1 + 16);
  *(_DWORD *)(*(_QWORD *)(a1 + 8) + 16LL * (unsigned int)(v15 - 1) + 8) = v13;
  if ( v15 == 1 )
  {
    if ( !v14 )
      return;
LABEL_11:
    v19 = a3;
    v20 = *(_DWORD *)(a1 + 16) - 1;
LABEL_12:
    sub_20FCF40(a1, v20, v19);
    return;
  }
  v17 = *(_QWORD *)(a1 + 8) + 16LL * (unsigned int)(v15 - 2);
  v18 = (unsigned __int64 *)(*(_QWORD *)v17 + 8LL * *(unsigned int *)(v17 + 12));
  *v18 = *v18 & 0xFFFFFFFFFFFFFFC0LL | (v13 - 1);
  if ( v14 )
    goto LABEL_11;
}
