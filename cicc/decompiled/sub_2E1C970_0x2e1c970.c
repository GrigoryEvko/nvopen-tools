// Function: sub_2E1C970
// Address: 0x2e1c970
//
void __fastcall sub_2E1C970(unsigned int *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v6; // rbx
  unsigned int v7; // edx
  __int64 v8; // rcx
  __int64 v9; // rsi
  int v10; // r14d
  __int64 *v11; // r13
  unsigned int v12; // r15d
  unsigned int v13; // eax
  bool v14; // r14
  unsigned int v15; // esi
  __int64 v16; // r14
  __int64 v17; // rdx
  unsigned __int64 *v18; // rcx
  __int64 v19; // rdx
  int v20; // esi
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 v23; // rax
  unsigned __int64 v24; // r14
  __int64 v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // r13
  __int64 v28; // rax
  unsigned int v29; // r9d
  unsigned int v30; // eax
  unsigned int v31; // edx
  __int64 v32; // r15
  unsigned int v33; // r14d
  int v34; // r14d
  __int64 v35; // rax
  unsigned int v36; // edx
  unsigned int v37; // r9d
  char v38; // si
  __int64 v39; // r15
  unsigned __int64 v40; // rax
  int v41; // esi
  __int64 v42; // rdx
  __int64 v43; // rax
  __int64 v44; // rax
  _QWORD *v45; // rdx
  _QWORD *v46; // rax
  unsigned int v47; // r14d
  __int64 *v48; // r13
  __int64 v49; // rbx
  int v50; // r9d
  __int64 v51; // rdi
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // rax
  unsigned __int64 *v55; // rax
  unsigned int i; // r15d
  __int64 v57; // rax
  unsigned int v58; // edx
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // rsi
  unsigned __int64 v64; // rcx
  __int64 v65; // rax
  __int64 v66; // [rsp+10h] [rbp-B0h]
  unsigned int v67; // [rsp+18h] [rbp-A8h]
  char v68; // [rsp+18h] [rbp-A8h]
  unsigned int v69; // [rsp+1Ch] [rbp-A4h]
  unsigned int v70; // [rsp+1Ch] [rbp-A4h]
  __int64 v71; // [rsp+20h] [rbp-A0h]
  __int64 v72; // [rsp+28h] [rbp-98h]
  __int64 v73; // [rsp+28h] [rbp-98h]
  unsigned int v74; // [rsp+30h] [rbp-90h]
  unsigned int v75; // [rsp+30h] [rbp-90h]
  __int64 *v76; // [rsp+30h] [rbp-90h]
  __int64 v77; // [rsp+38h] [rbp-88h]
  __int64 v78; // [rsp+38h] [rbp-88h]
  unsigned int v81[4]; // [rsp+50h] [rbp-70h] BYREF
  _DWORD v82[4]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v83[10]; // [rsp+70h] [rbp-50h] BYREF

  v6 = (__int64 *)(a1 + 2);
  v7 = a1[4];
  v77 = a2;
  if ( !v7 || (v8 = *((_QWORD *)a1 + 1), *(_DWORD *)(v8 + 12) >= *(_DWORD *)(v8 + 8)) )
  {
    v16 = 16LL * *(unsigned int *)(*(_QWORD *)a1 + 192LL);
    sub_F03AD0(a1 + 2, *(_DWORD *)(*(_QWORD *)a1 + 192LL));
    ++*(_DWORD *)(*((_QWORD *)a1 + 1) + v16 + 12);
    v7 = a1[4];
    v8 = *((_QWORD *)a1 + 1);
  }
  v9 = v8 + 16LL * v7 - 16;
  v10 = *(_DWORD *)(v9 + 12);
  v11 = *(__int64 **)v9;
  if ( !v10
    && (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2 >> 1) & 3) < (*(_DWORD *)((*v11 & 0xFFFFFFFFFFFFFFF8LL)
                                                                                                  + 24)
                                                                                      | (unsigned int)(*v11 >> 1) & 3) )
  {
    v21 = sub_F03A30(v6, v7 - 1);
    if ( v21 )
    {
      v22 = v21;
      v23 = v21 & 0x3F;
      v24 = v22 & 0xFFFFFFFFFFFFFFC0LL;
      v25 = a1[4];
      v9 = *((_QWORD *)a1 + 1) + 16 * v25 - 16;
      v26 = (unsigned int)v23;
      v11 = *(__int64 **)v9;
      if ( *(_QWORD *)(v24 + 8 * v23 + 128) != a4 || (v59 = 16 * v23, *(_QWORD *)(v24 + v59 + 8) != a2) )
      {
        v10 = *(_DWORD *)(v9 + 12);
        goto LABEL_5;
      }
      v78 = v59;
      v73 = v26;
      v76 = (__int64 *)(v24 + v59 + 8);
      sub_F03AD0((unsigned int *)v6, v25 - 1);
      v62 = *v11;
      if ( (*(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a3 >> 1) & 3) <= (*(_DWORD *)((*v11 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                           | (unsigned int)(*v11 >> 1)
                                                                                           & 3) )
      {
        v61 = (__int64)v76;
        v60 = v73;
        if ( v11[16] != a4 || a3 != v62 )
        {
          *(_QWORD *)(v24 + 16 * v73 + 8) = a3;
          v19 = *v76;
          v20 = a1[4] - 1;
          goto LABEL_12;
        }
      }
      v77 = *(_QWORD *)(v24 + v78);
      sub_2E1B3E0((__int64)a1, 0, v60, v61, v62);
    }
    else
    {
      **(_QWORD **)a1 = a2;
    }
    v9 = *((_QWORD *)a1 + 1) + 16LL * a1[4] - 16;
    v10 = *(_DWORD *)(v9 + 12);
    v11 = *(__int64 **)v9;
  }
LABEL_5:
  v12 = *(_DWORD *)(v9 + 8);
  v13 = sub_2E1A1C0((__int64)v11, (unsigned int *)(v9 + 12), v12, v77, a3, a4);
  v14 = v12 == v10;
  if ( v13 <= 8 )
    goto LABEL_6;
  v27 = a1[4] - 1;
  v74 = *(_DWORD *)(*((_QWORD *)a1 + 1) + 16 * v27 + 12);
  v28 = sub_F03A30(v6, a1[4] - 1);
  v29 = v74;
  v72 = v28;
  if ( v28 )
  {
    v31 = 1;
    v75 = 2;
    v83[0] = v28 & 0xFFFFFFFFFFFFFFC0LL;
    v30 = (v28 & 0x3F) + 1;
    v81[0] = v30;
    v29 += v30;
  }
  else
  {
    v75 = 1;
    v30 = 0;
    v31 = 0;
  }
  v32 = *((_QWORD *)a1 + 1) + 16 * v27;
  v33 = *(_DWORD *)(v32 + 8);
  v67 = v29;
  v69 = v31;
  v81[v31] = v33;
  v34 = v30 + v33;
  v71 = v31;
  v83[v31] = *(_QWORD *)v32;
  v35 = sub_F03C90(v6, v27);
  v36 = v69;
  v37 = v67;
  if ( v35 )
  {
    v38 = v35;
    v39 = v75;
    v40 = v35 & 0xFFFFFFFFFFFFFFC0LL;
    v70 = v69 + 2;
    v41 = (v38 & 0x3F) + 1;
    v83[v75] = v40;
    v34 += v41;
    v81[v75] = v41;
    if ( v34 + 1 > 8 * v36 + 16 )
    {
LABEL_20:
      v42 = v70++;
      goto LABEL_21;
    }
LABEL_43:
    v68 = 0;
    v75 = 0;
    goto LABEL_26;
  }
  if ( v34 + 1 <= 8 * (unsigned int)(v75 != 1) + 8 )
  {
    v70 = v75;
    goto LABEL_43;
  }
  if ( v75 != 1 )
  {
    v75 = v69;
    v41 = v81[v71];
    v39 = v71;
    v70 = 2;
    v40 = v83[v71];
    goto LABEL_20;
  }
  v41 = v81[1];
  v40 = v83[1];
  v42 = 1;
  v70 = 2;
  v39 = 1;
LABEL_21:
  v83[v42] = v40;
  v43 = *(_QWORD *)a1;
  v81[v42] = v41;
  v81[v39] = 0;
  v44 = *(_QWORD *)(v43 + 200);
  v45 = *(_QWORD **)v44;
  if ( *(_QWORD *)v44 )
  {
    *(_QWORD *)v44 = *v45;
    goto LABEL_23;
  }
  v63 = *(_QWORD *)(v44 + 8);
  *(_QWORD *)(v44 + 88) += 192LL;
  v64 = (v63 + 63) & 0xFFFFFFFFFFFFFFC0LL;
  if ( *(_QWORD *)(v44 + 16) < v64 + 192 || !v63 )
  {
    v65 = sub_9D1E70(v44 + 8, 192, 192, 6);
    v37 = v67;
    v45 = (_QWORD *)v65;
LABEL_23:
    memset(v45, 0, 0xC0u);
    v46 = v45;
    do
    {
      *v46 = 0;
      v46 += 2;
      *(v46 - 1) = 0;
    }
    while ( v45 + 16 != v46 );
    goto LABEL_25;
  }
  *(_QWORD *)(v44 + 8) = v64 + 192;
  if ( v64 )
  {
    v45 = (_QWORD *)((v63 + 63) & 0xFFFFFFFFFFFFFFC0LL);
    goto LABEL_23;
  }
LABEL_25:
  v83[v39] = v45;
  v68 = 1;
LABEL_26:
  v66 = sub_F03E60(v70, v34, 8, (__int64)v81, (__int64)v82, v37, 1u);
  sub_2E1B960((__int64)v83, v70, v81, (__int64)v82);
  if ( v72 )
    sub_F03AD0((unsigned int *)v6, v27);
  v47 = v27;
  v48 = v6;
  v49 = 0;
  while ( 1 )
  {
    v50 = v82[v49];
    v51 = (unsigned int)(v50 - 1);
    v52 = v83[v49];
    v53 = *(_QWORD *)(v52 + 16 * v51 + 8);
    if ( v75 != (_DWORD)v49 || !v68 )
      break;
    ++v49;
    v47 += (unsigned __int8)sub_2E1C6D0((__int64)a1, v47, v51 | v52 & 0xFFFFFFFFFFFFFFC0LL, v53);
    if ( v70 == v49 )
      goto LABEL_39;
LABEL_32:
    sub_F03D40(v48, v47);
  }
  *(_DWORD *)(*((_QWORD *)a1 + 1) + 16LL * v47 + 8) = v50;
  if ( v47 )
  {
    v54 = *((_QWORD *)a1 + 1) + 16LL * (v47 - 1);
    v55 = (unsigned __int64 *)(*(_QWORD *)v54 + 8LL * *(unsigned int *)(v54 + 12));
    *v55 = v51 | *v55 & 0xFFFFFFFFFFFFFFC0LL;
  }
  ++v49;
  sub_2E1A5E0((__int64)a1, v47, v53);
  if ( v70 != v49 )
    goto LABEL_32;
LABEL_39:
  for ( i = v70 - 1; i != (_DWORD)v66; --i )
    sub_F03AD0((unsigned int *)v48, v47);
  *(_DWORD *)(*((_QWORD *)a1 + 1) + 16LL * v47 + 12) = HIDWORD(v66);
  v57 = *((_QWORD *)a1 + 1) + 16LL * a1[4] - 16;
  v58 = *(_DWORD *)(v57 + 8);
  v14 = *(_DWORD *)(v57 + 12) == v58;
  v13 = sub_2E1A1C0(*(_QWORD *)v57, (unsigned int *)(v57 + 12), v58, v77, a3, a4);
LABEL_6:
  v15 = a1[4];
  *(_DWORD *)(*((_QWORD *)a1 + 1) + 16LL * (v15 - 1) + 8) = v13;
  if ( v15 == 1 )
  {
    if ( !v14 )
      return;
LABEL_11:
    v19 = a3;
    v20 = a1[4] - 1;
LABEL_12:
    sub_2E1A5E0((__int64)a1, v20, v19);
    return;
  }
  v17 = *((_QWORD *)a1 + 1) + 16LL * (v15 - 2);
  v18 = (unsigned __int64 *)(*(_QWORD *)v17 + 8LL * *(unsigned int *)(v17 + 12));
  *v18 = *v18 & 0xFFFFFFFFFFFFFFC0LL | (v13 - 1);
  if ( v14 )
    goto LABEL_11;
}
