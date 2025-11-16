// Function: sub_1F1F320
// Address: 0x1f1f320
//
void __fastcall sub_1F1F320(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v6; // r13
  __int64 v7; // r12
  unsigned int v8; // edx
  __int64 v9; // rcx
  __int64 v10; // rsi
  int v11; // ebx
  __int64 *v12; // r10
  unsigned int v13; // r15d
  unsigned int v14; // eax
  bool v15; // bl
  int v16; // esi
  __int64 v17; // rbx
  __int64 v18; // rdx
  unsigned __int64 *v19; // rcx
  __int64 v20; // rdx
  int v21; // esi
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rax
  unsigned __int64 v25; // rbx
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rbx
  __int64 v29; // rax
  int v30; // r9d
  __int64 v31; // r15
  unsigned int v32; // eax
  unsigned int v33; // ecx
  __int64 v34; // rdx
  unsigned int v35; // r10d
  __int64 v36; // rax
  __int64 v37; // r8
  unsigned int v38; // r10d
  int v39; // r9d
  char v40; // dl
  unsigned __int64 v41; // rax
  int v42; // edx
  __int64 v43; // rcx
  __int64 v44; // rax
  __int64 v45; // rax
  _QWORD *v46; // rdx
  unsigned int v47; // r14d
  __int64 v48; // rbx
  int v49; // r9d
  __int64 v50; // rdi
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // rax
  unsigned __int64 *v54; // rax
  unsigned int v55; // ebx
  unsigned int v56; // r12d
  __int64 v57; // rax
  unsigned int v58; // edx
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  __int64 v63; // rax
  __int64 v64; // [rsp+0h] [rbp-C0h]
  __int64 v65; // [rsp+10h] [rbp-B0h]
  int v66; // [rsp+18h] [rbp-A8h]
  char v67; // [rsp+18h] [rbp-A8h]
  __int64 v68; // [rsp+18h] [rbp-A8h]
  unsigned int v69; // [rsp+20h] [rbp-A0h]
  unsigned int v70; // [rsp+20h] [rbp-A0h]
  unsigned int v71; // [rsp+28h] [rbp-98h]
  __int64 v72; // [rsp+28h] [rbp-98h]
  int v73; // [rsp+28h] [rbp-98h]
  __int64 v74; // [rsp+30h] [rbp-90h]
  unsigned int v75; // [rsp+30h] [rbp-90h]
  __int64 *v76; // [rsp+30h] [rbp-90h]
  int v77; // [rsp+38h] [rbp-88h]
  unsigned int v78; // [rsp+38h] [rbp-88h]
  __int64 v79; // [rsp+38h] [rbp-88h]
  __int64 *v80; // [rsp+38h] [rbp-88h]
  __int64 v81; // [rsp+40h] [rbp-80h]
  __int64 v82; // [rsp+40h] [rbp-80h]
  unsigned int v84[4]; // [rsp+50h] [rbp-70h] BYREF
  _DWORD v85[4]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v86[10]; // [rsp+70h] [rbp-50h] BYREF

  v6 = a1 + 8;
  v7 = a1;
  v8 = *(_DWORD *)(a1 + 16);
  v81 = a2;
  if ( !v8 || (v9 = *(_QWORD *)(a1 + 8), *(_DWORD *)(v9 + 12) >= *(_DWORD *)(v9 + 8)) )
  {
    v17 = 16LL * *(unsigned int *)(*(_QWORD *)a1 + 184LL);
    sub_3945E40(a1 + 8, *(unsigned int *)(*(_QWORD *)a1 + 184LL));
    ++*(_DWORD *)(*(_QWORD *)(a1 + 8) + v17 + 12);
    v8 = *(_DWORD *)(a1 + 16);
    v9 = *(_QWORD *)(a1 + 8);
  }
  v10 = v9 + 16LL * v8 - 16;
  v11 = *(_DWORD *)(v10 + 12);
  v12 = *(__int64 **)v10;
  if ( !v11
    && (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2 >> 1) & 3) < (*(_DWORD *)((*v12 & 0xFFFFFFFFFFFFFFF8LL)
                                                                                                  + 24)
                                                                                      | (unsigned int)(*v12 >> 1) & 3) )
  {
    v22 = sub_3945DA0(v6, v8 - 1);
    if ( v22 )
    {
      v23 = v22;
      v24 = v22 & 0x3F;
      v25 = v23 & 0xFFFFFFFFFFFFFFC0LL;
      v26 = *(unsigned int *)(a1 + 16);
      v10 = *(_QWORD *)(v7 + 8) + 16 * v26 - 16;
      v27 = (unsigned int)v24;
      v12 = *(__int64 **)v10;
      if ( *(_DWORD *)(v25 + 4 * v24 + 144) != a4 || (v59 = 16 * v24, *(_QWORD *)(v25 + v59 + 8) != a2) )
      {
        v11 = *(_DWORD *)(v10 + 12);
        goto LABEL_5;
      }
      v80 = *(__int64 **)v10;
      v82 = v59;
      v72 = v27;
      v76 = (__int64 *)(v25 + v59 + 8);
      sub_3945E40(v6, (unsigned int)(v26 - 1));
      v62 = *v80;
      if ( (*(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a3 >> 1) & 3) <= (*(_DWORD *)((*v80 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                           | (unsigned int)(*v80 >> 1)
                                                                                           & 3) )
      {
        v61 = (__int64)v76;
        v60 = v72;
        if ( *((_DWORD *)v80 + 36) != a4 || a3 != v62 )
        {
          *(_QWORD *)(v25 + 16 * v72 + 8) = a3;
          v20 = *v76;
          v21 = *(_DWORD *)(v7 + 16) - 1;
          goto LABEL_12;
        }
      }
      v81 = *(_QWORD *)(v25 + v82);
      sub_1F192F0(v7, 0, v60, v61, v62);
    }
    else
    {
      **(_QWORD **)a1 = a2;
    }
    v10 = *(_QWORD *)(v7 + 8) + 16LL * *(unsigned int *)(v7 + 16) - 16;
    v11 = *(_DWORD *)(v10 + 12);
    v12 = *(__int64 **)v10;
  }
LABEL_5:
  v13 = *(_DWORD *)(v10 + 8);
  v14 = sub_1F15E30((__int64)v12, (unsigned int *)(v10 + 12), v13, v81, a3, a4);
  v15 = v13 == v11;
  if ( v14 <= 9 )
    goto LABEL_6;
  v28 = (unsigned int)(*(_DWORD *)(v7 + 16) - 1);
  v77 = *(_DWORD *)(*(_QWORD *)(v7 + 8) + 16 * v28 + 12);
  v29 = sub_3945DA0(v6, v28);
  v30 = v77;
  v31 = v29;
  if ( v29 )
  {
    v78 = 2;
    v32 = (v29 & 0x3F) + 1;
    v86[0] = v31 & 0xFFFFFFFFFFFFFFC0LL;
    v33 = 1;
    v84[0] = v32;
    v30 += v32;
  }
  else
  {
    v78 = 1;
    v32 = 0;
    v33 = 0;
  }
  v34 = *(_QWORD *)(v7 + 8) + 16 * v28;
  v35 = *(_DWORD *)(v34 + 8);
  v66 = v30;
  v69 = v33;
  v84[v33] = v35;
  v71 = v32 + v35;
  v86[v33] = *(_QWORD *)v34;
  v74 = v33;
  v36 = sub_3945FF0(v6, (unsigned int)v28);
  v37 = v74;
  v38 = v71;
  v39 = v66;
  if ( v36 )
  {
    v40 = v36;
    v37 = v78;
    v41 = v36 & 0xFFFFFFFFFFFFFFC0LL;
    v75 = v69 + 2;
    v42 = (v40 & 0x3F) + 1;
    v86[v78] = v41;
    v38 = v42 + v71;
    v84[v78] = v42;
    if ( v42 + v71 + 1 > 9 * v69 + 18 )
      goto LABEL_20;
  }
  else
  {
    if ( v71 + 1 > 9 * (unsigned int)(v78 != 1) + 9 )
    {
      if ( v78 == 1 )
      {
        v42 = v84[1];
        v41 = v86[1];
        v43 = 1;
        v75 = 2;
        v37 = 1;
        goto LABEL_21;
      }
      v42 = v84[v74];
      v41 = v86[v74];
      v78 = v69;
      v75 = 2;
LABEL_20:
      v43 = v75++;
LABEL_21:
      v86[v43] = v41;
      v44 = *(_QWORD *)v7;
      v84[v43] = v42;
      v84[v37] = 0;
      v45 = *(_QWORD *)(v44 + 192);
      v46 = *(_QWORD **)v45;
      if ( *(_QWORD *)v45 )
      {
        *(_QWORD *)v45 = *v46;
      }
      else
      {
        v68 = v37;
        v70 = v38;
        v73 = v39;
        v63 = sub_145CBF0((__int64 *)(v45 + 8), 192, 64);
        v39 = v73;
        v38 = v70;
        v37 = v68;
        v46 = (_QWORD *)v63;
      }
      v67 = 1;
      memset(v46, 0, 0xB8u);
      v86[v37] = v46;
      goto LABEL_24;
    }
    v75 = v78;
  }
  v67 = 0;
  v78 = 0;
LABEL_24:
  v65 = sub_39461C0(v75, v38, 9, (unsigned int)v84, (unsigned int)v85, v39, 1);
  sub_1F19780((__int64)v86, v75, v84, (__int64)v85);
  if ( v31 )
    sub_3945E40(v6, (unsigned int)v28);
  v64 = a3;
  v47 = v28;
  v48 = 0;
  while ( 1 )
  {
    v49 = v85[v48];
    v50 = (unsigned int)(v49 - 1);
    v51 = v86[v48];
    v52 = *(_QWORD *)(v51 + 16 * v50 + 8);
    if ( v78 != (_DWORD)v48 || !v67 )
      break;
    ++v48;
    v47 += (unsigned __int8)sub_1F1F080(v7, v47, v50 | v51 & 0xFFFFFFFFFFFFFFC0LL, v52);
    if ( v75 == v48 )
      goto LABEL_37;
LABEL_30:
    sub_39460A0(v6, v47);
  }
  *(_DWORD *)(*(_QWORD *)(v7 + 8) + 16LL * v47 + 8) = v49;
  if ( v47 )
  {
    v53 = *(_QWORD *)(v7 + 8) + 16LL * (v47 - 1);
    v54 = (unsigned __int64 *)(*(_QWORD *)v53 + 8LL * *(unsigned int *)(v53 + 12));
    *v54 = v50 | *v54 & 0xFFFFFFFFFFFFFFC0LL;
  }
  ++v48;
  sub_1F18EF0(v7, v47, v52);
  if ( v75 != v48 )
    goto LABEL_30;
LABEL_37:
  v55 = v47;
  a3 = v64;
  if ( v75 - 1 != (_DWORD)v65 )
  {
    v79 = v7;
    v56 = v75 - 1;
    do
    {
      --v56;
      sub_3945E40(v6, v55);
    }
    while ( v56 != (_DWORD)v65 );
    v7 = v79;
  }
  *(_DWORD *)(*(_QWORD *)(v7 + 8) + 16LL * v55 + 12) = HIDWORD(v65);
  v57 = *(_QWORD *)(v7 + 8) + 16LL * *(unsigned int *)(v7 + 16) - 16;
  v58 = *(_DWORD *)(v57 + 8);
  v15 = *(_DWORD *)(v57 + 12) == v58;
  v14 = sub_1F15E30(*(_QWORD *)v57, (unsigned int *)(v57 + 12), v58, v81, v64, a4);
LABEL_6:
  v16 = *(_DWORD *)(v7 + 16);
  *(_DWORD *)(*(_QWORD *)(v7 + 8) + 16LL * (unsigned int)(v16 - 1) + 8) = v14;
  if ( v16 == 1 )
  {
    if ( !v15 )
      return;
LABEL_11:
    v20 = a3;
    v21 = *(_DWORD *)(v7 + 16) - 1;
LABEL_12:
    sub_1F18EF0(v7, v21, v20);
    return;
  }
  v18 = *(_QWORD *)(v7 + 8) + 16LL * (unsigned int)(v16 - 2);
  v19 = (unsigned __int64 *)(*(_QWORD *)v18 + 8LL * *(unsigned int *)(v18 + 12));
  *v19 = *v19 & 0xFFFFFFFFFFFFFFC0LL | (v14 - 1);
  if ( v15 )
    goto LABEL_11;
}
