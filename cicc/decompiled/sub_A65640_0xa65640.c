// Function: sub_A65640
// Address: 0xa65640
//
__int64 __fastcall sub_A65640(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 *v3; // rbx
  __int64 v4; // rdi
  void (*v5)(); // rax
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rdi
  char v9; // al
  char v10; // al
  unsigned __int16 v11; // di
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r12
  unsigned int i; // r14d
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdi
  _BYTE *v21; // rax
  __int64 v22; // rdx
  char v23; // al
  size_t v24; // r12
  char *v25; // r14
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int16 v30; // r12
  __int64 v31; // rax
  __int16 v32; // ax
  __int64 v34; // r12
  __int64 v35; // r13
  __int64 v36; // rdi
  _BYTE *v37; // rax
  _BYTE *v38; // rsi
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rdi
  _BYTE *v46; // rax
  __int64 v47; // rdx
  int v48; // eax
  __int64 v49; // rdi
  int v50; // ecx
  _WORD *v51; // rdx
  __int64 v52; // rax
  __int64 j; // r15
  __int64 v54; // rsi
  __int64 v55; // r14
  __int64 v56; // rax
  __int64 v57; // r12
  __int64 v58; // rsi
  __int64 v59; // rdi
  __int64 v60; // r12
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rax
  __int64 v64; // r14
  __int64 v65; // r12
  __int64 v66; // rax
  int v67; // eax
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // r12
  __int64 v74; // rdx
  __int64 v75; // rbx
  __int64 v76; // r14
  __int64 v77; // rcx
  _QWORD *v78; // rax
  __int64 (__fastcall **v79)(); // rdx
  unsigned __int64 v80; // rcx
  unsigned __int64 v81; // r10
  const char *v82; // r12
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // r12
  _QWORD *v87; // rax
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rdx
  char *v92; // [rsp+8h] [rbp-E8h]
  _QWORD *v93; // [rsp+10h] [rbp-E0h]
  int v95; // [rsp+20h] [rbp-D0h]
  __int64 v96; // [rsp+20h] [rbp-D0h]
  int v97; // [rsp+20h] [rbp-D0h]
  __int64 *v98; // [rsp+20h] [rbp-D0h]
  __int64 v99; // [rsp+28h] [rbp-C8h]
  __int64 v100; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v101; // [rsp+48h] [rbp-A8h] BYREF
  __int64 (__fastcall **v102)(); // [rsp+50h] [rbp-A0h] BYREF
  _QWORD *v103; // [rsp+58h] [rbp-98h]
  _QWORD v104[2]; // [rsp+60h] [rbp-90h] BYREF
  _QWORD *v105; // [rsp+70h] [rbp-80h] BYREF
  __int64 v106; // [rsp+78h] [rbp-78h]
  _QWORD v107[14]; // [rsp+80h] [rbp-70h] BYREF

  v2 = a2;
  v3 = a1;
  v4 = a1[33];
  if ( v4 )
  {
    v5 = *(void (**)())(*(_QWORD *)v4 + 16LL);
    if ( v5 != nullsub_30 )
      ((void (__fastcall *)(__int64, __int64, __int64))v5)(v4, a2, *v3);
  }
  if ( (*(_BYTE *)(a2 + 35) & 8) != 0 )
    sub_904010(*v3, "; Materializable\n");
  v100 = *(_QWORD *)(a2 + 120);
  if ( (unsigned __int8)sub_A74740(&v100, 0xFFFFFFFFLL) )
  {
    v68 = sub_A74680(&v100);
    LOBYTE(v104[0]) = 0;
    v101 = v68;
    v102 = (__int64 (__fastcall **)())v104;
    v103 = 0;
    v73 = sub_A73280(&v101, 0xFFFFFFFFLL, v69, v70, v71, v72);
    v74 = sub_A73290(&v101);
    if ( v73 != v74 )
    {
      v98 = v3;
      v75 = v73;
      v76 = v74;
      do
      {
        if ( !(unsigned __int8)sub_A71840(v75) )
        {
          v78 = v103;
          if ( v103 )
          {
            v79 = v102;
            v80 = 15;
            v81 = (unsigned __int64)v103 + 1;
            if ( v102 != v104 )
              v80 = v104[0];
            if ( v81 > v80 )
            {
              v92 = (char *)v103 + 1;
              v93 = v103;
              sub_2240BB0(&v102, v103, 0, 0, 1);
              v79 = v102;
              v81 = (unsigned __int64)v92;
              v78 = v93;
            }
            *((_BYTE *)v78 + (_QWORD)v79) = 32;
            v103 = (_QWORD *)v81;
            *((_BYTE *)v78 + (_QWORD)v102 + 1) = 0;
          }
          sub_A759D0(&v105, v75, 0);
          sub_2241490(&v102, v105, v106, v77);
          if ( v105 != v107 )
            j_j___libc_free_0(v105, v107[0] + 1LL);
        }
        v75 += 8;
      }
      while ( v76 != v75 );
      v3 = v98;
    }
    if ( v103 )
    {
      v89 = sub_904010(*v3, "; Function Attrs: ");
      v90 = sub_CB6200(v89, v102, v103);
      sub_A51310(v90, 0xAu);
    }
    if ( v102 != v104 )
      j_j___libc_free_0(v102, v104[0] + 1LL);
  }
  v6 = v3[4];
  *(_QWORD *)(v6 + 16) = a2;
  *(_BYTE *)(v6 + 24) = 0;
  if ( (unsigned __int8)sub_B2FC80(a2) )
  {
    sub_904010(*v3, "declare");
    v106 = 0x400000000LL;
    v105 = v107;
    sub_B9A9D0(a2, &v105);
    sub_A5C960(v3, (unsigned int *)&v105, " ", 1u);
    sub_A51310(*v3, 0x20u);
    if ( v105 != v107 )
      _libc_free(v105, 32);
  }
  else
  {
    sub_904010(*v3, "define ");
  }
  v7 = *v3;
  sub_A51210((__int64)&v105, *(_BYTE *)(a2 + 32) & 0xF);
  sub_CB6200(v7, v105, v106);
  if ( v105 != v107 )
    j_j___libc_free_0(v105, v107[0] + 1LL);
  sub_A518A0(a2, *v3);
  v8 = *v3;
  v9 = (*(_BYTE *)(a2 + 32) >> 4) & 3;
  if ( v9 == 1 )
  {
    sub_904010(v8, "hidden ");
    v8 = *v3;
  }
  else if ( v9 == 2 )
  {
    sub_904010(v8, "protected ");
    v8 = *v3;
  }
  v10 = *(_BYTE *)(a2 + 33) & 3;
  if ( v10 == 1 )
  {
    sub_904010(v8, "dllimport ");
    v11 = (*(_WORD *)(a2 + 2) >> 4) & 0x3FF;
    if ( !v11 )
      goto LABEL_19;
LABEL_94:
    sub_A51410(v11, *v3);
    sub_904010(*v3, " ");
    goto LABEL_19;
  }
  if ( v10 == 2 )
    sub_904010(v8, "dllexport ");
  v11 = (*(_WORD *)(a2 + 2) >> 4) & 0x3FF;
  if ( v11 )
    goto LABEL_94;
LABEL_19:
  v99 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int8)sub_A74740(&v100, 0) )
  {
    v55 = *v3;
    sub_A76D50(&v105, &v100, 0, 0);
    v56 = sub_CB6200(v55, v105, v106);
    sub_A51310(v56, 0x20u);
    if ( v105 != v107 )
      j_j___libc_free_0(v105, v107[0] + 1LL);
  }
  sub_A57EC0((__int64)(v3 + 5), **(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL), *v3);
  v12 = v3[4];
  v13 = *(_QWORD *)(a2 + 40);
  v14 = *v3;
  v102 = off_4979428;
  v104[0] = v12;
  v103 = v3 + 5;
  v104[1] = v13;
  sub_A51310(v14, 0x20u);
  sub_A5A730(*v3, a2, (__int64)&v102);
  sub_A51310(*v3, 0x28u);
  if ( !(unsigned __int8)sub_B2FC80(a2) || *((_BYTE *)v3 + 320) )
  {
    if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a2);
      v34 = *(_QWORD *)(a2 + 96);
      v96 = v34 + 40LL * *(_QWORD *)(a2 + 104);
      if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
      {
        sub_B2C6D0(a2);
        v34 = *(_QWORD *)(a2 + 96);
      }
    }
    else
    {
      v34 = *(_QWORD *)(a2 + 96);
      v96 = v34 + 40LL * *(_QWORD *)(a2 + 104);
    }
    if ( v96 != v34 )
    {
      v35 = v96;
      do
      {
        while ( 1 )
        {
          v39 = *(unsigned int *)(v34 + 32);
          if ( (_DWORD)v39 )
          {
            sub_904010(*v3, ", ");
            v39 = *(unsigned int *)(v34 + 32);
          }
          v40 = sub_A744E0(&v100, v39);
          v41 = *v3;
          v105 = (_QWORD *)v40;
          sub_A57EC0((__int64)(v3 + 5), *(_QWORD *)(v34 + 8), v41);
          if ( v105 )
          {
            v45 = *v3;
            v46 = *(_BYTE **)(*v3 + 32);
            if ( (unsigned __int64)v46 >= *(_QWORD *)(*v3 + 24) )
            {
              sub_CB5D20(v45, 32);
            }
            else
            {
              v47 = (__int64)(v46 + 1);
              *(_QWORD *)(v45 + 32) = v46 + 1;
              *v46 = 32;
            }
            sub_A58630(v3, (__int64)&v105, v47, v42, v43, v44);
          }
          if ( (*(_BYTE *)(v34 + 7) & 0x10) != 0 )
            break;
          v48 = sub_A5A650(v3[4], v34);
          v49 = *v3;
          v50 = v48;
          v51 = *(_WORD **)(*v3 + 32);
          if ( *(_QWORD *)(*v3 + 24) - (_QWORD)v51 <= 1u )
          {
            v97 = v48;
            v52 = sub_CB6200(v49, " %", 2);
            v50 = v97;
            v49 = v52;
          }
          else
          {
            *v51 = 9504;
            *(_QWORD *)(v49 + 32) += 2LL;
          }
          v34 += 40;
          sub_CB59F0(v49, v50);
          if ( v34 == v35 )
            goto LABEL_75;
        }
        v36 = *v3;
        v37 = *(_BYTE **)(*v3 + 32);
        if ( (unsigned __int64)v37 >= *(_QWORD *)(*v3 + 24) )
        {
          sub_CB5D20(v36, 32);
        }
        else
        {
          *(_QWORD *)(v36 + 32) = v37 + 1;
          *v37 = 32;
        }
        v38 = (_BYTE *)v34;
        v34 += 40;
        sub_A55040(*v3, v38);
      }
      while ( v34 != v35 );
LABEL_75:
      v2 = a2;
    }
    goto LABEL_29;
  }
  v95 = *(_DWORD *)(v99 + 12) - 1;
  if ( *(_DWORD *)(v99 + 12) != 1 )
  {
    v15 = 8;
    for ( i = 0; ; ++i )
    {
      sub_A57EC0((__int64)(v3 + 5), *(_QWORD *)(*(_QWORD *)(v99 + 16) + v15), *v3);
      v105 = (_QWORD *)sub_A744E0(&v100, i);
      if ( v105 )
      {
        v20 = *v3;
        v21 = *(_BYTE **)(*v3 + 32);
        if ( (unsigned __int64)v21 >= *(_QWORD *)(*v3 + 24) )
        {
          sub_CB5D20(v20, 32);
        }
        else
        {
          v22 = (__int64)(v21 + 1);
          *(_QWORD *)(v20 + 32) = v21 + 1;
          *v21 = 32;
        }
        sub_A58630(v3, (__int64)&v105, v22, v17, v18, v19);
      }
      if ( v95 == i + 1 )
        break;
      v15 += 8;
      sub_904010(*v3, ", ");
    }
LABEL_29:
    if ( !(*(_DWORD *)(v99 + 8) >> 8) )
      goto LABEL_33;
    if ( *(_DWORD *)(v99 + 12) != 1 )
      sub_904010(*v3, ", ");
    goto LABEL_32;
  }
  if ( *(_DWORD *)(v99 + 8) >> 8 )
LABEL_32:
    sub_904010(*v3, "...");
LABEL_33:
  sub_A51310(*v3, 0x29u);
  v23 = *(_BYTE *)(v2 + 32) >> 6;
  if ( v23 == 1 )
  {
    v24 = 18;
    v25 = "local_unnamed_addr";
    goto LABEL_35;
  }
  v24 = 12;
  v25 = "unnamed_addr";
  if ( v23 == 2 )
  {
LABEL_35:
    v26 = sub_A51310(*v3, 0x20u);
    sub_A51340(v26, v25, v24);
    goto LABEL_36;
  }
  if ( v23 )
    BUG();
LABEL_36:
  v27 = *(_QWORD *)(v2 + 40);
  if ( *(_DWORD *)(*(_QWORD *)(v2 + 8) + 8LL) >> 8 || !v27 || *(_DWORD *)(v27 + 320) )
  {
    v28 = sub_904010(*v3, " addrspace(");
    v29 = sub_CB59D0(v28, *(_DWORD *)(*(_QWORD *)(v2 + 8) + 8LL) >> 8);
    sub_904010(v29, ")");
  }
  if ( (unsigned __int8)sub_A74740(&v100, 0xFFFFFFFFLL) )
  {
    v63 = sub_904010(*v3, " #");
    v64 = v3[4];
    v65 = v63;
    v66 = sub_A74680(&v100);
    v67 = sub_A5A580(v64, v66);
    sub_CB59F0(v65, v67);
  }
  if ( (*(_BYTE *)(v2 + 35) & 4) == 0 )
  {
    if ( *(char *)(v2 + 33) >= 0 )
      goto LABEL_44;
    goto LABEL_101;
  }
  sub_904010(*v3, " section \"");
  v57 = *v3;
  v58 = 0;
  v59 = 0;
  if ( (*(_BYTE *)(v2 + 35) & 4) != 0 )
  {
    v59 = sub_B31D10(v2);
    v58 = v91;
  }
  sub_C92400(v59, v58, v57);
  sub_A51310(*v3, 0x22u);
  if ( *(char *)(v2 + 33) < 0 )
  {
LABEL_101:
    sub_904010(*v3, " partition \"");
    v60 = *v3;
    v61 = sub_B30A70(v2);
    sub_C92400(v61, v62, v60);
    sub_A51310(*v3, 0x22u);
  }
LABEL_44:
  sub_A550E0(*v3, v2);
  v30 = (*(_WORD *)(v2 + 34) >> 1) & 0x3F;
  if ( v30 )
  {
    v31 = sub_904010(*v3, " align ");
    sub_CB59D0(v31, 1LL << ((unsigned __int8)v30 - 1));
  }
  v32 = *(_WORD *)(v2 + 2);
  if ( (v32 & 0x4000) != 0 )
  {
    v86 = sub_904010(*v3, " gc \"");
    v87 = (_QWORD *)sub_B2DBE0(v2);
    v88 = sub_CB6200(v86, *v87, v87[1]);
    sub_A51310(v88, 0x22u);
    v32 = *(_WORD *)(v2 + 2);
  }
  if ( (v32 & 2) != 0 )
  {
    sub_904010(*v3, " prefix ");
    v85 = sub_B2E510(v2, " prefix ");
    sub_A5B360(v3, v85, 1);
    v32 = *(_WORD *)(v2 + 2);
  }
  if ( (v32 & 4) != 0 )
  {
    sub_904010(*v3, " prologue ");
    v84 = sub_B2E520(v2, " prologue ");
    sub_A5B360(v3, v84, 1);
    v32 = *(_WORD *)(v2 + 2);
  }
  if ( (v32 & 8) != 0 )
  {
    sub_904010(*v3, " personality ");
    v83 = sub_B2E500(v2);
    sub_A5B360(v3, v83, 1);
  }
  if ( (_BYTE)qword_4F809C8 )
  {
    if ( (*(_BYTE *)(v2 + 7) & 0x20) != 0 )
    {
      v82 = (const char *)sub_B91C10(v2, 2);
      if ( v82 )
      {
        sub_904010(*v3, " ");
        sub_A61DE0(v82, *v3, v3[1]);
      }
    }
  }
  if ( (unsigned __int8)sub_B2FC80(v2) )
  {
    sub_A51310(*v3, 0xAu);
  }
  else
  {
    v105 = v107;
    v106 = 0x400000000LL;
    sub_B9A9D0(v2, &v105);
    sub_A5C960(v3, (unsigned int *)&v105, " ", 1u);
    sub_904010(*v3, " {");
    for ( j = *(_QWORD *)(v2 + 80); v2 + 72 != j; j = *(_QWORD *)(j + 8) )
    {
      v54 = j - 24;
      if ( !j )
        v54 = 0;
      sub_A651F0(v3, v54);
    }
    sub_A5B410(v3, v2);
    sub_904010(*v3, "}\n");
    if ( v105 != v107 )
      _libc_free(v105, "}\n");
  }
  return sub_A56010(v3[4]);
}
