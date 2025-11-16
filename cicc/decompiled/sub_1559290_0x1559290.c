// Function: sub_1559290
// Address: 0x1559290
//
__int64 __fastcall sub_1559290(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 *v3; // rbx
  __int64 v4; // rdi
  void (*v5)(); // rax
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rdi
  char v9; // al
  char v10; // al
  unsigned __int16 v11; // di
  __int64 v12; // rdi
  _BYTE *v13; // rax
  __int64 v14; // r12
  unsigned int v15; // r15d
  unsigned int v16; // r13d
  __int64 v17; // r15
  _BYTE *v18; // rax
  char v19; // al
  size_t v20; // r12
  const char *v21; // r15
  __int64 v22; // rax
  __int16 v23; // ax
  __int64 v25; // r12
  const char *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r13
  _BYTE *v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rdi
  _BYTE *v32; // rax
  __int64 v33; // rsi
  __int64 i; // r15
  __int64 v35; // rsi
  __int64 *v36; // rax
  __int64 v37; // rax
  __int64 v38; // r12
  __int64 v39; // r15
  __int64 v40; // rbx
  __int64 v41; // rcx
  unsigned __int64 v42; // rax
  const char *v43; // rcx
  unsigned __int64 v44; // rdx
  unsigned __int64 v45; // r10
  __int64 v46; // r12
  __int64 v47; // rax
  __int64 *v48; // rax
  __int64 *v49; // rax
  __int64 v50; // r12
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r12
  __int64 v59; // rsi
  __int64 v60; // rdi
  __int64 v61; // rax
  __int64 v62; // r15
  __int64 v63; // r12
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rcx
  int v67; // eax
  __int64 v68; // rdx
  __int64 v69; // rax
  __int64 v70; // rax
  unsigned __int64 v71; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v73; // [rsp+10h] [rbp-E0h]
  __int64 v74; // [rsp+20h] [rbp-D0h]
  __int64 *v75; // [rsp+20h] [rbp-D0h]
  int v76; // [rsp+28h] [rbp-C8h]
  __int64 v77; // [rsp+28h] [rbp-C8h]
  __int64 v78; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v79; // [rsp+48h] [rbp-A8h] BYREF
  const char *v80; // [rsp+50h] [rbp-A0h] BYREF
  unsigned __int64 v81; // [rsp+58h] [rbp-98h]
  _QWORD v82[2]; // [rsp+60h] [rbp-90h] BYREF
  const char *v83; // [rsp+70h] [rbp-80h] BYREF
  __int64 v84; // [rsp+78h] [rbp-78h]
  _QWORD v85[14]; // [rsp+80h] [rbp-70h] BYREF

  v2 = a2;
  v3 = a1;
  sub_1549FC0(*a1, 0xAu);
  v4 = a1[29];
  if ( v4 )
  {
    v5 = *(void (**)())(*(_QWORD *)v4 + 16LL);
    if ( v5 != nullsub_524 )
      ((void (__fastcall *)(__int64, __int64, __int64))v5)(v4, a2, *v3);
  }
  if ( (*(_BYTE *)(a2 + 34) & 0x40) != 0 )
    sub_1263B40(*v3, "; Materializable\n");
  v78 = *(_QWORD *)(a2 + 112);
  if ( (unsigned __int8)sub_15602F0(&v78, 0xFFFFFFFFLL) )
  {
    v37 = sub_1560250(&v78);
    LOBYTE(v82[0]) = 0;
    v79 = v37;
    v80 = (const char *)v82;
    v81 = 0;
    v38 = sub_155EE30(&v79);
    v39 = sub_155EE40(&v79);
    if ( v38 != v39 )
    {
      v75 = v3;
      v40 = v38;
      do
      {
        if ( !(unsigned __int8)sub_155D3E0(v40) )
        {
          v42 = v81;
          if ( v81 )
          {
            v43 = v80;
            v44 = 15;
            v45 = v81 + 1;
            if ( v80 != (const char *)v82 )
              v44 = v82[0];
            if ( v45 > v44 )
            {
              v71 = v81 + 1;
              v73 = v81;
              sub_2240BB0(&v80, v81, 0, 0, 1);
              v43 = v80;
              v45 = v71;
              v42 = v73;
            }
            v43[v42] = 32;
            v81 = v45;
            v80[v42 + 1] = 0;
          }
          sub_155D8D0(&v83, v40, 0);
          sub_2241490(&v80, v83, v84, v41);
          if ( v83 != (const char *)v85 )
            j_j___libc_free_0(v83, v85[0] + 1LL);
        }
        v40 += 8;
      }
      while ( v39 != v40 );
      v3 = v75;
    }
    if ( v81 )
    {
      v69 = sub_1263B40(*v3, "; Function Attrs: ");
      v70 = sub_16E7EE0(v69, v80, v81);
      sub_1549FC0(v70, 0xAu);
    }
    if ( v80 != (const char *)v82 )
      j_j___libc_free_0(v80, v82[0] + 1LL);
  }
  v6 = v3[4];
  *(_QWORD *)(v6 + 8) = a2;
  *(_BYTE *)(v6 + 16) = 0;
  if ( (unsigned __int8)sub_15E4F60(a2) )
  {
    sub_1263B40(*v3, "declare");
    v84 = 0x400000000LL;
    v83 = (const char *)v85;
    sub_1626D60(a2, &v83);
    sub_1550BA0(v3, (unsigned int *)&v83, " ", 1u);
    sub_1549FC0(*v3, 0x20u);
    if ( v83 != (const char *)v85 )
      _libc_free((unsigned __int64)v83);
  }
  else
  {
    sub_1263B40(*v3, "define ");
  }
  v7 = *v3;
  sub_1549EC0((__int64)&v83, *(_BYTE *)(a2 + 32) & 0xF);
  sub_16E7EE0(v7, v83, v84);
  if ( v83 != (const char *)v85 )
    j_j___libc_free_0(v83, v85[0] + 1LL);
  sub_154A4E0(a2, *v3);
  v8 = *v3;
  v9 = (*(_BYTE *)(a2 + 32) >> 4) & 3;
  if ( v9 == 1 )
  {
    sub_1263B40(v8, "hidden ");
    v8 = *v3;
    v10 = *(_BYTE *)(a2 + 33) & 3;
    if ( v10 != 1 )
      goto LABEL_16;
LABEL_88:
    sub_1263B40(v8, "dllimport ");
    v11 = (*(_WORD *)(a2 + 18) >> 4) & 0x3FF;
    if ( !v11 )
      goto LABEL_19;
LABEL_89:
    sub_154A100(v11, *v3);
    sub_1263B40(*v3, " ");
    goto LABEL_19;
  }
  if ( v9 == 2 )
  {
    sub_1263B40(v8, "protected ");
    v8 = *v3;
  }
  v10 = *(_BYTE *)(a2 + 33) & 3;
  if ( v10 == 1 )
    goto LABEL_88;
LABEL_16:
  if ( v10 == 2 )
    sub_1263B40(v8, "dllexport ");
  v11 = (*(_WORD *)(a2 + 18) >> 4) & 0x3FF;
  if ( v11 )
    goto LABEL_89;
LABEL_19:
  v74 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int8)sub_15602F0(&v78, 0) )
  {
    v46 = *v3;
    sub_1560450(&v83, &v78, 0, 0);
    v47 = sub_16E7EE0(v46, v83, v84);
    sub_1549FC0(v47, 0x20u);
    if ( v83 != (const char *)v85 )
      j_j___libc_free_0(v83, v85[0] + 1LL);
  }
  sub_154DAA0((__int64)(v3 + 5), **(_QWORD **)(*(_QWORD *)(a2 + 24) + 16LL), *v3);
  sub_1549FC0(*v3, 0x20u);
  sub_1550E20(*v3, a2, (__int64)(v3 + 5), v3[4], *(_QWORD *)(a2 + 40));
  v12 = *v3;
  v13 = *(_BYTE **)(*v3 + 24);
  if ( (unsigned __int64)v13 >= *(_QWORD *)(*v3 + 16) )
  {
    sub_16E7DE0(v12, 40);
  }
  else
  {
    *(_QWORD *)(v12 + 24) = v13 + 1;
    *v13 = 40;
  }
  if ( !(unsigned __int8)sub_15E4F60(a2) || *((_BYTE *)v3 + 296) )
  {
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    {
      sub_15E08E0(a2);
      v25 = *(_QWORD *)(a2 + 88);
      v77 = v25 + 40LL * *(_QWORD *)(a2 + 96);
      if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
      {
        sub_15E08E0(a2);
        v25 = *(_QWORD *)(a2 + 88);
      }
    }
    else
    {
      v25 = *(_QWORD *)(a2 + 88);
      v77 = v25 + 40LL * *(_QWORD *)(a2 + 96);
    }
    if ( v77 != v25 )
    {
      do
      {
        while ( 1 )
        {
          v30 = *(unsigned int *)(v25 + 32);
          if ( (_DWORD)v30 )
          {
            sub_1263B40(*v3, ", ");
            v30 = *(unsigned int *)(v25 + 32);
          }
          v26 = (const char *)sub_1560230(&v78, v30);
          v27 = *v3;
          v80 = v26;
          sub_154DAA0((__int64)(v3 + 5), *(_QWORD *)v25, v27);
          if ( v80 )
          {
            v28 = *v3;
            v29 = *(_BYTE **)(*v3 + 24);
            if ( (unsigned __int64)v29 >= *(_QWORD *)(*v3 + 16) )
            {
              v28 = sub_16E7DE0(*v3, 32);
            }
            else
            {
              *(_QWORD *)(v28 + 24) = v29 + 1;
              *v29 = 32;
            }
            sub_155F820(&v83, &v80, 0);
            sub_16E7EE0(v28, v83, v84);
            if ( v83 != (const char *)v85 )
              j_j___libc_free_0(v83, v85[0] + 1LL);
          }
          if ( (*(_BYTE *)(v25 + 23) & 0x20) != 0 )
            break;
          v25 += 40;
          if ( v77 == v25 )
            goto LABEL_75;
        }
        v31 = *v3;
        v32 = *(_BYTE **)(*v3 + 24);
        if ( (unsigned __int64)v32 >= *(_QWORD *)(*v3 + 16) )
        {
          sub_16E7DE0(v31, 32);
        }
        else
        {
          *(_QWORD *)(v31 + 24) = v32 + 1;
          *v32 = 32;
        }
        v33 = v25;
        v25 += 40;
        sub_154B790(*v3, v33);
      }
      while ( v77 != v25 );
LABEL_75:
      v2 = a2;
    }
    goto LABEL_33;
  }
  v76 = *(_DWORD *)(v74 + 12) - 1;
  if ( *(_DWORD *)(v74 + 12) != 1 )
  {
    v14 = 8;
    v15 = 0;
    while ( 1 )
    {
      v16 = v15 + 1;
      sub_154DAA0((__int64)(v3 + 5), *(_QWORD *)(*(_QWORD *)(v74 + 16) + v14), *v3);
      v80 = (const char *)sub_1560230(&v78, v15);
      if ( v80 )
      {
        v17 = *v3;
        v18 = *(_BYTE **)(*v3 + 24);
        if ( (unsigned __int64)v18 >= *(_QWORD *)(*v3 + 16) )
        {
          v17 = sub_16E7DE0(*v3, 32);
        }
        else
        {
          *(_QWORD *)(v17 + 24) = v18 + 1;
          *v18 = 32;
        }
        sub_155F820(&v83, &v80, 0);
        sub_16E7EE0(v17, v83, v84);
        if ( v83 != (const char *)v85 )
          j_j___libc_free_0(v83, v85[0] + 1LL);
      }
      if ( v76 == v16 )
        break;
      v14 += 8;
      v15 = v16;
      sub_1263B40(*v3, ", ");
    }
    v2 = a2;
LABEL_33:
    if ( !(*(_DWORD *)(v74 + 8) >> 8) )
      goto LABEL_37;
    if ( *(_DWORD *)(v74 + 12) != 1 )
      sub_1263B40(*v3, ", ");
    goto LABEL_36;
  }
  if ( *(_DWORD *)(v74 + 8) >> 8 )
LABEL_36:
    sub_1263B40(*v3, "...");
LABEL_37:
  sub_1549FC0(*v3, 0x29u);
  v19 = *(_BYTE *)(v2 + 32) >> 6;
  if ( v19 == 1 )
  {
    v20 = 18;
    v21 = "local_unnamed_addr";
  }
  else
  {
    v20 = 12;
    v21 = "unnamed_addr";
    if ( v19 != 2 )
      goto LABEL_40;
  }
  v22 = sub_1549FC0(*v3, 0x20u);
  sub_1549FF0(v22, v21, v20);
LABEL_40:
  if ( (unsigned __int8)sub_15602F0(&v78, 0xFFFFFFFFLL) )
  {
    v61 = sub_1263B40(*v3, " #");
    v62 = v3[4];
    v63 = v61;
    v64 = sub_1560250(&v78);
    v67 = sub_154F2E0(v62, v64, v65, v66);
    sub_16E7AB0(v63, v67);
  }
  if ( (*(_BYTE *)(v2 + 34) & 0x20) != 0 )
  {
    sub_1263B40(*v3, " section \"");
    v58 = *v3;
    v59 = 0;
    v60 = 0;
    if ( (*(_BYTE *)(v2 + 34) & 0x20) != 0 )
    {
      v60 = sub_15E61A0(v2, 0, v56, v57);
      v59 = v68;
    }
    sub_16D16F0(v60, v59, v58);
    sub_1549FC0(*v3, 0x22u);
  }
  sub_154B830(*v3, v2);
  if ( (unsigned int)(1 << (*(_DWORD *)(v2 + 32) >> 15)) >> 1 )
  {
    v55 = sub_1263B40(*v3, " align ");
    sub_16E7A90(v55, (unsigned int)(1 << (*(_DWORD *)(v2 + 32) >> 15)) >> 1);
  }
  v23 = *(_WORD *)(v2 + 18);
  if ( (v23 & 0x4000) != 0 )
  {
    v50 = sub_1263B40(*v3, " gc \"");
    v53 = sub_15E0FA0(v2, " gc \"", v51, v52);
    v54 = sub_16E7EE0(v50, *(const char **)v53, *(_QWORD *)(v53 + 8));
    sub_1549FC0(v54, 0x22u);
    v23 = *(_WORD *)(v2 + 18);
  }
  if ( (v23 & 2) != 0 )
  {
    sub_1263B40(*v3, " prefix ");
    v49 = (__int64 *)sub_15E3920(v2);
    sub_15520E0(v3, v49, 1);
    v23 = *(_WORD *)(v2 + 18);
  }
  if ( (v23 & 4) != 0 )
  {
    sub_1263B40(*v3, " prologue ");
    v48 = (__int64 *)sub_15E3950(v2);
    sub_15520E0(v3, v48, 1);
    v23 = *(_WORD *)(v2 + 18);
  }
  if ( (v23 & 8) != 0 )
  {
    sub_1263B40(*v3, " personality ");
    v36 = (__int64 *)sub_15E38F0(v2);
    sub_15520E0(v3, v36, 1);
  }
  if ( (unsigned __int8)sub_15E4F60(v2) )
  {
    sub_1549FC0(*v3, 0xAu);
  }
  else
  {
    v83 = (const char *)v85;
    v84 = 0x400000000LL;
    sub_1626D60(v2, &v83);
    sub_1550BA0(v3, (unsigned int *)&v83, " ", 1u);
    sub_1263B40(*v3, " {");
    for ( i = *(_QWORD *)(v2 + 80); v2 + 72 != i; i = *(_QWORD *)(i + 8) )
    {
      v35 = i - 24;
      if ( !i )
        v35 = 0;
      sub_1558F20(v3, v35);
    }
    sub_1552500(v3, v2);
    sub_1263B40(*v3, "}\n");
    if ( v83 != (const char *)v85 )
      _libc_free((unsigned __int64)v83);
  }
  return sub_154BF90(v3[4]);
}
