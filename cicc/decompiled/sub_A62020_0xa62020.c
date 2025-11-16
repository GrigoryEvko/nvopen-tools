// Function: sub_A62020
// Address: 0xa62020
//
void __fastcall sub_A62020(__int64 *a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rdi
  char v7; // al
  __int64 v8; // rbx
  __int64 v9; // r8
  char v10; // al
  unsigned __int8 v11; // di
  char v12; // al
  size_t v13; // rdx
  char *v14; // rsi
  __int64 v15; // rax
  unsigned int v16; // edx
  char v17; // al
  const char *v18; // rsi
  __int64 v19; // rsi
  unsigned int v20; // eax
  int v21; // r15d
  __int64 v22; // rax
  __int64 v23; // rax
  __int16 v24; // r15
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // r15
  int v28; // eax
  int v29; // r15d
  __int64 v30; // r15
  __int64 v31; // rsi
  __int64 v32; // rdi
  __int64 v33; // r15
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rdx
  unsigned int v37; // [rsp+Ch] [rbp-A4h]
  _QWORD v38[4]; // [rsp+10h] [rbp-A0h] BYREF
  _QWORD *v39; // [rsp+30h] [rbp-80h] BYREF
  __int64 v40; // [rsp+38h] [rbp-78h]
  _QWORD v41[14]; // [rsp+40h] [rbp-70h] BYREF

  v3 = a2;
  if ( (unsigned __int8)sub_B2F600(a2) )
    sub_904010(*a1, "; Materializable\n");
  v4 = a1[4];
  v5 = *(_QWORD *)(a2 + 40);
  v6 = *a1;
  v38[1] = a1 + 5;
  v38[2] = v4;
  v38[3] = v5;
  v38[0] = off_4979428;
  sub_A5A730(v6, a2, (__int64)v38);
  sub_904010(*a1, " = ");
  v7 = sub_B2FC80(a2);
  LOBYTE(a2) = *(_BYTE *)(a2 + 32);
  if ( v7 )
  {
    LODWORD(a2) = a2 & 0xF;
    if ( (_DWORD)a2 )
      goto LABEL_5;
    sub_904010(*a1, "external ");
    LODWORD(a2) = *(unsigned __int8 *)(v3 + 32);
  }
  LOBYTE(a2) = a2 & 0xF;
LABEL_5:
  v8 = *a1;
  sub_A51210((__int64)&v39, (unsigned __int8)a2);
  sub_CB6200(v8, v39, v40);
  if ( v39 != v41 )
    j_j___libc_free_0(v39, v41[0] + 1LL);
  sub_A518A0(v3, *a1);
  v9 = *a1;
  v10 = (*(_BYTE *)(v3 + 32) >> 4) & 3;
  if ( v10 == 1 )
  {
    sub_904010(*a1, "hidden ");
    v9 = *a1;
  }
  else if ( v10 == 2 )
  {
    sub_904010(*a1, "protected ");
    v9 = *a1;
  }
  v11 = *(_BYTE *)(v3 + 33);
  if ( (v11 & 3) == 1 )
  {
    sub_904010(v9, "dllimport ");
    v9 = *a1;
    v11 = *(_BYTE *)(v3 + 33);
  }
  else if ( (v11 & 3) == 2 )
  {
    sub_904010(v9, "dllexport ");
    v9 = *a1;
    v11 = *(_BYTE *)(v3 + 33);
  }
  sub_A513A0((v11 >> 2) & 7, v9);
  v12 = *(_BYTE *)(v3 + 32) >> 6;
  if ( v12 == 1 )
  {
    v13 = 18;
    v14 = "local_unnamed_addr";
LABEL_15:
    v15 = sub_A51340(*a1, v14, v13);
    sub_A51310(v15, 0x20u);
    v16 = *(_DWORD *)(*(_QWORD *)(v3 + 8) + 8LL) >> 8;
    if ( !v16 )
      goto LABEL_16;
LABEL_27:
    v37 = v16;
    v22 = sub_904010(*a1, "addrspace(");
    v23 = sub_CB59D0(v22, v37);
    sub_904010(v23, ") ");
    v17 = *(_BYTE *)(v3 + 80);
    if ( (v17 & 2) == 0 )
      goto LABEL_17;
    goto LABEL_28;
  }
  v13 = 12;
  v14 = "unnamed_addr";
  if ( v12 == 2 )
    goto LABEL_15;
  if ( v12 )
    BUG();
  v16 = *(_DWORD *)(*(_QWORD *)(v3 + 8) + 8LL) >> 8;
  if ( v16 )
    goto LABEL_27;
LABEL_16:
  v17 = *(_BYTE *)(v3 + 80);
  if ( (v17 & 2) == 0 )
    goto LABEL_17;
LABEL_28:
  sub_904010(*a1, "externally_initialized ");
  v17 = *(_BYTE *)(v3 + 80);
LABEL_17:
  v18 = "constant ";
  if ( (v17 & 1) == 0 )
    v18 = "global ";
  sub_904010(*a1, v18);
  v19 = *(_QWORD *)(v3 + 24);
  sub_A57EC0((__int64)(a1 + 5), v19, *a1);
  if ( !(unsigned __int8)sub_B2FC80(v3) )
  {
    sub_A51310(*a1, 0x20u);
    v19 = *(_QWORD *)(v3 - 32);
    sub_A5B360(a1, v19, 0);
  }
  if ( (*(_BYTE *)(v3 + 35) & 4) == 0 )
  {
    if ( *(char *)(v3 + 33) >= 0 )
      goto LABEL_23;
    goto LABEL_52;
  }
  sub_904010(*a1, ", section \"");
  v30 = *a1;
  v31 = 0;
  v32 = 0;
  if ( (*(_BYTE *)(v3 + 35) & 4) != 0 )
  {
    v32 = sub_B31D10(v3);
    v31 = v36;
  }
  sub_C92400(v32, v31, v30);
  v19 = 34;
  sub_A51310(*a1, 0x22u);
  if ( *(char *)(v3 + 33) < 0 )
  {
LABEL_52:
    sub_904010(*a1, ", partition \"");
    v33 = *a1;
    v34 = sub_B30A70(v3);
    sub_C92400(v34, v35, v33);
    v19 = 34;
    sub_A51310(*a1, 0x22u);
  }
LABEL_23:
  v20 = *(unsigned __int16 *)(v3 + 34);
  LOWORD(v20) = (unsigned __int16)v20 >> 1;
  v21 = (v20 >> 6) & 7;
  if ( v21 )
  {
    sub_904010(*a1, ", code_model \"");
    switch ( v21 )
    {
      case 1:
        sub_904010(*a1, "tiny");
        break;
      case 2:
        sub_904010(*a1, "small");
        break;
      case 3:
        sub_904010(*a1, "kernel");
        break;
      case 4:
        sub_904010(*a1, "medium");
        break;
      case 5:
        sub_904010(*a1, "large");
        break;
      default:
        break;
    }
    v19 = 34;
    sub_A51310(*a1, 0x22u);
  }
  if ( (*(_BYTE *)(v3 + 34) & 1) != 0 )
  {
    v29 = *(_DWORD *)sub_B31490(v3, v19);
    if ( (v29 & 1) != 0 )
    {
      sub_904010(*a1, ", no_sanitize_address");
      if ( (v29 & 2) == 0 )
      {
LABEL_46:
        if ( (v29 & 4) == 0 )
          goto LABEL_47;
LABEL_59:
        sub_904010(*a1, ", sanitize_memtag");
LABEL_47:
        if ( (v29 & 8) != 0 )
          sub_904010(*a1, ", sanitize_address_dyninit");
        goto LABEL_32;
      }
    }
    else if ( (v29 & 2) == 0 )
    {
      goto LABEL_46;
    }
    sub_904010(*a1, ", no_sanitize_hwaddress");
    if ( (v29 & 4) == 0 )
      goto LABEL_47;
    goto LABEL_59;
  }
LABEL_32:
  sub_A550E0(*a1, v3);
  v24 = (*(_WORD *)(v3 + 34) >> 1) & 0x3F;
  if ( v24 )
  {
    v25 = sub_904010(*a1, ", align ");
    sub_CB59D0(v25, 1LL << ((unsigned __int8)v24 - 1));
  }
  v39 = v41;
  v40 = 0x400000000LL;
  sub_B9A9D0(v3, &v39);
  sub_A5C960(a1, (unsigned int *)&v39, ", ", 2u);
  v26 = *(_QWORD *)(v3 + 72);
  if ( v26 )
  {
    v27 = sub_904010(*a1, " #");
    v28 = sub_A5A580(a1[4], v26);
    sub_CB59F0(v27, v28);
  }
  sub_A61E50(a1, v3);
  if ( v39 != v41 )
    _libc_free(v39, v3);
}
