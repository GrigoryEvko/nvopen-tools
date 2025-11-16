// Function: sub_2E769A0
// Address: 0x2e769a0
//
__int64 __fastcall sub_2E769A0(__int64 a1, int a2)
{
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned int v6; // r15d
  _BYTE *v7; // rbx
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rsi
  const char *v14; // rsi
  void *v15; // rdi
  __int64 *v16; // rdi
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rax
  void *v20; // rax
  void *v21; // rax
  __int64 *v22; // rbx
  void *v23; // rax
  __int64 *v24; // rbx
  __int64 v25; // r14
  void *v26; // rax
  __int64 v27; // r15
  __int64 v28; // r14
  void *v29; // rax
  __int64 v30; // r15
  __int64 *v31; // rcx
  __int64 v32; // rax
  char v33; // al
  __int64 *v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  void *v46; // rax
  __int64 *v47; // rdi
  void *v48; // rax
  __int64 *v49; // rax
  __int64 *v50; // [rsp+10h] [rbp-10D0h]
  __int64 *v51; // [rsp+10h] [rbp-10D0h]
  __int64 v52; // [rsp+18h] [rbp-10C8h]
  __int64 *v53; // [rsp+18h] [rbp-10C8h]
  __int64 *v54; // [rsp+18h] [rbp-10C8h]
  __int64 *v55; // [rsp+18h] [rbp-10C8h]
  __int64 *v56; // [rsp+20h] [rbp-10C0h] BYREF
  __int64 v57; // [rsp+28h] [rbp-10B8h]
  _BYTE v58[16]; // [rsp+30h] [rbp-10B0h] BYREF
  __int64 v59; // [rsp+40h] [rbp-10A0h] BYREF
  char *v60; // [rsp+48h] [rbp-1098h]
  __int64 v61; // [rsp+50h] [rbp-1090h]
  int v62; // [rsp+58h] [rbp-1088h]
  char v63; // [rsp+5Ch] [rbp-1084h]
  char v64; // [rsp+60h] [rbp-1080h] BYREF
  unsigned __int64 v65[2]; // [rsp+80h] [rbp-1060h] BYREF
  _QWORD v66[64]; // [rsp+90h] [rbp-1050h] BYREF
  _BYTE *v67; // [rsp+290h] [rbp-E50h]
  __int64 v68; // [rsp+298h] [rbp-E48h]
  _BYTE v69[3584]; // [rsp+2A0h] [rbp-E40h] BYREF
  __int64 v70; // [rsp+10A0h] [rbp-40h]

  v65[1] = 0x4000000001LL;
  v65[0] = (unsigned __int64)v66;
  v66[0] = 0;
  v67 = v69;
  v68 = 0x4000000000LL;
  v70 = 0;
  if ( !sub_2E708B0(a1) )
    goto LABEL_2;
  v11 = *(_QWORD *)(a1 + 104);
  v12 = *(unsigned int *)(a1 + 8);
  if ( !v11 )
  {
    if ( (_DWORD)v12 )
    {
      v48 = sub_CB72A0();
      sub_904010((__int64)v48, "Tree has no parent but has roots!\n");
      v49 = (__int64 *)sub_CB72A0();
      if ( v49[4] != v49[2] )
      {
        v6 = 0;
        sub_CB5AE0(v49);
        goto LABEL_3;
      }
      goto LABEL_2;
    }
LABEL_32:
    v14 = "Tree doesn't have a root!\n";
    v15 = sub_CB72A0();
LABEL_17:
    sub_904010((__int64)v15, v14);
    v16 = (__int64 *)sub_CB72A0();
    if ( v16[4] != v16[2] )
      sub_CB5AE0(v16);
    goto LABEL_2;
  }
  if ( !(_DWORD)v12 )
    goto LABEL_32;
  v13 = *(_QWORD *)(v11 + 328);
  if ( **(_QWORD **)a1 != v13 )
  {
    v14 = "Tree's root is not its parent's entry node!\n";
    v15 = sub_CB72A0();
    goto LABEL_17;
  }
  v56 = (__int64 *)v58;
  v57 = 0x100000000LL;
  sub_2E6D5A0((__int64)&v56, v13, v12, v3, v4, v5);
  v19 = *(unsigned int *)(a1 + 8);
  if ( v19 != (unsigned int)v57 )
    goto LABEL_20;
  v31 = *(__int64 **)a1;
  v32 = v19;
  v59 = 0;
  v61 = 4;
  v63 = 1;
  v50 = &v31[v32];
  v60 = &v64;
  v62 = 0;
  if ( v31 == &v31[v32] )
  {
    v34 = v56;
    v51 = &v56[v32];
    if ( v56 != &v56[v32] )
      goto LABEL_38;
    goto LABEL_47;
  }
  do
  {
    v54 = v31;
    sub_AE6EC0((__int64)&v59, *v31);
    v33 = v63;
    v31 = v54 + 1;
  }
  while ( v50 != v54 + 1 );
  v34 = v56;
  v31 = &v56[(unsigned int)v57];
  v51 = v31;
  if ( v56 == v31 )
  {
LABEL_44:
    if ( !v33 )
      _libc_free((unsigned __int64)v60);
    v34 = v56;
LABEL_47:
    if ( v34 != (__int64 *)v58 )
      _libc_free((unsigned __int64)v34);
    if ( !(unsigned __int8)sub_2E74C80((__int64)v65, a1, (__int64)v34, (__int64)v31, v17, v18) )
      goto LABEL_2;
    if ( !(unsigned __int8)sub_2E6DD50(a1) )
      goto LABEL_2;
    v6 = sub_2E722A0((__int64 **)a1, a1, v35, v36, v37);
    if ( !(_BYTE)v6 )
      goto LABEL_2;
    if ( (unsigned int)(a2 - 1) > 1 )
      goto LABEL_3;
    if ( (unsigned __int8)sub_2E759D0((__int64)v65, a1, v38, v39, v40, v41) )
    {
      if ( a2 == 2 )
        v6 = sub_2E76740((__int64)v65, a1, v42, v43, v44, v45);
      goto LABEL_3;
    }
LABEL_2:
    v6 = 0;
    goto LABEL_3;
  }
LABEL_38:
  while ( 1 )
  {
    v55 = v34;
    if ( !(unsigned __int8)sub_B19060((__int64)&v59, *v34, (__int64)v34, (__int64)v31) )
      break;
    v34 = v55 + 1;
    if ( v51 == v55 + 1 )
    {
      v33 = v63;
      goto LABEL_44;
    }
  }
  if ( !v63 )
    _libc_free((unsigned __int64)v60);
LABEL_20:
  v20 = sub_CB72A0();
  sub_904010((__int64)v20, "Tree has different roots than freshly computed ones!\n");
  v21 = sub_CB72A0();
  sub_904010((__int64)v21, "\tPDT roots: ");
  v22 = *(__int64 **)a1;
  v52 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v52 )
  {
    do
    {
      v28 = *v22;
      v29 = sub_CB72A0();
      v30 = (__int64)v29;
      if ( v28 )
        sub_2E39560(v28, (__int64)v29);
      else
        sub_904010((__int64)v29, "nullptr");
      ++v22;
      sub_904010(v30, ", ");
    }
    while ( (__int64 *)v52 != v22 );
  }
  v23 = sub_CB72A0();
  sub_904010((__int64)v23, "\n\tComputed roots: ");
  v24 = v56;
  v53 = &v56[(unsigned int)v57];
  if ( v56 != v53 )
  {
    do
    {
      v25 = *v24;
      v26 = sub_CB72A0();
      v27 = (__int64)v26;
      if ( v25 )
        sub_2E39560(v25, (__int64)v26);
      else
        sub_904010((__int64)v26, "nullptr");
      ++v24;
      sub_904010(v27, ", ");
    }
    while ( v53 != v24 );
  }
  v46 = sub_CB72A0();
  sub_904010((__int64)v46, "\n");
  v47 = (__int64 *)sub_CB72A0();
  if ( v47[4] != v47[2] )
    sub_CB5AE0(v47);
  if ( v56 == (__int64 *)v58 )
    goto LABEL_2;
  _libc_free((unsigned __int64)v56);
  v6 = 0;
LABEL_3:
  v7 = v67;
  v8 = (unsigned __int64)&v67[56 * (unsigned int)v68];
  if ( v67 != (_BYTE *)v8 )
  {
    do
    {
      v8 -= 56LL;
      v9 = *(_QWORD *)(v8 + 24);
      if ( v9 != v8 + 40 )
        _libc_free(v9);
    }
    while ( v7 != (_BYTE *)v8 );
    v8 = (unsigned __int64)v67;
  }
  if ( (_BYTE *)v8 != v69 )
    _libc_free(v8);
  if ( (_QWORD *)v65[0] != v66 )
    _libc_free(v65[0]);
  return v6;
}
