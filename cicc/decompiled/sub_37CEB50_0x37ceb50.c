// Function: sub_37CEB50
// Address: 0x37ceb50
//
void __fastcall sub_37CEB50(__int64 a1, unsigned int a2, unsigned int a3, unsigned __int64 a4)
{
  __int64 v4; // rcx
  __int64 v5; // rdx
  int *v7; // rax
  __int64 v8; // r9
  int *v9; // r12
  unsigned int v10; // r13d
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rcx
  int *v14; // rdx
  __int64 v15; // rax
  int *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r8
  int *v20; // r12
  unsigned __int64 v21; // r14
  char v22; // al
  __int64 v23; // r13
  char v24; // r12
  unsigned int v25; // r13d
  unsigned __int64 v26; // r12
  __int64 v27; // rbx
  unsigned int *v28; // rax
  __int64 v29; // r14
  __int64 v30; // rax
  __int64 v31; // rsi
  unsigned int v32; // edx
  int *v33; // r15
  int v34; // ecx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // r12
  unsigned int v38; // ebx
  __int64 v39; // r13
  char v40; // al
  __int64 v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // r9
  __int64 v44; // r8
  unsigned __int64 v45; // r10
  __int64 v46; // rax
  unsigned __int64 *v47; // rax
  unsigned int *v48; // rdx
  int v49; // r8d
  __int64 v50; // r15
  int *v51; // rax
  int *v52; // rbx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  unsigned __int64 v56; // rbx
  unsigned __int64 v57; // rdi
  _BYTE *v58; // r8
  size_t v59; // rdx
  __int64 v60; // [rsp+18h] [rbp-F8h]
  int *v62; // [rsp+28h] [rbp-E8h]
  __int64 v63; // [rsp+38h] [rbp-D8h]
  __int64 v64; // [rsp+38h] [rbp-D8h]
  int *v65; // [rsp+40h] [rbp-D0h]
  unsigned __int64 v66; // [rsp+40h] [rbp-D0h]
  unsigned __int64 v67; // [rsp+48h] [rbp-C8h]
  char v68; // [rsp+50h] [rbp-C0h]
  unsigned int v69; // [rsp+58h] [rbp-B8h] BYREF
  unsigned int v70; // [rsp+5Ch] [rbp-B4h] BYREF
  char v71[40]; // [rsp+60h] [rbp-B0h] BYREF
  char v72; // [rsp+88h] [rbp-88h]
  _BYTE *v73; // [rsp+90h] [rbp-80h] BYREF
  __int64 v74; // [rsp+98h] [rbp-78h]
  _BYTE v75[24]; // [rsp+A0h] [rbp-70h] BYREF
  int v76; // [rsp+B8h] [rbp-58h] BYREF
  unsigned __int64 v77; // [rsp+C0h] [rbp-50h]
  int *v78; // [rsp+C8h] [rbp-48h]
  int *v79; // [rsp+D0h] [rbp-40h]
  __int64 v80; // [rsp+D8h] [rbp-38h]

  v69 = a3;
  v4 = *(_QWORD *)(a1 + 3136);
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL);
  v70 = a2;
  if ( *(_QWORD *)(v4 + 8LL * a2) != *(_QWORD *)(v5 + 8LL * a2) )
    return;
  v60 = a1 + 3408;
  v7 = sub_37BEF10(a1 + 3408, (int *)&v70);
  v9 = v7;
  v73 = v75;
  v74 = 0x400000000LL;
  v10 = v7[2];
  if ( v10 && v7 != (int *)&v73 )
  {
    v58 = v75;
    v59 = 4LL * v10;
    if ( v10 <= 4
      || (sub_C8D5F0((__int64)&v73, v75, v10, 4u, (__int64)v75, v8), v58 = v73, (v59 = 4LL * (unsigned int)v9[2]) != 0) )
    {
      memcpy(v58, *(const void **)v9, v59);
    }
    LODWORD(v74) = v10;
  }
  v77 = 0;
  v78 = &v76;
  v79 = &v76;
  v80 = 0;
  v11 = *((_QWORD *)v9 + 6);
  v76 = 0;
  if ( v11 )
  {
    v12 = sub_37B6830(v11, (__int64)&v76);
    v13 = v12;
    do
    {
      v14 = (int *)v12;
      v12 = *(_QWORD *)(v12 + 16);
    }
    while ( v12 );
    v78 = v14;
    v15 = v13;
    do
    {
      v16 = (int *)v15;
      v15 = *(_QWORD *)(v15 + 24);
    }
    while ( v15 );
    v17 = *((_QWORD *)v9 + 9);
    v79 = v16;
    v77 = v13;
    v80 = v17;
  }
  v20 = sub_37BEF10(v60, (int *)&v69);
  if ( v80 )
  {
    v65 = &v76;
    v21 = (unsigned __int64)v78;
    v22 = 0;
  }
  else
  {
    v21 = (unsigned __int64)v73;
    v65 = (int *)&v73[4 * (unsigned int)v74];
    v22 = 1;
  }
  v23 = (__int64)v20;
  v24 = v22;
  while ( v24 )
  {
    if ( v65 == (int *)v21 )
      goto LABEL_17;
    v48 = (unsigned int *)v21;
    v21 += 4LL;
    sub_2B5C0F0((__int64)v71, v23, v48, v18, v19);
  }
  while ( v65 != (int *)v21 )
  {
    sub_2B5C0F0((__int64)v71, v23, (unsigned int *)(v21 + 32), v18, v19);
    v21 = sub_220EF30(v21);
  }
LABEL_17:
  *(_QWORD *)(*(_QWORD *)(a1 + 3136) + 8LL * v69) = *(_QWORD *)(*(_QWORD *)(a1 + 3136) + 8LL * v70);
  v72 = 0;
  *(_DWORD *)v71 = v70;
  v25 = v69;
  if ( v80 )
  {
    v68 = 0;
    v26 = (unsigned __int64)v78;
    v62 = &v76;
  }
  else
  {
    v26 = (unsigned __int64)v73;
    v68 = 1;
    v62 = (int *)&v73[4 * (unsigned int)v74];
  }
  v27 = a1;
  if ( !v68 )
    goto LABEL_20;
  while ( 2 )
  {
    if ( v62 == (int *)v26 )
      goto LABEL_47;
    v28 = (unsigned int *)v26;
LABEL_22:
    v29 = *v28;
    v30 = *(unsigned int *)(v27 + 3464);
    v31 = *(_QWORD *)(v27 + 3448);
    if ( !(_DWORD)v30 )
      goto LABEL_46;
    v32 = (v30 - 1) & (37 * v29);
    v33 = (int *)(v31 + 88LL * v32);
    v34 = *v33;
    if ( (_DWORD)v29 != *v33 )
    {
      v49 = 1;
      while ( v34 != -1 )
      {
        v32 = (v30 - 1) & (v49 + v32);
        v33 = (int *)(v31 + 88LL * v32);
        v34 = *v33;
        if ( (_DWORD)v29 == *v33 )
          goto LABEL_24;
        ++v49;
      }
LABEL_46:
      v33 = (int *)(v31 + 88 * v30);
    }
LABEL_24:
    v35 = (unsigned int)v33[4];
    v36 = *((_QWORD *)v33 + 1);
    if ( v36 == v36 + 48 * v35 )
      goto LABEL_34;
    v66 = v26;
    v37 = v36 + 48 * v35;
    v63 = v27;
    v38 = v25;
    v39 = *((_QWORD *)v33 + 1);
    do
    {
      while ( 1 )
      {
        v40 = *(_BYTE *)(v39 + 40);
        if ( v40 != v72 )
          goto LABEL_28;
        if ( v40 )
          break;
        if ( *(_DWORD *)v39 == *(_DWORD *)v71 )
          goto LABEL_27;
LABEL_28:
        v39 += 48;
        if ( v37 == v39 )
          goto LABEL_33;
      }
      if ( (unsigned __int8)sub_2EAB6C0(v39, v71) )
      {
LABEL_27:
        *(_DWORD *)v39 = v38;
        *(_BYTE *)(v39 + 40) = 0;
        goto LABEL_28;
      }
      v39 += 48;
    }
    while ( v37 != v39 );
LABEL_33:
    v25 = v38;
    v26 = v66;
    v27 = v63;
LABEL_34:
    v41 = *(_QWORD *)(*(_QWORD *)(v27 + 32) + 32LL) + 48 * v29;
    sub_37BA660(*(_QWORD **)(v27 + 16), (__int64)(v33 + 2), v41, *(_QWORD *)(v41 + 40), (__int64)(v33 + 18));
    v44 = v42;
    v67 = v29 | v67 & 0xFFFFFFFF00000000LL;
    v45 = v67;
    v46 = *(unsigned int *)(v27 + 3480);
    if ( v46 + 1 > (unsigned __int64)*(unsigned int *)(v27 + 3484) )
    {
      v64 = v42;
      sub_C8D5F0(v27 + 3472, (const void *)(v27 + 3488), v46 + 1, 0x10u, v42, v43);
      v46 = *(unsigned int *)(v27 + 3480);
      v44 = v64;
      v45 = v67;
    }
    v47 = (unsigned __int64 *)(*(_QWORD *)(v27 + 3472) + 16 * v46);
    *v47 = v45;
    v47[1] = v44;
    ++*(_DWORD *)(v27 + 3480);
    if ( v68 )
    {
      v26 += 4LL;
      continue;
    }
    break;
  }
  v26 = sub_220EF30(v26);
LABEL_20:
  if ( v62 != (int *)v26 )
  {
    v28 = (unsigned int *)(v26 + 32);
    goto LABEL_22;
  }
LABEL_47:
  v50 = v27;
  v51 = sub_37BEF10(v60, (int *)&v70);
  v51[2] = 0;
  v52 = v51;
  sub_37B80B0(*((_QWORD *)v51 + 6));
  *((_QWORD *)v52 + 6) = 0;
  *((_QWORD *)v52 + 7) = v52 + 10;
  *((_QWORD *)v52 + 8) = v52 + 10;
  *((_QWORD *)v52 + 9) = 0;
  sub_37C43E0(v50, a4, 0, v53, v54, v55);
  if ( (_BYTE)qword_50512E8 )
    *(_QWORD *)(*(_QWORD *)(v50 + 3136) + 8LL * v70) = unk_5051170;
  v56 = v77;
  while ( v56 )
  {
    sub_37B80B0(*(_QWORD *)(v56 + 24));
    v57 = v56;
    v56 = *(_QWORD *)(v56 + 16);
    j_j___libc_free_0(v57);
  }
  if ( v73 != v75 )
    _libc_free((unsigned __int64)v73);
}
