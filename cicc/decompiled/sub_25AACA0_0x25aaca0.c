// Function: sub_25AACA0
// Address: 0x25aaca0
//
void __fastcall sub_25AACA0(__int64 *a1)
{
  __int64 v1; // rcx
  __int64 *v2; // r12
  __int64 *v3; // rbx
  __int64 v4; // r14
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 *v7; // rdx
  __int64 v8; // rsi
  __int64 v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // r12
  unsigned __int8 *v13; // r14
  __int64 v14; // rdx
  __int64 v15; // rsi
  char *v16; // rax
  unsigned __int8 *v17; // rsi
  __int64 v18; // r14
  __int64 v19; // rbx
  __int64 v20; // r14
  unsigned __int8 *v21; // rsi
  __int64 *v22; // rax
  __int64 *v23; // r12
  __int64 v24; // r13
  __int64 *v25; // rbx
  __int64 v26; // rax
  _QWORD *v27; // rbx
  _QWORD *v28; // r12
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  unsigned int v31; // eax
  size_t v32; // rdx
  _BYTE *v33; // rsi
  _BYTE *v34; // rax
  unsigned __int64 v35; // r15
  char *v36; // r14
  size_t v37; // rdx
  char *v38; // rdx
  __int64 v39; // rax
  __int64 *v40; // rax
  int v41; // edx
  _BYTE *v42; // rax
  unsigned __int64 v43; // r15
  int v44; // r10d
  size_t v45; // [rsp+0h] [rbp-740h]
  size_t v46; // [rsp+0h] [rbp-740h]
  size_t v47; // [rsp+0h] [rbp-740h]
  int v48; // [rsp+1Ch] [rbp-724h]
  char *v49[3]; // [rsp+28h] [rbp-718h] BYREF
  __int64 v50; // [rsp+40h] [rbp-700h] BYREF
  char *v51[3]; // [rsp+48h] [rbp-6F8h] BYREF
  __int64 (__fastcall **v52)(); // [rsp+60h] [rbp-6E0h] BYREF
  int v53; // [rsp+68h] [rbp-6D8h]
  unsigned __int64 v54[3]; // [rsp+70h] [rbp-6D0h] BYREF
  int v55; // [rsp+88h] [rbp-6B8h]
  unsigned __int64 v56[3]; // [rsp+90h] [rbp-6B0h] BYREF
  int v57; // [rsp+A8h] [rbp-698h]
  unsigned __int64 v58[4]; // [rsp+B0h] [rbp-690h] BYREF
  __int64 *v59; // [rsp+D0h] [rbp-670h]
  __int64 v60; // [rsp+D8h] [rbp-668h]
  int v61; // [rsp+E0h] [rbp-660h]
  char v62; // [rsp+E4h] [rbp-65Ch]
  char v63; // [rsp+E8h] [rbp-658h] BYREF
  size_t v64; // [rsp+1F0h] [rbp-550h] BYREF
  char *v65; // [rsp+1F8h] [rbp-548h] BYREF
  _QWORD *v66; // [rsp+200h] [rbp-540h]
  __int64 v67; // [rsp+208h] [rbp-538h]
  unsigned int v68; // [rsp+210h] [rbp-530h]
  __int64 v69; // [rsp+218h] [rbp-528h] BYREF
  char *v70; // [rsp+220h] [rbp-520h]
  __int64 v71; // [rsp+228h] [rbp-518h]
  int v72; // [rsp+230h] [rbp-510h]
  char v73; // [rsp+234h] [rbp-50Ch]
  char v74; // [rsp+238h] [rbp-508h] BYREF
  _BYTE *v75; // [rsp+2B8h] [rbp-488h]
  __int64 v76; // [rsp+2C0h] [rbp-480h]
  _BYTE v77[512]; // [rsp+2C8h] [rbp-478h] BYREF
  _BYTE *v78; // [rsp+4C8h] [rbp-278h]
  __int64 v79; // [rsp+4D0h] [rbp-270h]
  _BYTE v80[520]; // [rsp+4D8h] [rbp-268h] BYREF
  int v81; // [rsp+6E0h] [rbp-60h] BYREF
  unsigned __int64 v82; // [rsp+6E8h] [rbp-58h]
  int *v83; // [rsp+6F0h] [rbp-50h]
  int *v84; // [rsp+6F8h] [rbp-48h]
  __int64 v85; // [rsp+700h] [rbp-40h]

  v52 = (__int64 (__fastcall **)())&unk_4A1EF50;
  v64 = 3;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v50 = 2;
  memset(v51, 0, sizeof(v51));
  memset(v49, 0, sizeof(v49));
  v53 = 0;
  memset(v54, 0, sizeof(v54));
  v55 = 0;
  memset(v56, 0, sizeof(v56));
  v57 = 0;
  memset(v58, 0, 24);
  sub_25A5690((__int64)v54, v49);
  v55 = v50;
  sub_25A5690((__int64)v56, v51);
  v57 = v64;
  sub_25A5690((__int64)v58, &v65);
  if ( v49[0] )
    j_j___libc_free_0((unsigned __int64)v49[0]);
  if ( v51[0] )
    j_j___libc_free_0((unsigned __int64)v51[0]);
  if ( v65 )
    j_j___libc_free_0((unsigned __int64)v65);
  v2 = (__int64 *)a1[4];
  v3 = a1 + 3;
  v58[3] = 0;
  v52 = off_49D3B00;
  v64 = (size_t)&v52;
  v70 = &v74;
  v75 = v77;
  v76 = 0x4000000000LL;
  v79 = 0x4000000000LL;
  v59 = (__int64 *)&v63;
  v60 = 32;
  v61 = 0;
  v62 = 1;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v71 = 16;
  v72 = 0;
  v73 = 1;
  v78 = v80;
  v81 = 0;
  v82 = 0;
  v83 = &v81;
  v84 = &v81;
  v85 = 0;
  if ( v2 == a1 + 3 )
    goto LABEL_42;
  do
  {
    while ( 1 )
    {
      v4 = (__int64)(v2 - 7);
      if ( !v2 )
        v4 = 0;
      if ( !sub_B2FC80(v4) && !(unsigned __int8)sub_310F810(v4) )
        break;
      v2 = (__int64 *)v2[1];
      if ( v3 == v2 )
        goto LABEL_17;
    }
    v8 = *(_QWORD *)(v4 + 80);
    if ( v8 )
      v8 -= 24;
    sub_25A5A10((__int64)&v64, v8, v7, v1, v5, v6);
    v2 = (__int64 *)v2[1];
  }
  while ( v3 != v2 );
LABEL_17:
  LODWORD(v9) = v79;
  v10 = v76;
  if ( !(_DWORD)v79 )
    goto LABEL_41;
  if ( !(_DWORD)v76 )
    goto LABEL_32;
  while ( 1 )
  {
    v1 = v10--;
    v11 = *(_QWORD *)&v75[8 * v1 - 8];
    LODWORD(v76) = v10;
    v12 = *(_QWORD *)(v11 + 16);
    if ( !v12 )
      goto LABEL_30;
    do
    {
      v13 = *(unsigned __int8 **)(v12 + 24);
      v14 = *v13;
      if ( (unsigned __int8)v14 <= 0x1Cu )
        goto LABEL_28;
      v15 = *((_QWORD *)v13 + 5);
      if ( !v73 )
      {
        if ( !sub_C8CA60((__int64)&v69, v15) )
          goto LABEL_28;
        v14 = *v13;
        v17 = v13;
        if ( (_BYTE)v14 == 84 )
          goto LABEL_27;
        goto LABEL_98;
      }
      v16 = v70;
      v1 = (__int64)&v70[8 * HIDWORD(v71)];
      if ( v70 != (char *)v1 )
      {
        while ( v15 != *(_QWORD *)v16 )
        {
          v16 += 8;
          if ( (char *)v1 == v16 )
            goto LABEL_28;
        }
        v17 = *(unsigned __int8 **)(v12 + 24);
        if ( (_BYTE)v14 == 84 )
        {
LABEL_27:
          sub_25A8730(&v64, (__int64)v17);
          goto LABEL_28;
        }
LABEL_98:
        sub_25AA570((unsigned __int64)&v64, v17, v14, v1, v5, v6);
      }
LABEL_28:
      v12 = *(_QWORD *)(v12 + 8);
    }
    while ( v12 );
    v10 = v76;
LABEL_30:
    if ( !v10 )
    {
      LODWORD(v9) = v79;
      if ( !(_DWORD)v79 )
        break;
      do
      {
LABEL_32:
        v1 = (unsigned int)v9;
        v9 = (unsigned int)(v9 - 1);
        v18 = *(_QWORD *)&v78[8 * v1 - 8];
        LODWORD(v79) = v9;
        v19 = *(_QWORD *)(v18 + 56);
        v20 = v18 + 48;
        if ( v19 != v20 )
        {
          do
          {
            while ( 1 )
            {
              if ( !v19 )
                BUG();
              v21 = (unsigned __int8 *)(v19 - 24);
              if ( *(_BYTE *)(v19 - 24) != 84 )
                break;
              sub_25A8730(&v64, (__int64)v21);
              v19 = *(_QWORD *)(v19 + 8);
              if ( v20 == v19 )
                goto LABEL_38;
            }
            sub_25AA570((unsigned __int64)&v64, v21, v9, v1, v5, v6);
            v19 = *(_QWORD *)(v19 + 8);
          }
          while ( v20 != v19 );
LABEL_38:
          LODWORD(v9) = v79;
        }
      }
      while ( (_DWORD)v9 );
      v10 = v76;
LABEL_41:
      if ( !v10 )
        break;
    }
  }
LABEL_42:
  v50 = *a1;
  v22 = v59;
  if ( v62 )
    v23 = &v59[HIDWORD(v60)];
  else
    v23 = &v59[(unsigned int)v60];
  if ( v59 != v23 )
  {
    while ( 1 )
    {
      v24 = *v22;
      v25 = v22;
      if ( (unsigned __int64)*v22 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v23 == ++v22 )
        goto LABEL_47;
    }
LABEL_70:
    if ( v23 == v25 )
      goto LABEL_47;
    v30 = *(_QWORD *)(v24 - 32) & 0xFFFFFFFFFFFFFFF9LL;
    if ( v68 )
    {
      v31 = (v68 - 1) & (v30 ^ (v30 >> 9));
      v32 = (size_t)&v66[5 * v31];
      v1 = *(_QWORD *)v32;
      if ( v30 == *(_QWORD *)v32 )
      {
LABEL_73:
        if ( (_QWORD *)v32 != &v66[5 * v68] )
        {
          v33 = *(_BYTE **)(v32 + 16);
          v48 = *(_DWORD *)(v32 + 8);
          v34 = *(_BYTE **)(v32 + 24);
          v35 = v34 - v33;
          if ( v34 == v33 )
          {
            v37 = 0;
            v36 = 0;
          }
          else
          {
            if ( v35 > 0x7FFFFFFFFFFFFFF8LL )
              goto LABEL_103;
            v45 = v32;
            v36 = (char *)sub_22077B0(v35);
            v34 = *(_BYTE **)(v45 + 24);
            v33 = *(_BYTE **)(v45 + 16);
            v37 = v34 - v33;
          }
          if ( v33 != v34 )
            goto LABEL_78;
          goto LABEL_79;
        }
      }
      else
      {
        v41 = 1;
        while ( v1 != -2 )
        {
          v44 = v41 + 1;
          v31 = (v68 - 1) & (v41 + v31);
          v32 = (size_t)&v66[5 * v31];
          v1 = *(_QWORD *)v32;
          if ( v30 == *(_QWORD *)v32 )
            goto LABEL_73;
          v41 = v44;
        }
      }
    }
    v32 = v64;
    v33 = *(_BYTE **)(v64 + 80);
    v47 = v64;
    v48 = *(_DWORD *)(v64 + 72);
    v42 = *(_BYTE **)(v64 + 88);
    v43 = v42 - v33;
    if ( v42 == v33 )
    {
      v37 = 0;
      v36 = 0;
    }
    else
    {
      if ( v43 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_103:
        sub_4261EA(v30, v33, v32);
      v36 = (char *)sub_22077B0(v43);
      v42 = *(_BYTE **)(v47 + 88);
      v33 = *(_BYTE **)(v47 + 80);
      v37 = v42 - v33;
    }
    if ( v42 != v33 )
    {
LABEL_78:
      v46 = v37;
      memmove(v36, v33, v37);
      v37 = v46;
    }
LABEL_79:
    v38 = &v36[v37];
    if ( v48 == 1 && v36 != v38 )
    {
      v39 = sub_B8C880(&v50, (__int64 *)v36, (v38 - v36) >> 3, v1);
      sub_B99FD0(v24, 0x17u, v39);
    }
    if ( v36 )
      j_j___libc_free_0((unsigned __int64)v36);
    v40 = v25 + 1;
    if ( v25 + 1 == v23 )
      goto LABEL_47;
    do
    {
      v24 = *v40;
      v25 = v40;
      if ( (unsigned __int64)*v40 < 0xFFFFFFFFFFFFFFFELL )
        goto LABEL_70;
      ++v40;
    }
    while ( v23 != v40 );
  }
LABEL_47:
  sub_25A57F0(v82);
  if ( v78 != v80 )
    _libc_free((unsigned __int64)v78);
  if ( v75 != v77 )
    _libc_free((unsigned __int64)v75);
  if ( !v73 )
    _libc_free((unsigned __int64)v70);
  v26 = v68;
  if ( v68 )
  {
    v27 = v66;
    v28 = &v66[5 * v68];
    do
    {
      if ( *v27 != -16 && *v27 != -2 )
      {
        v29 = v27[2];
        if ( v29 )
          j_j___libc_free_0(v29);
      }
      v27 += 5;
    }
    while ( v28 != v27 );
    v26 = v68;
  }
  sub_C7D6A0((__int64)v66, 40 * v26, 8);
  if ( !v62 )
    _libc_free((unsigned __int64)v59);
  v52 = (__int64 (__fastcall **)())&unk_4A1EF50;
  if ( v58[0] )
    j_j___libc_free_0(v58[0]);
  if ( v56[0] )
    j_j___libc_free_0(v56[0]);
  if ( v54[0] )
    j_j___libc_free_0(v54[0]);
}
