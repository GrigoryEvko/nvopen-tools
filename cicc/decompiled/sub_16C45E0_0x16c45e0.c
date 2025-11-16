// Function: sub_16C45E0
// Address: 0x16c45e0
//
void __fastcall sub_16C45E0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v10; // al
  unsigned __int8 v11; // al
  unsigned __int8 v12; // al
  unsigned __int8 v13; // al
  __int64 v14; // r12
  _BYTE *v15; // r12
  char *v16; // rax
  __int64 v17; // rax
  _BYTE *v18; // r15
  __int64 v19; // r14
  size_t v20; // r13
  char *v21; // r10
  __int64 v22; // rax
  __int64 v23; // rdx
  size_t v24; // rax
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rdi
  size_t v27; // r9
  __int64 v28; // rdi
  const void *v29; // r13
  unsigned __int64 *v30; // r14
  size_t v31; // r12
  unsigned __int64 v32; // r13
  __int64 v33; // rax
  unsigned __int64 *v34; // rax
  const char *v35; // rdx
  size_t v36; // r13
  size_t v37; // rax
  unsigned __int64 v38; // r14
  __int64 v39; // rax
  unsigned __int64 *v40; // rax
  unsigned __int64 v41; // rdx
  size_t v42; // r13
  size_t v43; // rax
  unsigned __int64 v44; // r8
  __int64 v45; // rax
  unsigned __int64 *v46; // rax
  const char *v48; // rdx
  size_t v49; // rcx
  size_t v50; // rax
  unsigned __int64 v51; // rax
  __int64 v52; // rdx
  size_t v53; // r8
  unsigned __int64 *v54; // rdx
  char *s; // [rsp+8h] [rbp-1A8h]
  __int64 v56; // [rsp+40h] [rbp-170h]
  size_t v57; // [rsp+40h] [rbp-170h]
  char *v58; // [rsp+48h] [rbp-168h]
  size_t v59; // [rsp+48h] [rbp-168h]
  int v60; // [rsp+48h] [rbp-168h]
  const char *v61; // [rsp+48h] [rbp-168h]
  unsigned __int64 v62; // [rsp+48h] [rbp-168h]
  const char *v63; // [rsp+48h] [rbp-168h]
  unsigned __int64 v64; // [rsp+48h] [rbp-168h]
  unsigned __int64 v65; // [rsp+48h] [rbp-168h]
  _BYTE *v66; // [rsp+50h] [rbp-160h] BYREF
  __int16 v67; // [rsp+60h] [rbp-150h]
  _BYTE *v68; // [rsp+70h] [rbp-140h] BYREF
  __int64 v69; // [rsp+78h] [rbp-138h]
  _BYTE v70[32]; // [rsp+80h] [rbp-130h] BYREF
  _BYTE *v71; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v72; // [rsp+A8h] [rbp-108h]
  _BYTE v73[32]; // [rsp+B0h] [rbp-100h] BYREF
  _BYTE *v74; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v75; // [rsp+D8h] [rbp-D8h]
  _BYTE v76[32]; // [rsp+E0h] [rbp-D0h] BYREF
  _BYTE *v77; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v78; // [rsp+108h] [rbp-A8h]
  _BYTE v79[32]; // [rsp+110h] [rbp-A0h] BYREF
  _BYTE *v80; // [rsp+130h] [rbp-80h] BYREF
  __int64 v81; // [rsp+138h] [rbp-78h]
  _BYTE v82[112]; // [rsp+140h] [rbp-70h] BYREF

  v68 = v70;
  v69 = 0x2000000000LL;
  v72 = 0x2000000000LL;
  v75 = 0x2000000000LL;
  v78 = 0x2000000000LL;
  v80 = v82;
  v71 = v73;
  v81 = 0x400000000LL;
  v10 = *(_BYTE *)(a3 + 16);
  v74 = v76;
  v77 = v79;
  if ( v10 > 1u )
  {
    if ( *(_BYTE *)(a3 + 17) != 1 || (unsigned __int8)(v10 - 3) > 3u )
    {
      sub_16E2F40(a3, &v68);
      v52 = (unsigned int)v81;
      v51 = (unsigned __int64)v68;
      v53 = (unsigned int)v69;
      if ( (unsigned int)v81 >= HIDWORD(v81) )
      {
        v57 = (unsigned int)v69;
        v64 = (unsigned __int64)v68;
        sub_16CD150(&v80, v82, 0, 16);
        v52 = (unsigned int)v81;
        v53 = v57;
        v51 = v64;
      }
      goto LABEL_91;
    }
    v48 = *(const char **)a3;
    if ( v10 != 5 )
    {
      if ( v10 == 6 )
      {
        v49 = *((unsigned int *)v48 + 2);
        v51 = *(_QWORD *)v48;
        goto LABEL_97;
      }
      if ( v10 == 3 )
      {
        v49 = 0;
        if ( v48 )
        {
          v63 = v48;
          v50 = strlen(*(const char **)a3);
          v48 = v63;
          v49 = v50;
        }
        v51 = (unsigned __int64)v48;
        goto LABEL_97;
      }
    }
    v51 = *(_QWORD *)v48;
    v49 = *((_QWORD *)v48 + 1);
LABEL_97:
    v53 = v49;
    v52 = 0;
LABEL_91:
    v54 = (unsigned __int64 *)&v80[16 * v52];
    *v54 = v51;
    v54[1] = v53;
    LODWORD(v81) = v81 + 1;
  }
  v11 = *(_BYTE *)(a4 + 16);
  if ( v11 <= 1u )
    goto LABEL_3;
  if ( *(_BYTE *)(a4 + 17) == 1 && (unsigned __int8)(v11 - 3) <= 3u )
  {
    v41 = *(_QWORD *)a4;
    if ( v11 != 5 )
    {
      if ( v11 == 6 )
      {
        v42 = *(unsigned int *)(v41 + 8);
        v44 = *(_QWORD *)v41;
        goto LABEL_78;
      }
      if ( v11 == 3 )
      {
        v42 = 0;
        if ( v41 )
        {
          v62 = v41;
          v43 = strlen((const char *)v41);
          v41 = v62;
          v42 = v43;
        }
        v44 = v41;
        goto LABEL_78;
      }
    }
    v44 = *(_QWORD *)v41;
    v42 = *(_QWORD *)(v41 + 8);
  }
  else
  {
    sub_16E2F40(a4, &v71);
    v42 = (unsigned int)v72;
    v44 = (unsigned __int64)v71;
  }
LABEL_78:
  v45 = (unsigned int)v81;
  if ( (unsigned int)v81 >= HIDWORD(v81) )
  {
    v65 = v44;
    sub_16CD150(&v80, v82, 0, 16);
    v45 = (unsigned int)v81;
    v44 = v65;
  }
  v46 = (unsigned __int64 *)&v80[16 * v45];
  *v46 = v44;
  v46[1] = v42;
  LODWORD(v81) = v81 + 1;
LABEL_3:
  v12 = *(_BYTE *)(a5 + 16);
  if ( v12 <= 1u )
    goto LABEL_4;
  if ( *(_BYTE *)(a5 + 17) == 1 && (unsigned __int8)(v12 - 3) <= 3u )
  {
    v35 = *(const char **)a5;
    if ( v12 != 5 )
    {
      if ( v12 == 6 )
      {
        v36 = *((unsigned int *)v35 + 2);
        v38 = *(_QWORD *)v35;
        goto LABEL_66;
      }
      if ( v12 == 3 )
      {
        v36 = 0;
        if ( v35 )
        {
          v61 = *(const char **)a5;
          v37 = strlen(*(const char **)a5);
          v35 = v61;
          v36 = v37;
        }
        v38 = (unsigned __int64)v35;
        goto LABEL_66;
      }
    }
    v38 = *(_QWORD *)v35;
    v36 = *((_QWORD *)v35 + 1);
  }
  else
  {
    sub_16E2F40(a5, &v74);
    v36 = (unsigned int)v75;
    v38 = (unsigned __int64)v74;
  }
LABEL_66:
  v39 = (unsigned int)v81;
  if ( (unsigned int)v81 >= HIDWORD(v81) )
  {
    sub_16CD150(&v80, v82, 0, 16);
    v39 = (unsigned int)v81;
  }
  v40 = (unsigned __int64 *)&v80[16 * v39];
  *v40 = v38;
  v40[1] = v36;
  LODWORD(v81) = v81 + 1;
LABEL_4:
  v13 = *(_BYTE *)(a6 + 16);
  if ( v13 > 1u )
  {
    if ( *(_BYTE *)(a6 + 17) == 1 && (unsigned __int8)(v13 - 3) <= 3u )
    {
      v30 = *(unsigned __int64 **)a6;
      if ( v13 != 5 )
      {
        if ( v13 == 6 )
        {
          v31 = *((unsigned int *)v30 + 2);
          v32 = *v30;
          goto LABEL_54;
        }
        if ( v13 == 3 )
        {
          v31 = 0;
          if ( v30 )
            v31 = strlen((const char *)v30);
          v32 = (unsigned __int64)v30;
          goto LABEL_54;
        }
      }
      v32 = *v30;
      v31 = v30[1];
    }
    else
    {
      sub_16E2F40(a6, &v77);
      v31 = (unsigned int)v78;
      v32 = (unsigned __int64)v77;
    }
LABEL_54:
    v33 = (unsigned int)v81;
    if ( (unsigned int)v81 >= HIDWORD(v81) )
    {
      sub_16CD150(&v80, v82, 0, 16);
      v33 = (unsigned int)v81;
    }
    v34 = (unsigned __int64 *)&v80[16 * v33];
    v34[1] = v31;
    *v34 = v32;
    v14 = (unsigned int)(v81 + 1);
    LODWORD(v81) = v81 + 1;
    goto LABEL_6;
  }
  v14 = (unsigned int)v81;
LABEL_6:
  v15 = &v80[16 * v14];
  if ( v80 == v15 )
    goto LABEL_28;
  v16 = "/";
  if ( !a2 )
    v16 = "\\/";
  s = v16;
  v56 = a1 + 16;
  v17 = a1;
  v18 = v80;
  v19 = v17;
  do
  {
    v23 = *(unsigned int *)(v19 + 8);
    if ( !(_DWORD)v23 )
    {
      if ( !*((_QWORD *)v18 + 1) )
        goto LABEL_11;
      goto LABEL_25;
    }
    if ( !sub_16C36C0(*(_BYTE *)(*(_QWORD *)v19 + v23 - 1), a2) )
    {
      if ( !*((_QWORD *)v18 + 1) )
      {
LABEL_20:
        v23 = *(unsigned int *)(v19 + 8);
        if ( (_DWORD)v23 )
        {
          v66 = v18;
          v67 = 261;
          if ( !(unsigned __int8)sub_16C44E0((__int64)&v66, a2) )
          {
            v22 = *(unsigned int *)(v19 + 8);
            if ( (unsigned int)v22 >= *(_DWORD *)(v19 + 12) )
            {
              sub_16CD150(v19, v56, 0, 1);
              v22 = *(unsigned int *)(v19 + 8);
            }
            *(_BYTE *)(*(_QWORD *)v19 + v22) = a2 == 0 ? 92 : 47;
            v23 = (unsigned int)(*(_DWORD *)(v19 + 8) + 1);
            *(_DWORD *)(v19 + 8) = v23;
            goto LABEL_11;
          }
          goto LABEL_26;
        }
LABEL_11:
        v20 = *((_QWORD *)v18 + 1);
        v21 = *(char **)v18;
        if ( v20 > (unsigned __int64)*(unsigned int *)(v19 + 12) - v23 )
        {
          v58 = *(char **)v18;
          sub_16CD150(v19, v56, v20 + v23, 1);
          v23 = *(unsigned int *)(v19 + 8);
          v21 = v58;
        }
        if ( v20 )
        {
          memcpy((void *)(v23 + *(_QWORD *)v19), v21, v20);
          LODWORD(v23) = *(_DWORD *)(v19 + 8);
        }
        *(_DWORD *)(v19 + 8) = v20 + v23;
        goto LABEL_16;
      }
LABEL_25:
      if ( sub_16C36C0(**(_BYTE **)v18, a2) )
      {
LABEL_26:
        v23 = *(unsigned int *)(v19 + 8);
        goto LABEL_11;
      }
      goto LABEL_20;
    }
    v24 = strlen(s);
    v25 = sub_16D24E0(v18, s, v24, 0);
    v26 = *((_QWORD *)v18 + 1);
    if ( v25 > v26 )
    {
      *(_DWORD *)(v19 + 8) = *(_DWORD *)(v19 + 8);
    }
    else
    {
      v27 = v26 - v25;
      v28 = *(unsigned int *)(v19 + 8);
      v29 = (const void *)(*(_QWORD *)v18 + v25);
      if ( v27 > (unsigned __int64)*(unsigned int *)(v19 + 12) - v28 )
      {
        v59 = v27;
        sub_16CD150(v19, v56, v27 + v28, 1);
        v28 = *(unsigned int *)(v19 + 8);
        v27 = v59;
      }
      if ( v27 )
      {
        v60 = v27;
        memcpy((void *)(*(_QWORD *)v19 + v28), v29, v27);
        LODWORD(v28) = *(_DWORD *)(v19 + 8);
        LODWORD(v27) = v60;
      }
      *(_DWORD *)(v19 + 8) = v27 + v28;
    }
LABEL_16:
    v18 += 16;
  }
  while ( v15 != v18 );
  v15 = v80;
LABEL_28:
  if ( v15 != v82 )
    _libc_free((unsigned __int64)v15);
  if ( v77 != v79 )
    _libc_free((unsigned __int64)v77);
  if ( v74 != v76 )
    _libc_free((unsigned __int64)v74);
  if ( v71 != v73 )
    _libc_free((unsigned __int64)v71);
  if ( v68 != v70 )
    _libc_free((unsigned __int64)v68);
}
