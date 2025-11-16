// Function: sub_1703900
// Address: 0x1703900
//
__int64 __fastcall sub_1703900(__int64 a1)
{
  __int64 v1; // r14
  __int64 *v2; // rax
  unsigned __int64 v3; // rbx
  unsigned int v4; // eax
  unsigned int v6; // r15d
  __int64 v7; // r12
  unsigned __int64 *v8; // r13
  unsigned int v9; // eax
  unsigned int v10; // eax
  unsigned int *v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // rcx
  int v14; // r8d
  int v15; // r9d
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // rax
  unsigned int v19; // r8d
  unsigned int v20; // eax
  _BYTE *v21; // rdi
  _BYTE *v22; // r15
  unsigned __int64 *v23; // rax
  unsigned __int64 v24; // rbx
  __int64 v25; // r13
  unsigned int v26; // r12d
  unsigned __int64 *v27; // r8
  __int64 v28; // rdi
  unsigned int v29; // esi
  __int64 *v30; // rcx
  __int64 v31; // r10
  unsigned int v32; // eax
  _DWORD *v33; // rax
  __int64 v34; // rax
  __int64 v35; // r9
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rsi
  unsigned int v40; // ecx
  __int64 *v41; // rdx
  __int64 v42; // r8
  unsigned int v43; // r8d
  __int64 v44; // rdi
  int v45; // ecx
  int v46; // r11d
  unsigned __int64 v47; // r15
  unsigned int *v48; // r14
  _BYTE *v49; // rbx
  unsigned int v50; // eax
  __int64 v51; // rdx
  unsigned __int8 *v52; // rax
  unsigned __int8 *v53; // rcx
  unsigned __int8 *v54; // rdx
  char v55; // si
  int v56; // edx
  int v57; // r9d
  unsigned __int64 *v58; // [rsp+0h] [rbp-150h]
  __int64 v59; // [rsp+8h] [rbp-148h]
  unsigned __int64 v60; // [rsp+10h] [rbp-140h]
  unsigned int v61; // [rsp+38h] [rbp-118h]
  unsigned int v62; // [rsp+3Ch] [rbp-114h]
  unsigned __int64 *v63; // [rsp+48h] [rbp-108h]
  __int64 v64; // [rsp+48h] [rbp-108h]
  __int64 v65; // [rsp+48h] [rbp-108h]
  __int64 v66; // [rsp+50h] [rbp-100h] BYREF
  __int64 v67; // [rsp+58h] [rbp-F8h] BYREF
  _BYTE *v68; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v69; // [rsp+68h] [rbp-E8h]
  _BYTE v70[16]; // [rsp+70h] [rbp-E0h] BYREF
  _QWORD *v71; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v72; // [rsp+88h] [rbp-C8h]
  _QWORD v73[8]; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE *v74; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v75; // [rsp+D8h] [rbp-78h]
  _BYTE v76[112]; // [rsp+E0h] [rbp-70h] BYREF

  v1 = a1;
  v71 = v73;
  v72 = 0x800000000LL;
  v75 = 0x800000000LL;
  v2 = *(__int64 **)(a1 + 72);
  v74 = v76;
  v3 = *(v2 - 3);
  v60 = v3;
  v59 = *v2;
  v4 = sub_16431D0(*v2);
  v62 = v4;
  if ( *(_BYTE *)(v3 + 16) <= 0x10u )
    return v62;
  v6 = v4;
  v7 = a1 + 80;
  v8 = (unsigned __int64 *)&v67;
  v9 = sub_16431D0(*(_QWORD *)v3);
  v73[0] = v3;
  v61 = v9;
  v68 = (_BYTE *)v3;
  LODWORD(v72) = 1;
  *(_DWORD *)sub_1703640(a1 + 80, (unsigned __int64 *)&v68) = v6;
  v10 = v72;
  if ( (_DWORD)v72 )
  {
    while ( 1 )
    {
      while ( *(_BYTE *)(v71[v10 - 1] + 16LL) <= 0x10u )
      {
        LODWORD(v72) = --v10;
        if ( !v10 )
          goto LABEL_30;
      }
      v66 = v71[v10 - 1];
      v11 = (unsigned int *)sub_1703640(v7, (unsigned __int64 *)&v66);
      v68 = v70;
      v69 = 0x200000000LL;
      sub_1702010(v66, (__int64)&v68, v12, v13, v14, v15);
      v18 = (unsigned int)v75;
      if ( (_DWORD)v75 )
      {
        if ( *(_QWORD *)&v74[8 * (unsigned int)v75 - 8] == v66 )
          break;
      }
      if ( (unsigned int)v75 >= HIDWORD(v75) )
      {
        sub_16CD150((__int64)&v74, v76, 0, 8, v16, v17);
        v18 = (unsigned int)v75;
      }
      *(_QWORD *)&v74[8 * v18] = v66;
      LODWORD(v75) = v75 + 1;
      v19 = *v11;
      v20 = *v11;
      if ( v11[1] >= *v11 )
        v20 = v11[1];
      v11[1] = v20;
      v21 = v68;
      v22 = &v68[8 * (unsigned int)v69];
      if ( v22 != v68 )
      {
        v23 = v8;
        v24 = (unsigned __int64)v68;
        v25 = v7;
        v26 = v19;
        v27 = v23;
        while ( 1 )
        {
          v36 = *(_QWORD *)v24;
          if ( *(_BYTE *)(*(_QWORD *)v24 + 16LL) > 0x17u )
            break;
LABEL_22:
          v24 += 8LL;
          if ( v22 == (_BYTE *)v24 )
          {
            v21 = v68;
            v7 = v25;
            v8 = v27;
            goto LABEL_27;
          }
        }
        v37 = *(unsigned int *)(v1 + 104);
        v67 = *(_QWORD *)v24;
        if ( (_DWORD)v37 )
        {
          v28 = *(_QWORD *)(v1 + 88);
          v29 = (v37 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
          v30 = (__int64 *)(v28 + 16LL * v29);
          v31 = *v30;
          if ( v36 != *v30 )
          {
            v45 = 1;
            while ( v31 != -8 )
            {
              v46 = v45 + 1;
              v29 = (v37 - 1) & (v45 + v29);
              v30 = (__int64 *)(v28 + 16LL * v29);
              v31 = *v30;
              if ( v36 == *v30 )
                goto LABEL_16;
              v45 = v46;
            }
            goto LABEL_25;
          }
LABEL_16:
          if ( v30 == (__int64 *)(v28 + 16 * v37) )
            goto LABEL_25;
          v32 = *(_DWORD *)(*(_QWORD *)(v1 + 112) + 24LL * *((unsigned int *)v30 + 2) + 8);
        }
        else
        {
LABEL_25:
          v32 = 0;
        }
        if ( v26 > v32 )
        {
          v63 = v27;
          v33 = (_DWORD *)sub_1703640(v25, v27);
          v27 = v63;
          *v33 = v26;
          v34 = (unsigned int)v72;
          v35 = v67;
          if ( (unsigned int)v72 >= HIDWORD(v72) )
          {
            v58 = v63;
            v64 = v67;
            sub_16CD150((__int64)&v71, v73, 0, 8, (int)v27, v67);
            v34 = (unsigned int)v72;
            v27 = v58;
            v35 = v64;
          }
          v71[v34] = v35;
          LODWORD(v72) = v72 + 1;
        }
        goto LABEL_22;
      }
LABEL_27:
      if ( v21 != v70 )
        goto LABEL_28;
LABEL_29:
      v10 = v72;
      if ( !(_DWORD)v72 )
        goto LABEL_30;
    }
    LODWORD(v72) = v72 - 1;
    LODWORD(v75) = v75 - 1;
    v21 = &v68[8 * (unsigned int)v69];
    if ( v68 != v21 )
    {
      v65 = v1;
      v47 = (unsigned __int64)v68;
      v48 = v11;
      v49 = &v68[8 * (unsigned int)v69];
      do
      {
        if ( *(_BYTE *)(*(_QWORD *)v47 + 16LL) > 0x17u )
        {
          v67 = *(_QWORD *)v47;
          v50 = *(_DWORD *)(sub_1703640(v7, v8) + 4);
          if ( v48[1] >= v50 )
            v50 = v48[1];
          v48[1] = v50;
        }
        v47 += 8LL;
      }
      while ( v49 != (_BYTE *)v47 );
      v1 = v65;
      v21 = v68;
    }
    if ( v21 == v70 )
      goto LABEL_29;
LABEL_28:
    _libc_free((unsigned __int64)v21);
    goto LABEL_29;
  }
LABEL_30:
  v38 = *(unsigned int *)(v1 + 104);
  if ( !(_DWORD)v38 )
    goto LABEL_55;
  v39 = *(_QWORD *)(v1 + 88);
  v40 = (v38 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
  v41 = (__int64 *)(v39 + 16LL * v40);
  v42 = *v41;
  if ( v60 != *v41 )
  {
    v56 = 1;
    while ( v42 != -8 )
    {
      v57 = v56 + 1;
      v40 = (v38 - 1) & (v56 + v40);
      v41 = (__int64 *)(v39 + 16LL * v40);
      v42 = *v41;
      if ( v60 == *v41 )
        goto LABEL_32;
      v56 = v57;
    }
    goto LABEL_55;
  }
LABEL_32:
  if ( v41 == (__int64 *)(v39 + 16 * v38) )
  {
LABEL_55:
    v43 = 0;
LABEL_56:
    v51 = *(_QWORD *)(v1 + 8);
    v52 = *(unsigned __int8 **)(v51 + 24);
    v53 = &v52[*(unsigned int *)(v51 + 32)];
    if ( v52 != v53 )
    {
      v54 = *(unsigned __int8 **)(v51 + 24);
      do
      {
        if ( v61 == *v54 )
        {
          v55 = 1;
          goto LABEL_62;
        }
        ++v54;
      }
      while ( v53 != v54 );
      v55 = 0;
LABEL_62:
      while ( *v52 != v43 )
      {
        if ( v53 == ++v52 )
        {
          if ( *(_BYTE *)(v59 + 8) != 16 && v55 )
            v43 = v61;
          goto LABEL_63;
        }
      }
    }
    goto LABEL_63;
  }
  v43 = *(_DWORD *)(*(_QWORD *)(v1 + 112) + 24LL * *((unsigned int *)v41 + 2) + 12);
  if ( v62 < v43 )
  {
    v62 = v61;
    if ( *(_BYTE *)(v59 + 8) != 16 )
    {
      v44 = sub_15A9690(*(_QWORD *)(v1 + 8), *(_QWORD *)v59, v43);
      if ( v44 )
        v62 = sub_16431D0(v44);
    }
    goto LABEL_37;
  }
  if ( v43 != 1 )
    goto LABEL_56;
LABEL_63:
  v62 = v43;
LABEL_37:
  if ( v74 != v76 )
    _libc_free((unsigned __int64)v74);
  if ( v71 != v73 )
    _libc_free((unsigned __int64)v71);
  return v62;
}
