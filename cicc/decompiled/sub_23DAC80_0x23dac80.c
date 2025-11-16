// Function: sub_23DAC80
// Address: 0x23dac80
//
__int64 __fastcall sub_23DAC80(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // rax
  __int64 v3; // rbx
  unsigned int v4; // eax
  unsigned int v6; // r15d
  __int64 v7; // r12
  __int64 *v8; // r13
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned int v16; // eax
  _BYTE *v17; // rdx
  unsigned int *v18; // r15
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rcx
  __int64 v24; // rax
  unsigned __int8 *v25; // rbx
  __int64 v26; // rcx
  unsigned __int64 v27; // rdx
  unsigned int v28; // eax
  _BYTE *v29; // rdi
  _QWORD *v30; // r15
  __int64 *v31; // rax
  _QWORD *v32; // rbx
  __int64 v33; // r13
  unsigned int v34; // r12d
  __int64 v35; // rdi
  unsigned int v36; // esi
  __int64 v37; // r10
  __int64 v38; // rdx
  __int64 v39; // rax
  unsigned int v40; // eax
  _DWORD *v41; // rax
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  _BYTE *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rsi
  unsigned int v47; // ecx
  __int64 *v48; // rax
  __int64 v49; // r8
  unsigned int v50; // edx
  unsigned int v51; // r11d
  __int64 v52; // rdx
  _BYTE **v53; // rbx
  unsigned int *v54; // r14
  _BYTE **v55; // r15
  unsigned int v56; // eax
  __int64 v57; // rdi
  __int64 v58; // rax
  unsigned int v59; // r15d
  unsigned __int8 *v60; // rdi
  __int64 v61; // rsi
  __int64 v62; // r11
  unsigned __int8 *v63; // r10
  unsigned __int8 *v64; // rdx
  __int64 v65; // r9
  int v66; // eax
  int v67; // r9d
  __int64 v68; // [rsp+0h] [rbp-150h]
  __int64 v69; // [rsp+8h] [rbp-148h]
  __int64 v70; // [rsp+10h] [rbp-140h]
  unsigned int v71; // [rsp+38h] [rbp-118h]
  unsigned int v72; // [rsp+3Ch] [rbp-114h]
  __int64 v73; // [rsp+48h] [rbp-108h]
  _BYTE *v74; // [rsp+48h] [rbp-108h]
  __int64 v75; // [rsp+48h] [rbp-108h]
  unsigned __int8 *v76; // [rsp+50h] [rbp-100h] BYREF
  _BYTE *v77; // [rsp+58h] [rbp-F8h] BYREF
  __int64 v78; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v79; // [rsp+68h] [rbp-E8h]
  _BYTE v80[16]; // [rsp+70h] [rbp-E0h] BYREF
  _QWORD *v81; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v82; // [rsp+88h] [rbp-C8h]
  _QWORD v83[8]; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE *v84; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v85; // [rsp+D8h] [rbp-78h]
  _BYTE v86[112]; // [rsp+E0h] [rbp-70h] BYREF

  v1 = a1;
  v81 = v83;
  v82 = 0x800000000LL;
  v85 = 0x800000000LL;
  v2 = *(_QWORD *)(a1 + 80);
  v84 = v86;
  v3 = *(_QWORD *)(v2 - 32);
  v70 = v3;
  v69 = *(_QWORD *)(v2 + 8);
  v4 = sub_BCB060(v69);
  v72 = v4;
  if ( *(_BYTE *)v3 <= 0x15u )
    return v72;
  v6 = v4;
  v7 = a1 + 88;
  v8 = (__int64 *)&v77;
  v9 = sub_BCB060(*(_QWORD *)(v3 + 8));
  v83[0] = v3;
  v71 = v9;
  v78 = v3;
  LODWORD(v82) = 1;
  *(_DWORD *)sub_23DA960(a1 + 88, &v78, v10, v11, v12, v13) = v6;
  v16 = v82;
  if ( (_DWORD)v82 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v17 = (_BYTE *)v81[v16 - 1];
        if ( *v17 > 0x15u )
          break;
        LODWORD(v82) = --v16;
        if ( !v16 )
          goto LABEL_29;
      }
      v76 = (unsigned __int8 *)v81[v16 - 1];
      v18 = (unsigned int *)sub_23DA960(v7, (__int64 *)&v76, (__int64)v17, (__int64)v81, v14, v15);
      v78 = (__int64)v80;
      v79 = 0x200000000LL;
      sub_23D8620(v76, (__int64)&v78, v19, v20, v21, v22);
      if ( !(_DWORD)v85 )
        break;
      v23 = (__int64)v84;
      v24 = (unsigned int)v85;
      v25 = v76;
      if ( *(unsigned __int8 **)&v84[8 * (unsigned int)v85 - 8] == v76 )
      {
        v29 = (_BYTE *)v78;
        v52 = (unsigned int)(v85 - 1);
        LODWORD(v82) = v82 - 1;
        LODWORD(v85) = v85 - 1;
        if ( v78 + 8LL * (unsigned int)v79 != v78 )
        {
          v75 = v1;
          v53 = (_BYTE **)v78;
          v54 = v18;
          v55 = (_BYTE **)(v78 + 8LL * (unsigned int)v79);
          do
          {
            if ( **v53 > 0x1Cu )
            {
              v77 = *v53;
              v56 = *(_DWORD *)(sub_23DA960(v7, v8, v52, v23, v14, v15) + 4);
              if ( v54[1] >= v56 )
                v56 = v54[1];
              v54[1] = v56;
            }
            ++v53;
          }
          while ( v55 != v53 );
          v1 = v75;
          v29 = (_BYTE *)v78;
        }
        goto LABEL_26;
      }
      v26 = HIDWORD(v85);
      v27 = (unsigned int)v85 + 1LL;
      if ( v27 > HIDWORD(v85) )
        goto LABEL_43;
LABEL_10:
      *(_QWORD *)&v84[8 * v24] = v25;
      LODWORD(v85) = v85 + 1;
      v14 = *v18;
      v28 = *v18;
      if ( v18[1] >= (unsigned int)v14 )
        v28 = v18[1];
      v18[1] = v28;
      v29 = (_BYTE *)v78;
      v30 = (_QWORD *)(v78 + 8LL * (unsigned int)v79);
      if ( v30 != (_QWORD *)v78 )
      {
        v31 = v8;
        v32 = (_QWORD *)v78;
        v33 = v7;
        v34 = v14;
        v14 = (__int64)v31;
        while ( 1 )
        {
          v44 = (_BYTE *)*v32;
          if ( *(_BYTE *)*v32 > 0x1Cu )
            break;
LABEL_21:
          if ( v30 == ++v32 )
          {
            v29 = (_BYTE *)v78;
            v7 = v33;
            v8 = (__int64 *)v14;
            goto LABEL_26;
          }
        }
        v38 = *(unsigned int *)(v1 + 112);
        v77 = (_BYTE *)*v32;
        if ( (_DWORD)v38 )
        {
          v15 = (unsigned int)(v38 - 1);
          v35 = *(_QWORD *)(v1 + 96);
          v36 = v15 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
          v26 = v35 + 16LL * v36;
          v37 = *(_QWORD *)v26;
          if ( v44 != *(_BYTE **)v26 )
          {
            v26 = 1;
            while ( v37 != -4096 )
            {
              v51 = v26 + 1;
              v36 = v15 & (v26 + v36);
              v26 = v35 + 16LL * v36;
              v37 = *(_QWORD *)v26;
              if ( v44 == *(_BYTE **)v26 )
                goto LABEL_15;
              v26 = v51;
            }
            goto LABEL_24;
          }
LABEL_15:
          v38 = v35 + 16 * v38;
          if ( v26 == v38 )
            goto LABEL_24;
          v39 = *(unsigned int *)(v26 + 8);
          v26 = *(_QWORD *)(v1 + 120);
          v40 = *(_DWORD *)(v26 + 24 * v39 + 8);
        }
        else
        {
LABEL_24:
          v40 = 0;
        }
        if ( v34 > v40 )
        {
          v73 = v14;
          v41 = (_DWORD *)sub_23DA960(v33, (__int64 *)v14, v38, v26, v14, v15);
          v14 = v73;
          *v41 = v34;
          v42 = (unsigned int)v82;
          v26 = HIDWORD(v82);
          v15 = (__int64)v77;
          v43 = (unsigned int)v82 + 1LL;
          if ( v43 > HIDWORD(v82) )
          {
            v68 = v73;
            v74 = v77;
            sub_C8D5F0((__int64)&v81, v83, v43, 8u, v14, (__int64)v77);
            v42 = (unsigned int)v82;
            v14 = v68;
            v15 = (__int64)v74;
          }
          v81[v42] = v15;
          LODWORD(v82) = v82 + 1;
        }
        goto LABEL_21;
      }
LABEL_26:
      if ( v29 != v80 )
        _libc_free((unsigned __int64)v29);
      v16 = v82;
      if ( !(_DWORD)v82 )
        goto LABEL_29;
    }
    v24 = 0;
    v26 = HIDWORD(v85);
    v25 = v76;
    v27 = 1;
    if ( HIDWORD(v85) )
      goto LABEL_10;
LABEL_43:
    sub_C8D5F0((__int64)&v84, v86, v27, 8u, v14, v15);
    v24 = (unsigned int)v85;
    goto LABEL_10;
  }
LABEL_29:
  v45 = *(unsigned int *)(v1 + 112);
  v46 = *(_QWORD *)(v1 + 96);
  if ( !(_DWORD)v45 )
    goto LABEL_54;
  v47 = (v45 - 1) & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
  v48 = (__int64 *)(v46 + 16LL * v47);
  v49 = *v48;
  if ( v70 != *v48 )
  {
    v66 = 1;
    while ( v49 != -4096 )
    {
      v67 = v66 + 1;
      v47 = (v45 - 1) & (v66 + v47);
      v48 = (__int64 *)(v46 + 16LL * v47);
      v49 = *v48;
      if ( v70 == *v48 )
        goto LABEL_31;
      v66 = v67;
    }
    goto LABEL_54;
  }
LABEL_31:
  if ( v48 == (__int64 *)(v46 + 16 * v45) )
  {
LABEL_54:
    v72 = 0;
LABEL_55:
    v58 = *(_QWORD *)(v1 + 16);
    v78 = v71;
    v59 = v71;
    v60 = *(unsigned __int8 **)(v58 + 32);
    v61 = (__int64)&v60[*(_QWORD *)(v58 + 40)];
    sub_23D8560(v60, v61, &v78);
    v78 = v62;
    v64 = sub_23D8560(v63, v61, &v78);
    if ( (unsigned int)*(unsigned __int8 *)(v69 + 8) - 17 > 1 && (unsigned __int8 *)v61 == v64 )
    {
      if ( v61 == v65 )
        v59 = v72;
      v72 = v59;
    }
    goto LABEL_34;
  }
  v50 = *(_DWORD *)(*(_QWORD *)(v1 + 120) + 24LL * *((unsigned int *)v48 + 2) + 12);
  if ( v72 < v50 )
  {
    v72 = v71;
    if ( (unsigned int)*(unsigned __int8 *)(v69 + 8) - 17 > 1 )
    {
      v57 = sub_AE44B0(*(_QWORD *)(v1 + 16), *(_QWORD *)v69, v50);
      if ( v57 )
        v72 = sub_BCB060(v57);
    }
    goto LABEL_34;
  }
  v72 = 1;
  if ( v50 != 1 )
  {
    v72 = *(_DWORD *)(*(_QWORD *)(v1 + 120) + 24LL * *((unsigned int *)v48 + 2) + 12);
    goto LABEL_55;
  }
LABEL_34:
  if ( v84 != v86 )
    _libc_free((unsigned __int64)v84);
  if ( v81 != v83 )
    _libc_free((unsigned __int64)v81);
  return v72;
}
