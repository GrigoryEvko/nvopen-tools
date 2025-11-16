// Function: sub_33644B0
// Address: 0x33644b0
//
__int64 __fastcall sub_33644B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  int v6; // eax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // r15
  __int64 v15; // rdi
  __int64 (__fastcall *v16)(__int64, __int64); // r8
  unsigned __int64 v17; // rbx
  int v18; // eax
  __int64 v19; // r9
  int v20; // edx
  __int64 *v21; // rax
  __int64 v22; // rcx
  int v23; // edx
  int v24; // edi
  char v25; // si
  _DWORD *v26; // rax
  int v27; // edi
  char v28; // al
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rcx
  __int64 v32; // rcx
  unsigned int v33; // r8d
  __int64 *v34; // r11
  __int64 v35; // rax
  char v36; // cl
  __int64 v37; // rdi
  bool v38; // zf
  __int64 *v39; // rcx
  __int64 v40; // rdi
  __int64 *v41; // rsi
  __int64 (__fastcall *v42)(__int64, __int64); // rax
  int v43; // esi
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // rdx
  int v47; // eax
  __int64 v48; // rdx
  _QWORD *v49; // rdx
  __int64 v50; // r11
  __int64 v51; // rax
  __int64 v52; // rdx
  int v53; // eax
  __int64 v54; // rdx
  _QWORD *v55; // rax
  unsigned int v56; // edx
  __int64 v57; // rax
  unsigned __int64 v58; // rdx
  __int64 v59; // rax
  __int64 *v60; // rdx
  unsigned int v61; // eax
  __int64 v62; // rcx
  unsigned int v63; // esi
  _QWORD *v64; // rdx
  __int64 v65; // rdi
  __int64 (__fastcall *v66)(__int64, __int64); // rcx
  __int64 *v67; // rdx
  __int64 v68; // r10
  __int64 v69; // rcx
  unsigned int v70; // edi
  _QWORD *v71; // rsi
  __int64 *v72; // rdx
  __int64 v73; // rdx
  _QWORD *v74; // rax
  unsigned int v75; // edx
  __int64 v76; // rax
  unsigned int v77; // edx
  __int64 v78; // rax
  unsigned int v79; // edx
  __int64 v80; // [rsp+0h] [rbp-40h]
  __int64 v81; // [rsp+8h] [rbp-38h]
  __int64 v82; // [rsp+8h] [rbp-38h]
  __int64 v83; // [rsp+8h] [rbp-38h]

  v3 = a2;
  v6 = *(_DWORD *)(a2 + 24);
  if ( v6 == 298 )
  {
    v48 = *(_QWORD *)(a2 + 40);
    v12 = *(_QWORD *)(v48 + 48);
    v13 = *(_QWORD *)(v48 + 40);
    v14 = *(unsigned int *)(v48 + 48);
  }
  else
  {
    if ( v6 != 299 )
    {
      if ( (unsigned int)(v6 - 366) > 1 )
      {
        *(_QWORD *)(a1 + 48) = 0;
        *(_OWORD *)a1 = 0;
        *(_OWORD *)(a1 + 16) = 0;
        *(_OWORD *)(a1 + 32) = 0;
      }
      else
      {
        v7 = *(_QWORD *)(a2 + 40);
        v8 = *(_QWORD *)(a2 + 104);
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
        v9 = *(_QWORD *)(v7 + 40);
        LODWORD(v7) = *(_DWORD *)(v7 + 48);
        *(_QWORD *)a1 = v9;
        *(_DWORD *)(a1 + 8) = v7;
        if ( v8 < 0 )
        {
          *(_BYTE *)(a1 + 40) = 0;
        }
        else
        {
          *(_QWORD *)(a1 + 32) = v8;
          *(_BYTE *)(a1 + 40) = 1;
        }
        *(_BYTE *)(a1 + 48) = 0;
      }
      return a1;
    }
    v11 = *(_QWORD *)(a2 + 40);
    v12 = *(_QWORD *)(v11 + 88);
    v13 = *(_QWORD *)(v11 + 80);
    v14 = *(unsigned int *)(v11 + 88);
  }
  v15 = *(_QWORD *)(a3 + 16);
  v16 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v15 + 2128LL);
  if ( v16 == sub_302E0D0 )
  {
    v17 = (unsigned int)v14 | v12 & 0xFFFFFFFF00000000LL;
  }
  else
  {
    v57 = ((__int64 (__fastcall *)(__int64, __int64, unsigned __int64))v16)(
            v15,
            v13,
            v12 & 0xFFFFFFFF00000000LL | (unsigned int)v14);
    v3 = a2;
    v17 = v58;
    v13 = v57;
    v14 = (unsigned int)v58;
  }
  v18 = (*(_WORD *)(v3 + 32) >> 7) & 7;
  if ( v18 != 1 )
  {
    v19 = 0;
    if ( v18 != 2 )
      goto LABEL_13;
    v51 = 80;
    if ( *(_DWORD *)(v3 + 24) != 298 )
      v51 = 120;
    v52 = *(_QWORD *)(*(_QWORD *)(v3 + 40) + v51);
    v53 = *(_DWORD *)(v52 + 24);
    if ( v53 == 11 || v53 == 35 )
    {
      v54 = *(_QWORD *)(v52 + 96);
      v55 = *(_QWORD **)(v54 + 24);
      v56 = *(_DWORD *)(v54 + 32);
      if ( v56 > 0x40 )
      {
        v19 = -*v55;
      }
      else
      {
        v19 = 0;
        if ( v56 )
          v19 = -((__int64)((_QWORD)v55 << (64 - (unsigned __int8)v56)) >> (64 - (unsigned __int8)v56));
      }
      goto LABEL_13;
    }
    goto LABEL_44;
  }
  v45 = 80;
  if ( *(_DWORD *)(v3 + 24) != 298 )
    v45 = 120;
  v46 = *(_QWORD *)(*(_QWORD *)(v3 + 40) + v45);
  v47 = *(_DWORD *)(v46 + 24);
  if ( v47 != 35 && v47 != 11 )
  {
LABEL_44:
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 40) = 1;
    *(_BYTE *)(a1 + 48) = 0;
    return a1;
  }
  v59 = *(_QWORD *)(v46 + 96);
  v60 = *(__int64 **)(v59 + 24);
  v61 = *(_DWORD *)(v59 + 32);
  if ( v61 > 0x40 )
  {
    v19 = *v60;
  }
  else
  {
    v19 = 0;
    if ( v61 )
      v19 = (__int64)((_QWORD)v60 << (64 - (unsigned __int8)v61)) >> (64 - (unsigned __int8)v61);
  }
  while ( 1 )
  {
LABEL_13:
    while ( 1 )
    {
      v20 = *(_DWORD *)(v13 + 24);
      if ( v20 == 187 )
        break;
      if ( v20 > 187 )
      {
        if ( (unsigned int)(v20 - 298) > 1 )
        {
LABEL_37:
          v43 = 0;
          v44 = 0;
          goto LABEL_38;
        }
        v27 = (*(_WORD *)(v13 + 32) >> 7) & 7;
        v28 = (v20 == 298) == (_DWORD)v14 && v27 != 0;
        if ( !v28 )
          goto LABEL_51;
        v29 = 120;
        v30 = *(_QWORD *)(v13 + 40);
        if ( v20 == 298 )
          v29 = 80;
        v31 = *(_QWORD *)(v30 + v29);
        v28 = *(_DWORD *)(v31 + 24) == 11 || *(_DWORD *)(v31 + 24) == 35;
        if ( !v28 )
          goto LABEL_51;
        v32 = *(_QWORD *)(v31 + 96);
        v33 = *(_DWORD *)(v32 + 32);
        v34 = *(__int64 **)(v32 + 24);
        if ( v33 <= 0x40 )
        {
          v35 = 0;
          if ( v33 )
            v35 = (__int64)((_QWORD)v34 << (64 - (unsigned __int8)v33)) >> (64 - (unsigned __int8)v33);
        }
        else
        {
          v35 = *v34;
        }
        v36 = v27 - 2;
        v37 = v19 + v35;
        v19 -= v35;
        v38 = (v36 & 0xFD) == 0;
        v39 = (__int64 *)(v30 + 80);
        if ( !v38 )
          v19 = v37;
        v40 = *(_QWORD *)(a3 + 16);
        v41 = (__int64 *)(v30 + 40);
        if ( v20 == 299 )
          v41 = v39;
        v42 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v40 + 2128LL);
        if ( v42 == sub_302E0D0 )
        {
          v13 = *v41;
          v14 = *((unsigned int *)v41 + 2);
        }
        else
        {
          v83 = v19;
          v78 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64 *))v42)(v40, *v41, v41[1], v39);
          v19 = v83;
          v13 = v78;
          v14 = v79;
        }
      }
      else
      {
        if ( v20 != 56 )
          goto LABEL_37;
        v21 = *(__int64 **)(v13 + 40);
        v22 = v21[5];
        v23 = *(_DWORD *)(v22 + 24);
        if ( v23 != 35 && v23 != 11 )
          goto LABEL_18;
        v62 = *(_QWORD *)(v22 + 96);
        v63 = *(_DWORD *)(v62 + 32);
        v64 = *(_QWORD **)(v62 + 24);
        if ( v63 <= 0x40 )
        {
          if ( v63 )
            v19 += (__int64)((_QWORD)v64 << (64 - (unsigned __int8)v63)) >> (64 - (unsigned __int8)v63);
        }
        else
        {
          v19 += *v64;
        }
        v65 = *(_QWORD *)(a3 + 16);
        v66 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v65 + 2128LL);
        if ( v66 != sub_302E0D0 )
          goto LABEL_84;
LABEL_66:
        v13 = *v21;
        v14 = *((unsigned int *)v21 + 2);
      }
    }
    v49 = *(_QWORD **)(v13 + 40);
    v50 = v49[5];
    v28 = *(_DWORD *)(v50 + 24) == 11 || *(_DWORD *)(v50 + 24) == 35;
    if ( !v28 )
      goto LABEL_51;
    v80 = v19;
    v81 = v49[5];
    v28 = sub_33DD210(a3, *v49, v49[1], *(_QWORD *)(v50 + 96) + 24LL, 0);
    v19 = v80;
    if ( !v28 )
      break;
    v73 = *(_QWORD *)(v81 + 96);
    v74 = *(_QWORD **)(v73 + 24);
    v75 = *(_DWORD *)(v73 + 32);
    if ( v75 > 0x40 )
    {
      v19 = *v74 + v80;
    }
    else if ( v75 )
    {
      v19 = ((__int64)((_QWORD)v74 << (64 - (unsigned __int8)v75)) >> (64 - (unsigned __int8)v75)) + v80;
    }
    v65 = *(_QWORD *)(a3 + 16);
    v66 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v65 + 2128LL);
    v21 = *(__int64 **)(v13 + 40);
    if ( v66 == sub_302E0D0 )
      goto LABEL_66;
LABEL_84:
    v82 = v19;
    v76 = ((__int64 (__fastcall *)(__int64, __int64, __int64))v66)(v65, *v21, v21[1]);
    v19 = v82;
    v13 = v76;
    v14 = v77;
  }
  if ( *(_DWORD *)(v13 + 24) != 56 )
  {
LABEL_51:
    v43 = 0;
    v44 = 0;
    goto LABEL_39;
  }
  v21 = *(__int64 **)(v13 + 40);
  v22 = v21[5];
  v23 = *(_DWORD *)(v22 + 24);
LABEL_18:
  if ( v23 != 58 )
  {
    v24 = *((_DWORD *)v21 + 12);
    v13 = *v21;
    v25 = 0;
    v14 = *((unsigned int *)v21 + 2);
    if ( v23 == 213 )
    {
      v26 = *(_DWORD **)(v22 + 40);
      v25 = 1;
      v22 = *(_QWORD *)v26;
      v24 = v26[2];
      v23 = *(_DWORD *)(*(_QWORD *)v26 + 24LL);
    }
    if ( v23 != 56
      || (v67 = *(__int64 **)(v22 + 40),
          v68 = v67[5],
          (v28 = *(_DWORD *)(v68 + 24) == 11 || *(_DWORD *)(v68 + 24) == 35) == 0) )
    {
      *(_QWORD *)a1 = v13;
      *(_DWORD *)(a1 + 8) = v14;
      *(_QWORD *)(a1 + 16) = v22;
      *(_DWORD *)(a1 + 24) = v24;
      *(_QWORD *)(a1 + 32) = v19;
      *(_BYTE *)(a1 + 40) = 1;
      *(_BYTE *)(a1 + 48) = v25;
      return a1;
    }
    v69 = *(_QWORD *)(v68 + 96);
    v70 = *(_DWORD *)(v69 + 32);
    v71 = *(_QWORD **)(v69 + 24);
    if ( v70 > 0x40 )
    {
      v19 += *v71;
    }
    else if ( v70 )
    {
      v19 += (__int64)((_QWORD)v71 << (64 - (unsigned __int8)v70)) >> (64 - (unsigned __int8)v70);
    }
    v44 = *v67;
    v43 = *((_DWORD *)v67 + 2);
    if ( *(_DWORD *)(*v67 + 24) == 213 )
    {
      v72 = *(__int64 **)(v44 + 40);
      v44 = *v72;
      v43 = *((_DWORD *)v72 + 2);
      goto LABEL_39;
    }
LABEL_38:
    v28 = 0;
LABEL_39:
    *(_QWORD *)a1 = v13;
    *(_QWORD *)(a1 + 16) = v44;
    *(_DWORD *)(a1 + 24) = v43;
    *(_QWORD *)(a1 + 8) = v17 & 0xFFFFFFFF00000000LL | v14;
    *(_QWORD *)(a1 + 32) = v19;
    *(_BYTE *)(a1 + 40) = 1;
    *(_BYTE *)(a1 + 48) = v28;
    return a1;
  }
  *(_QWORD *)a1 = v13;
  *(_QWORD *)(a1 + 32) = v19;
  *(_BYTE *)(a1 + 40) = 1;
  *(_QWORD *)(a1 + 8) = v14 | v17 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 48) = 0;
  return a1;
}
