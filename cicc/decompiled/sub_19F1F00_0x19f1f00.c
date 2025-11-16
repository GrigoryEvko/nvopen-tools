// Function: sub_19F1F00
// Address: 0x19f1f00
//
void __fastcall sub_19F1F00(__int64 a1, __int64 a2, __int64 a3)
{
  char v6; // al
  unsigned int v7; // ebx
  int v8; // r15d
  unsigned int v9; // esi
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r12
  char *v13; // rax
  __int64 ****v14; // rax
  __int64 v15; // rax
  __int64 v16; // r15
  __int64 v17; // r12
  __int64 v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // r9
  int v22; // esi
  unsigned int v23; // ecx
  __int64 *v24; // rax
  __int64 v25; // r10
  int v26; // ecx
  unsigned int v27; // esi
  unsigned int v28; // ecx
  unsigned int v29; // r9d
  unsigned int v30; // r10d
  __int64 v31; // r15
  __int64 v32; // rdx
  __int64 v33; // r11
  char v34; // r9
  __int64 v35; // r10
  unsigned int v36; // ecx
  __int64 v37; // rdi
  __int64 v38; // r8
  __int64 v39; // rdi
  __int64 v40; // rsi
  __int64 v41; // rsi
  __int64 v42; // rdi
  __int64 v43; // rbx
  __int64 v44; // rax
  __int64 v45; // rdx
  int v46; // r8d
  __int64 *v47; // r11
  __int64 v48; // rdx
  _DWORD *v49; // rax
  __int64 v50; // rbx
  __int64 v51; // rax
  unsigned __int8 v52; // dl
  __int64 v53; // r8
  __int64 v54; // r15
  unsigned int v55; // ebx
  __int64 v56; // r12
  int v57; // eax
  __int64 v58; // rdx
  __int64 v59; // rcx
  char v60; // cl
  __int64 v61; // rsi
  __int64 v62; // rdx
  unsigned __int8 v63; // al
  __int64 v64; // rax
  int v65; // r11d
  __int64 *v66; // r8
  __int64 v67; // rsi
  __int64 v68; // r9
  __int64 v69; // rdi
  __int64 v70; // r9
  __int64 v71; // rdx
  __int64 v72; // [rsp+8h] [rbp-158h]
  __int64 v73; // [rsp+8h] [rbp-158h]
  __int64 v74; // [rsp+10h] [rbp-150h] BYREF
  __int64 *v75; // [rsp+18h] [rbp-148h] BYREF
  __int64 v76; // [rsp+20h] [rbp-140h] BYREF
  __int64 v77; // [rsp+28h] [rbp-138h]
  _QWORD *v78; // [rsp+30h] [rbp-130h] BYREF
  unsigned int v79; // [rsp+38h] [rbp-128h]
  char v80; // [rsp+130h] [rbp-30h] BYREF

  v6 = *(_BYTE *)(a2 + 16);
  if ( v6 != 26 )
  {
    if ( v6 != 27 )
      goto LABEL_3;
    v13 = (char *)&v78;
    v76 = 0;
    v77 = 1;
    do
    {
      *(_QWORD *)v13 = -8;
      v13 += 16;
    }
    while ( v13 != &v80 );
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v14 = *(__int64 *****)(a2 - 8);
    else
      v14 = (__int64 ****)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v15 = sub_19E1ED0(a1, *v14);
    if ( *(_BYTE *)(v15 + 16) != 13 )
    {
      if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1 )
      {
        v16 = 48LL * (((*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1) - 1) + 72;
        v17 = 24;
        while ( 1 )
        {
          v19 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
          v20 = *(_QWORD *)(v19 + v17);
          v74 = v20;
          if ( (v77 & 1) != 0 )
          {
            v21 = &v78;
            v22 = 15;
          }
          else
          {
            v27 = v79;
            v21 = v78;
            if ( !v79 )
            {
              v28 = v77;
              ++v76;
              v24 = 0;
              v29 = ((unsigned int)v77 >> 1) + 1;
LABEL_27:
              v30 = 3 * v27;
              goto LABEL_28;
            }
            v22 = v79 - 1;
          }
          v23 = v22 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          v24 = &v21[2 * v23];
          v25 = *v24;
          if ( v20 != *v24 )
            break;
          v26 = *((_DWORD *)v24 + 2) + 1;
LABEL_21:
          *((_DWORD *)v24 + 2) = v26;
          v17 += 48;
          sub_19F1B70(a1, a3, v20);
          if ( v16 == v17 )
            goto LABEL_52;
        }
        v46 = 1;
        v47 = 0;
        while ( v25 != -8 )
        {
          if ( v47 || v25 != -16 )
            v24 = v47;
          v65 = v46 + 1;
          v23 = v22 & (v46 + v23);
          v66 = &v21[2 * v23];
          v25 = *v66;
          if ( v20 == *v66 )
          {
            v24 = &v21[2 * v23];
            v26 = *((_DWORD *)v66 + 2) + 1;
            goto LABEL_21;
          }
          v46 = v65;
          v47 = v24;
          v24 = &v21[2 * v23];
        }
        v28 = v77;
        v30 = 48;
        v27 = 16;
        if ( v47 )
          v24 = v47;
        ++v76;
        v29 = ((unsigned int)v77 >> 1) + 1;
        if ( (v77 & 1) == 0 )
        {
          v27 = v79;
          goto LABEL_27;
        }
LABEL_28:
        if ( v30 <= 4 * v29 )
        {
          v27 *= 2;
        }
        else if ( v27 - HIDWORD(v77) - v29 > v27 >> 3 )
        {
LABEL_30:
          LODWORD(v77) = (2 * (v28 >> 1) + 2) | v28 & 1;
          if ( *v24 != -8 )
            --HIDWORD(v77);
          *v24 = v20;
          v26 = 1;
          v20 = v74;
          *((_DWORD *)v24 + 2) = 0;
          goto LABEL_21;
        }
        sub_1917CA0((__int64)&v76, v27);
        sub_190F380((__int64)&v76, &v74, &v75);
        v24 = v75;
        v20 = v74;
        v28 = v77;
        goto LABEL_30;
      }
      goto LABEL_52;
    }
    v31 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v72 = ((*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1) - 1;
    v32 = v72 >> 2;
    if ( v72 >> 2 )
    {
      v33 = 4 * v32;
      v32 = 0;
      v34 = *(_BYTE *)(a2 + 23) & 0x40;
      v35 = a2 - 24LL * (unsigned int)v31;
      v36 = 8;
      while ( 1 )
      {
        v38 = v32 + 1;
        v41 = v35;
        if ( v34 )
          v41 = *(_QWORD *)(a2 - 8);
        v42 = *(_QWORD *)(v41 + 24LL * (v36 - 6));
        if ( v42 )
        {
          if ( v15 == v42 )
            goto LABEL_46;
        }
        v37 = *(_QWORD *)(v41 + 24LL * (v36 - 4));
        if ( v37 && v15 == v37 )
          goto LABEL_47;
        v38 = v32 + 3;
        v39 = *(_QWORD *)(v41 + 24LL * (v36 - 2));
        if ( v39 && v15 == v39 )
        {
          v38 = v32 + 2;
          goto LABEL_47;
        }
        v32 += 4;
        v40 = *(_QWORD *)(v41 + 24LL * v36);
        if ( v40 && v15 == v40 )
          goto LABEL_47;
        v36 += 8;
        if ( v32 == v33 )
        {
          v59 = v72 - v32;
          goto LABEL_81;
        }
      }
    }
    v59 = ((*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1) - 1;
LABEL_81:
    switch ( v59 )
    {
      case 2LL:
        v67 = v32;
        v60 = *(_BYTE *)(a2 + 23) & 0x40;
        break;
      case 3LL:
        v67 = v32 + 1;
        v60 = *(_BYTE *)(a2 + 23) & 0x40;
        if ( v60 )
          v68 = *(_QWORD *)(a2 - 8);
        else
          v68 = a2 - 24LL * (unsigned int)v31;
        v69 = *(_QWORD *)(v68 + 24LL * (unsigned int)(2 * (v32 + 1)));
        if ( v69 && v15 == v69 )
        {
LABEL_46:
          v38 = v32;
LABEL_47:
          if ( v38 != v72 && (_DWORD)v38 != -2 )
          {
            v43 = 24LL * (unsigned int)(2 * v38 + 3);
LABEL_50:
            v44 = sub_13CF970(a2);
            v45 = *(_QWORD *)(v44 + v43);
            if ( *(_QWORD *)(v44 + 24) == v45 )
            {
              sub_19F1B70(a1, a3, *(_QWORD *)(v44 + 24));
              if ( (v77 & 1) != 0 )
                return;
LABEL_53:
              j___libc_free_0(v78);
              return;
            }
            sub_19F1B70(a1, a3, v45);
LABEL_52:
            if ( (v77 & 1) != 0 )
              return;
            goto LABEL_53;
          }
LABEL_89:
          v43 = 24;
          goto LABEL_50;
        }
        break;
      case 1LL:
        v38 = v32;
        v60 = *(_BYTE *)(a2 + 23) & 0x40;
        goto LABEL_85;
      default:
        goto LABEL_89;
    }
    v38 = v67 + 1;
    if ( v60 )
      v70 = *(_QWORD *)(a2 - 8);
    else
      v70 = a2 - 24LL * (unsigned int)v31;
    v71 = *(_QWORD *)(v70 + 24LL * (unsigned int)(2 * (v67 + 1)));
    if ( v71 && v15 == v71 )
    {
      v38 = v67;
      goto LABEL_47;
    }
LABEL_85:
    if ( v60 )
      v61 = *(_QWORD *)(a2 - 8);
    else
      v61 = a2 - 24 * v31;
    v62 = *(_QWORD *)(v61 + 24LL * (unsigned int)(2 * v38 + 2));
    if ( !v62 || v15 != v62 )
      goto LABEL_89;
    goto LABEL_47;
  }
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) == 3 )
  {
    v50 = *(_QWORD *)(a2 - 72);
    v51 = sub_19E1ED0(a1, (__int64 ***)v50);
    v52 = *(_BYTE *)(v51 + 16);
    if ( v52 > 0x10u )
    {
      v63 = *(_BYTE *)(v50 + 16);
      if ( v63 <= 0x17u )
      {
        if ( v63 == 13 )
        {
          v53 = *(_QWORD *)(a2 - 24);
          v54 = *(_QWORD *)(a2 - 48);
          v51 = v50;
LABEL_70:
          v55 = *(_DWORD *)(v51 + 32);
          v56 = v51 + 24;
          if ( v55 <= 0x40 )
          {
            if ( *(_QWORD *)(v51 + 24) != 1 )
            {
              if ( *(_QWORD *)(v51 + 24) )
                return;
              goto LABEL_73;
            }
          }
          else
          {
            v73 = v53;
            v57 = sub_16A57B0(v51 + 24);
            v53 = v73;
            if ( v57 != v55 - 1 )
            {
              if ( v55 != (unsigned int)sub_16A57B0(v56) )
                return;
LABEL_73:
              v58 = v54;
LABEL_76:
              sub_19F1B70(a1, a3, v58);
              return;
            }
          }
          v58 = v53;
          goto LABEL_76;
        }
        v54 = *(_QWORD *)(a2 - 48);
        v53 = *(_QWORD *)(a2 - 24);
LABEL_75:
        sub_19F1B70(a1, a3, v53);
        v58 = v54;
        goto LABEL_76;
      }
      v64 = sub_19EDFA0(a1, v50);
      if ( *(_DWORD *)(v64 + 8) != 1 )
      {
        v53 = *(_QWORD *)(a2 - 24);
        v54 = *(_QWORD *)(a2 - 48);
        goto LABEL_75;
      }
      v51 = *(_QWORD *)(v64 + 24);
      v53 = *(_QWORD *)(a2 - 24);
      v54 = *(_QWORD *)(a2 - 48);
      if ( !v51 )
        goto LABEL_75;
      v52 = *(_BYTE *)(v51 + 16);
    }
    else
    {
      v53 = *(_QWORD *)(a2 - 24);
      v54 = *(_QWORD *)(a2 - 48);
    }
    if ( v52 == 13 )
      goto LABEL_70;
    goto LABEL_75;
  }
LABEL_3:
  v7 = 0;
  v8 = sub_15F4D60(a2);
  if ( v8 )
  {
    do
    {
      v9 = v7++;
      v10 = sub_15F4DF0(a2, v9);
      sub_19F1B70(a1, a3, v10);
    }
    while ( v8 != v7 );
  }
  v11 = sub_19E6CE0(a1, a2);
  v12 = v11;
  if ( v11 && *(_BYTE *)(v11 + 16) != 21 )
  {
    v48 = sub_19E2780(a1 + 1960, v11);
    if ( v12 != *(_QWORD *)(v48 + 40) )
    {
      v49 = sub_19E13F0(a1, 0, 0);
      *((_QWORD *)v49 + 5) = v12;
      v48 = (__int64)v49;
    }
    if ( (unsigned __int8)sub_19E7150(a1, v12, v48) )
      sub_19E5C50(a1, v12);
  }
}
