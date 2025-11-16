// Function: callback
// Address: 0x39f97a0
//
__int64 __fastcall callback(struct dl_phdr_info *a1, unsigned __int64 a2, _DWORD *a3)
{
  const Elf64_Phdr *dlpi_phdr; // rax
  Elf64_Addr dlpi_addr; // r8
  Elf64_Addr *v8; // r12
  __int64 dlpi_phnum; // rdx
  Elf64_Addr v10; // r13
  Elf64_Addr v11; // r14
  __int64 v12; // r10
  const Elf64_Phdr *v13; // r11
  const Elf64_Phdr *v14; // rdx
  const Elf64_Phdr *v15; // rcx
  Elf64_Word p_type; // esi
  Elf64_Addr v17; // rsi
  Elf64_Addr *v18; // rax
  Elf64_Addr v19; // rdx
  _BYTE *v20; // r8
  _BYTE *v21; // rbp
  unsigned __int64 dlpi_subs; // rsi
  __int64 *v24; // rdx
  __int64 v25; // r10
  unsigned __int64 v26; // r9
  Elf64_Addr *v27; // r11
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rsi
  unsigned __int64 *v30; // rdx
  char v31; // di
  char *v32; // r8
  unsigned __int8 v33; // dl
  char *v34; // rsi
  char *v35; // rax
  char v36; // dl
  unsigned __int64 v37; // rax
  __int64 v38; // rdx
  unsigned __int64 v39; // rax
  unsigned int *v40; // rax
  char v41; // al
  unsigned __int8 v42; // cl
  char *v43; // rsi
  unsigned __int8 v44; // cl
  char *v45; // rsi
  char *v46; // rax
  char *v47; // rdx
  unsigned __int64 v48; // r8
  unsigned __int64 v49; // rdi
  unsigned __int64 v50; // rcx
  char *v51; // r12
  unsigned __int64 v52; // rax
  _BYTE *v53; // r13
  char v54; // al
  unsigned __int8 v55; // cl
  __int64 v56; // rdx
  _BYTE *v57; // r8
  Elf64_Addr *v58; // rdi
  Elf64_Addr v59; // rax
  Elf64_Addr *v60; // [rsp+8h] [rbp-80h]
  unsigned int *v61; // [rsp+10h] [rbp-78h] BYREF
  unsigned __int64 v62; // [rsp+18h] [rbp-70h] BYREF
  unsigned __int64 v63[13]; // [rsp+20h] [rbp-68h] BYREF

  dlpi_phdr = a1->dlpi_phdr;
  dlpi_addr = a1->dlpi_addr;
  if ( a3[10] && a2 > 0x2F )
  {
    dlpi_subs = a1->dlpi_subs;
    if ( a1->dlpi_adds != qword_4CF6DE0 || qword_5057728 != dlpi_subs )
    {
      qword_4CF6DE0 = a1->dlpi_adds;
      v24 = (__int64 *)&unk_5057770;
      qword_5057728 = dlpi_subs;
      do
      {
        *(v24 - 6) = 0;
        *(v24 - 5) = 0;
        *(v24 - 1) = (__int64)v24;
        v24 += 6;
      }
      while ( &qword_50578F0 != v24 );
      v8 = 0;
      qword_50578B8 = 0;
      qword_5057730 = (__int64)&unk_5057740;
      a3[10] = 0;
      v60 = 0;
      goto LABEL_5;
    }
    v25 = qword_5057730;
    if ( !qword_5057730 )
    {
LABEL_4:
      v60 = 0;
      v8 = 0;
LABEL_5:
      dlpi_phnum = a1->dlpi_phnum;
      if ( !(_WORD)dlpi_phnum )
        return 0;
      v10 = 0;
      v11 = 0;
      v12 = 0;
      v13 = 0;
      v14 = &dlpi_phdr[dlpi_phnum];
      v15 = 0;
      while ( 1 )
      {
        p_type = dlpi_phdr->__p_type;
        if ( dlpi_phdr->__p_type != 1 )
          break;
        v17 = dlpi_addr + dlpi_phdr->p_vaddr;
        if ( *(_QWORD *)a3 < v17 )
        {
LABEL_10:
          if ( v14 == ++dlpi_phdr )
            goto LABEL_16;
        }
        else
        {
          if ( *(_QWORD *)a3 < v17 + dlpi_phdr->p_memsz )
          {
            v10 = v17 + dlpi_phdr->p_memsz;
            v11 = dlpi_addr + dlpi_phdr->p_vaddr;
            v12 = 1;
          }
          if ( v14 == ++dlpi_phdr )
          {
LABEL_16:
            if ( v12 )
            {
              if ( a2 > 0x2F )
              {
                v18 = (Elf64_Addr *)qword_5057730;
                if ( v8 && v60 )
                {
                  v19 = v8[5];
                  qword_5057730 = (__int64)v8;
                  v60[5] = v19;
                  v8[5] = (Elf64_Addr)v18;
                  v18 = v8;
                }
                v18[2] = dlpi_addr;
                v18[3] = (Elf64_Addr)v13;
                v18[4] = (Elf64_Addr)v15;
                *v18 = v11;
                v18[1] = v10;
              }
              goto LABEL_22;
            }
            return 0;
          }
        }
      }
      if ( p_type == 1685382480 )
      {
        v13 = dlpi_phdr;
      }
      else if ( p_type == 2 )
      {
        v15 = dlpi_phdr;
      }
      goto LABEL_10;
    }
    v8 = (Elf64_Addr *)qword_5057730;
    v26 = *(_QWORD *)a3;
    v27 = 0;
    v28 = *(_QWORD *)qword_5057730;
    v29 = *(_QWORD *)(qword_5057730 + 8);
    if ( *(_QWORD *)a3 < *(_QWORD *)qword_5057730 )
    {
LABEL_36:
      while ( v29 | v28 )
      {
        v30 = (unsigned __int64 *)v8[5];
        if ( !v30 )
          break;
        v27 = v8;
        v8 = (Elf64_Addr *)v8[5];
        v28 = *v30;
        v29 = v8[1];
        if ( v26 >= v28 )
          goto LABEL_35;
      }
      v60 = v27;
      goto LABEL_5;
    }
LABEL_35:
    if ( v26 >= v29 )
      goto LABEL_36;
    v58 = v27;
    dlpi_addr = v8[2];
    v13 = (const Elf64_Phdr *)v8[3];
    if ( (Elf64_Addr *)qword_5057730 != v8 )
    {
      v59 = v8[5];
      qword_5057730 = (__int64)v8;
      v58[5] = v59;
      v8[5] = v25;
    }
LABEL_22:
    if ( !v13 )
      return 0;
    v20 = (_BYTE *)(v13->p_vaddr + dlpi_addr);
    v21 = v20;
    if ( *v20 != 1 )
      return 1;
    v31 = v20[1];
    v32 = v20 + 4;
    if ( v31 != -1 )
    {
      v33 = v31 & 0x70;
      if ( (v31 & 0x70) == 0x20 )
      {
        v34 = (char *)*((_QWORD *)a3 + 1);
        goto LABEL_45;
      }
      if ( v33 <= 0x20u )
      {
        if ( (v21[1] & 0x60) != 0 )
          goto LABEL_98;
      }
      else
      {
        if ( v33 == 48 )
        {
          v34 = (char *)*((_QWORD *)a3 + 2);
LABEL_45:
          v35 = sub_39F8BA0(v31, v34, v32, (unsigned __int64 *)&v61);
          v36 = v21[2];
          if ( v36 == -1 || v21[3] != 59 )
            goto LABEL_47;
          v44 = v36 & 0x70;
          if ( (v36 & 0x70) == 0x20 )
          {
            v45 = (char *)*((_QWORD *)a3 + 1);
          }
          else
          {
            if ( v44 <= 0x20u )
            {
              if ( (v21[2] & 0x60) != 0 )
                goto LABEL_98;
            }
            else
            {
              if ( v44 == 48 )
              {
                v45 = (char *)*((_QWORD *)a3 + 2);
                goto LABEL_68;
              }
              if ( v44 != 80 )
                goto LABEL_98;
            }
            v45 = 0;
          }
LABEL_68:
          v46 = sub_39F8BA0(v36, v45, v35, &v62);
          v47 = v46;
          if ( !v62 )
            return 1;
          v48 = (unsigned __int8)v46 & 3;
          if ( ((unsigned __int8)v46 & 3) != 0 )
          {
LABEL_47:
            v37 = *((_QWORD *)a3 + 1);
            v63[0] = 0;
            v63[4] = 4;
            v38 = *(_QWORD *)a3;
            v63[1] = v37;
            v39 = *((_QWORD *)a3 + 2);
            v63[3] = (unsigned __int64)v61;
            v63[2] = v39;
            v40 = sub_39F9490((__int64)v63, v61, v38);
            *((_QWORD *)a3 + 4) = v40;
            if ( !v40 )
              return 1;
            v41 = sub_39F8CF0((__int64)v40 - (int)v40[1] + 4);
            if ( v41 != -1 )
            {
              v42 = v41 & 0x70;
              if ( (v41 & 0x70) == 0x20 )
              {
                v43 = (char *)*((_QWORD *)a3 + 1);
                goto LABEL_55;
              }
              if ( v42 <= 0x20u )
              {
                if ( (v41 & 0x60) != 0 )
                  goto LABEL_98;
              }
              else
              {
                if ( v42 == 48 )
                {
                  v43 = (char *)*((_QWORD *)a3 + 2);
LABEL_55:
                  sub_39F8BA0(v41, v43, (char *)(*((_QWORD *)a3 + 4) + 8LL), &v62);
                  *((_QWORD *)a3 + 3) = v62;
                  return 1;
                }
                if ( v42 != 80 )
LABEL_98:
                  abort();
              }
            }
            v43 = 0;
            goto LABEL_55;
          }
          v49 = *(_QWORD *)a3;
          if ( (unsigned __int64)&v21[*(int *)v46] > *(_QWORD *)a3 )
            return 1;
          v50 = v62 - 1;
          v51 = &v46[8 * v62 - 8];
          if ( (unsigned __int64)&v21[*(int *)v51] > v49 )
          {
            do
            {
              while ( 1 )
              {
                if ( v48 >= v50 )
                  abort();
                v52 = (v48 + v50) >> 1;
                v51 = &v47[8 * v52];
                if ( (unsigned __int64)&v21[*(int *)v51] <= v49 )
                  break;
                v50 = (v48 + v50) >> 1;
              }
              v48 = v52 + 1;
            }
            while ( (unsigned __int64)&v21[*(int *)&v47[8 * v52 + 8]] <= v49 );
          }
          v53 = &v21[*((int *)v51 + 1)];
          v54 = sub_39F8CF0((__int64)&v53[-*((int *)v53 + 1) + 4]);
          if ( v54 == -1 )
          {
            v56 = 8;
            goto LABEL_79;
          }
          v55 = v54 & 7;
          if ( (v54 & 7) == 2 )
          {
            v56 = 10;
            goto LABEL_79;
          }
          if ( v55 <= 2u )
          {
            if ( v55 )
              goto LABEL_98;
          }
          else
          {
            v56 = 12;
            if ( v55 == 3 )
            {
LABEL_79:
              sub_39F8BA0(v54 & 0xF, 0, &v53[v56], v63);
              v57 = &v21[*(int *)v51];
              if ( *(_QWORD *)a3 < (unsigned __int64)&v57[v63[0]] )
                *((_QWORD *)a3 + 4) = v53;
              *((_QWORD *)a3 + 3) = v57;
              return 1;
            }
            if ( v55 != 4 )
              goto LABEL_98;
          }
          v56 = 16;
          goto LABEL_79;
        }
        if ( v33 != 80 )
          goto LABEL_98;
      }
    }
    v34 = 0;
    goto LABEL_45;
  }
  if ( a2 > 0x19 )
    goto LABEL_4;
  return 0xFFFFFFFFLL;
}
