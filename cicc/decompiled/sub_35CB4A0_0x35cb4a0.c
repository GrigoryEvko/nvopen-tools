// Function: sub_35CB4A0
// Address: 0x35cb4a0
//
__int64 __fastcall sub_35CB4A0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  int v6; // eax
  __int64 v7; // rdi
  __int64 v8; // rax
  char *v9; // rbx
  __int64 v10; // r12
  char *v11; // r15
  char v12; // al
  char v13; // al
  unsigned int v14; // r8d
  int v15; // eax
  char v16; // al
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int16 *v19; // rax
  __int64 v20; // rdx
  int v21; // edi
  int v22; // eax
  __int64 v23; // rax
  __int64 v25; // rdx
  unsigned int *v26; // rax
  unsigned int *v27; // rdi
  int v28; // edx
  __int64 (*v29)(); // rax
  int v30; // ecx
  int v31; // edx
  char v32; // al
  __int64 v33; // rax
  int *v34; // rdx
  int *v35; // rcx
  int v36; // eax
  __int64 **v37; // r15
  __int64 v38; // rax
  __int64 v39; // rax
  unsigned __int8 *v40; // rax
  __int64 v41; // rdi
  unsigned __int8 v42; // al
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  int *v46; // rcx
  __int64 v47; // rax
  __int64 **v48; // rbx
  unsigned __int8 *v49; // rdi
  unsigned __int8 *v50; // rax
  __int64 v51; // rdi
  unsigned __int8 v52; // al
  __int64 **v53; // r12
  __int64 v54; // rdi
  unsigned __int8 *v55; // rdi
  unsigned __int8 *v56; // rax
  __int64 v57; // rdi
  unsigned __int8 v58; // al
  __int64 v59; // rdi
  unsigned __int8 *v60; // rdi
  unsigned __int8 *v61; // rax
  __int64 v62; // rdi
  unsigned __int8 v63; // al
  __int64 v64; // rdi
  unsigned __int8 *v65; // rdi
  unsigned __int8 *v66; // rax
  __int64 v67; // rdi
  unsigned __int8 v68; // al
  __int64 v69; // rdi
  unsigned __int64 v70; // rdi
  unsigned __int64 v71; // rdi
  unsigned __int64 v72; // rdi
  unsigned __int64 v73; // rdi
  __int64 v74; // rax
  unsigned __int8 *v75; // rax
  __int64 v76; // rdi
  unsigned __int8 v77; // al
  __int64 v78; // rax
  unsigned __int8 *v79; // rax
  __int64 v80; // rdi
  unsigned __int8 v81; // al
  unsigned __int64 v82; // rax
  unsigned __int64 v83; // rax
  unsigned __int64 v84; // rax
  unsigned int v85; // [rsp+0h] [rbp-40h]
  unsigned int v86; // [rsp+0h] [rbp-40h]
  int *v87; // [rsp+0h] [rbp-40h]

  if ( !a4 )
    goto LABEL_2;
  v30 = *(_DWORD *)(a2 + 44);
  v31 = v30 & 4;
  if ( (unsigned int)*(unsigned __int16 *)(a2 + 68) - 1 > 1 || (*(_BYTE *)(*(_QWORD *)(a2 + 32) + 64LL) & 8) == 0 )
  {
    if ( (v30 & 4) != 0 || (v30 & 8) == 0 )
    {
      v43 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 19) & 1LL;
    }
    else
    {
      LOBYTE(v43) = sub_2E88A90(a2, 0x80000, 1);
      v30 = *(_DWORD *)(a2 + 44);
      v31 = v30 & 4;
    }
    if ( !(_BYTE)v43
      && ((unsigned int)*(unsigned __int16 *)(a2 + 68) - 1 > 1 || (*(_BYTE *)(*(_QWORD *)(a2 + 32) + 64LL) & 0x10) == 0) )
    {
      if ( v31 || (v30 & 8) == 0 )
        v44 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 20) & 1LL;
      else
        LOBYTE(v44) = sub_2E88A90(a2, 0x100000, 1);
      if ( !(_BYTE)v44 )
        goto LABEL_2;
      v30 = *(_DWORD *)(a2 + 44);
      v31 = v30 & 4;
    }
  }
  if ( !v31 && (v30 & 8) != 0 )
    v32 = sub_2E88A90(a2, 128, 1);
  else
    v32 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 7;
  if ( v32 )
    return 1;
  if ( sub_2E8B090(a2) )
    return 1;
  v33 = *(_QWORD *)(a2 + 48);
  v34 = (int *)(v33 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v33 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 1;
  if ( (v33 & 7) != 0 )
  {
    if ( (v33 & 7) != 3 || !*v34 )
      return 1;
    v35 = (int *)(*(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL);
  }
  else
  {
    *(_QWORD *)(a2 + 48) = v34;
    LOBYTE(v33) = v33 & 0xF8;
    v35 = v34;
  }
  v36 = v33 & 7;
  if ( v36 )
  {
    if ( v36 != 3 )
      goto LABEL_2;
    v37 = (__int64 **)(v35 + 4);
    v45 = 2LL * *v35;
    v46 = &v35[v45 + 4];
    v47 = (v45 * 4) >> 5;
    v87 = v46;
    if ( v47 > 0 )
    {
      v48 = &v37[4 * v47];
      while ( 1 )
      {
        v64 = **v37;
        if ( !v64 )
          goto LABEL_59;
        if ( ((v64 >> 2) & 1) != 0 )
        {
          if ( ((v64 >> 2) & 1) == 0 )
            goto LABEL_59;
          v70 = v64 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v70 || *(_DWORD *)(v70 + 8) != 2 )
            goto LABEL_59;
        }
        else
        {
          v65 = (unsigned __int8 *)(v64 & 0xFFFFFFFFFFFFFFF8LL);
          if ( !v65 )
            goto LABEL_59;
          v66 = sub_98ACB0(v65, 6u);
          v67 = (__int64)v66;
          if ( !v66 )
            goto LABEL_59;
          v68 = *v66;
          if ( v68 == 22 )
          {
            if ( (unsigned __int8)sub_B2BAE0(v67) )
              goto LABEL_59;
          }
          else if ( v68 > 3u )
          {
            goto LABEL_59;
          }
        }
        v53 = v37 + 1;
        v69 = *v37[1];
        if ( !v69 )
        {
LABEL_106:
          v37 = v53;
          goto LABEL_59;
        }
        if ( ((v69 >> 2) & 1) != 0 )
        {
          if ( ((v69 >> 2) & 1) == 0 )
            goto LABEL_106;
          v71 = v69 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v71 || *(_DWORD *)(v71 + 8) != 2 )
            goto LABEL_106;
        }
        else
        {
          v49 = (unsigned __int8 *)(v69 & 0xFFFFFFFFFFFFFFF8LL);
          if ( !v49 )
            goto LABEL_106;
          v50 = sub_98ACB0(v49, 6u);
          v51 = (__int64)v50;
          if ( !v50 )
            goto LABEL_106;
          v52 = *v50;
          if ( v52 == 22 )
          {
            if ( (unsigned __int8)sub_B2BAE0(v51) )
              goto LABEL_106;
          }
          else if ( v52 > 3u )
          {
            goto LABEL_106;
          }
        }
        v53 = v37 + 2;
        v54 = *v37[2];
        if ( !v54 )
          goto LABEL_106;
        if ( ((v54 >> 2) & 1) != 0 )
        {
          if ( ((v54 >> 2) & 1) == 0 )
            goto LABEL_106;
          v72 = v54 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v72 || *(_DWORD *)(v72 + 8) != 2 )
            goto LABEL_106;
        }
        else
        {
          v55 = (unsigned __int8 *)(v54 & 0xFFFFFFFFFFFFFFF8LL);
          if ( !v55 )
            goto LABEL_106;
          v56 = sub_98ACB0(v55, 6u);
          v57 = (__int64)v56;
          if ( !v56 )
            goto LABEL_106;
          v58 = *v56;
          if ( v58 == 22 )
          {
            if ( (unsigned __int8)sub_B2BAE0(v57) )
              goto LABEL_106;
          }
          else if ( v58 > 3u )
          {
            goto LABEL_106;
          }
        }
        v53 = v37 + 3;
        v59 = *v37[3];
        if ( !v59 )
          goto LABEL_106;
        if ( ((v59 >> 2) & 1) != 0 )
        {
          if ( ((v59 >> 2) & 1) == 0 )
            goto LABEL_106;
          v73 = v59 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v73 || *(_DWORD *)(v73 + 8) != 2 )
            goto LABEL_106;
        }
        else
        {
          v60 = (unsigned __int8 *)(v59 & 0xFFFFFFFFFFFFFFF8LL);
          if ( !v60 )
            goto LABEL_106;
          v61 = sub_98ACB0(v60, 6u);
          v62 = (__int64)v61;
          if ( !v61 )
            goto LABEL_106;
          v63 = *v61;
          if ( v63 == 22 )
          {
            if ( (unsigned __int8)sub_B2BAE0(v62) )
              goto LABEL_106;
          }
          else if ( v63 > 3u )
          {
            goto LABEL_106;
          }
        }
        v37 += 4;
        if ( v37 == v48 )
          goto LABEL_146;
      }
    }
    v48 = v37;
LABEL_146:
    v37 = v48;
    v38 = ((char *)v87 - (char *)v48) >> 3;
    if ( (char *)v87 - (char *)v48 == 16 )
      goto LABEL_138;
  }
  else
  {
    v37 = (__int64 **)(a2 + 48);
    *(_QWORD *)(a2 + 48) = v35;
    v87 = (int *)(a2 + 56);
    v38 = 1;
  }
  if ( v38 != 3 )
  {
    if ( v38 != 1 )
      goto LABEL_2;
LABEL_53:
    v39 = **v37;
    if ( v39 )
    {
      if ( ((v39 >> 2) & 1) != 0 )
      {
        if ( ((v39 >> 2) & 1) != 0 )
        {
          v82 = v39 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v82 )
          {
            if ( *(_DWORD *)(v82 + 8) == 2 )
              goto LABEL_2;
          }
        }
      }
      else if ( (v39 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v40 = sub_98ACB0((unsigned __int8 *)(v39 & 0xFFFFFFFFFFFFFFF8LL), 6u);
        v41 = (__int64)v40;
        if ( v40 )
        {
          v42 = *v40;
          if ( v42 == 22 )
          {
            if ( !(unsigned __int8)sub_B2BAE0(v41) )
              goto LABEL_2;
          }
          else if ( v42 <= 3u )
          {
            goto LABEL_2;
          }
        }
      }
    }
    goto LABEL_59;
  }
  v74 = **v37;
  if ( !v74 )
    goto LABEL_59;
  if ( ((v74 >> 2) & 1) != 0 )
  {
    if ( ((v74 >> 2) & 1) == 0 )
      goto LABEL_59;
    v84 = v74 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v84 || *(_DWORD *)(v84 + 8) != 2 )
      goto LABEL_59;
  }
  else
  {
    if ( (v74 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_59;
    v75 = sub_98ACB0((unsigned __int8 *)(v74 & 0xFFFFFFFFFFFFFFF8LL), 6u);
    v76 = (__int64)v75;
    if ( !v75 )
      goto LABEL_59;
    v77 = *v75;
    if ( v77 == 22 )
    {
      if ( (unsigned __int8)sub_B2BAE0(v76) )
        goto LABEL_59;
    }
    else if ( v77 > 3u )
    {
      goto LABEL_59;
    }
  }
  ++v37;
LABEL_138:
  v78 = **v37;
  if ( v78 )
  {
    if ( ((v78 >> 2) & 1) != 0 )
    {
      if ( ((v78 >> 2) & 1) != 0 )
      {
        v83 = v78 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v83 )
        {
          if ( *(_DWORD *)(v83 + 8) == 2 )
            goto LABEL_144;
        }
      }
    }
    else if ( (v78 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v79 = sub_98ACB0((unsigned __int8 *)(v78 & 0xFFFFFFFFFFFFFFF8LL), 6u);
      v80 = (__int64)v79;
      if ( v79 )
      {
        v81 = *v79;
        if ( v81 == 22 )
        {
          if ( !(unsigned __int8)sub_B2BAE0(v80) )
          {
LABEL_144:
            ++v37;
            goto LABEL_53;
          }
        }
        else if ( v81 <= 3u )
        {
          goto LABEL_144;
        }
      }
    }
  }
LABEL_59:
  if ( v37 != (__int64 **)v87 )
    return 1;
LABEL_2:
  v6 = *(unsigned __int16 *)(a2 + 68);
  if ( *(_DWORD *)(a1 + 584) == v6 || *(_DWORD *)(a1 + 588) == v6 )
    return 1;
  v7 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 24) + 32LL) + 16LL);
  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v7 + 200LL))(v7);
  v9 = *(char **)(a2 + 32);
  v10 = v8;
  v11 = &v9[40 * (*(_DWORD *)(a2 + 40) & 0xFFFFFF)];
  if ( v11 == v9 )
    return 0;
  while ( 1 )
  {
    v12 = *v9;
    if ( !*v9 )
    {
      if ( (v9[3] & 0x10) == 0 )
      {
        v13 = v9[4];
        if ( (v13 & 1) != 0 || (v13 & 2) != 0 )
          goto LABEL_23;
      }
      v14 = *((_DWORD *)v9 + 2);
      if ( !v14 )
        goto LABEL_23;
      v15 = *(_DWORD *)(a2 + 44);
      if ( (v15 & 4) != 0 || (v15 & 8) == 0 )
      {
        v16 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 7;
      }
      else
      {
        v85 = *((_DWORD *)v9 + 2);
        v16 = sub_2E88A90(a2, 128, 1);
        v14 = v85;
      }
      if ( !v16 && v14 == *(_DWORD *)(a1 + 592) )
        return 1;
      v17 = *(_QWORD *)(a1 + 224);
      v18 = *(_QWORD *)(v17 + 8);
      v19 = (__int16 *)(*(_QWORD *)(v17 + 56) + 2LL * (*(_DWORD *)(v18 + 24LL * v14 + 16) >> 12));
      v20 = *(_DWORD *)(v18 + 24LL * v14 + 16) & 0xFFF;
      if ( v19 )
      {
        while ( !*(_WORD *)(*(_QWORD *)(a1 + 288) + 2LL * (unsigned int)v20) )
        {
          v21 = *v19++;
          v20 = (unsigned int)(v21 + v20);
          if ( !(_WORD)v21 )
            goto LABEL_17;
        }
        return 1;
      }
LABEL_17:
      v22 = *(_DWORD *)(a2 + 44);
      if ( (v22 & 4) == 0 && (v22 & 8) != 0 )
      {
        v86 = v14;
        LOBYTE(v23) = sub_2E88A90(a2, 32, 1);
        v14 = v86;
      }
      else
      {
        v23 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 24LL) >> 5) & 1LL;
      }
      if ( !(_BYTE)v23 )
      {
        v29 = *(__int64 (**)())(*(_QWORD *)v10 + 696LL);
        if ( v29 != sub_2FF5310 )
        {
          if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64))v29)(v10, v14, v20) )
            return 1;
        }
      }
      goto LABEL_21;
    }
    if ( v12 != 12 )
      goto LABEL_22;
    v25 = sub_35CAC40(a1, a3);
    v26 = *(unsigned int **)(v25 + 32);
    v27 = &v26[*(unsigned int *)(v25 + 40)];
    if ( v27 != v26 )
      break;
LABEL_21:
    v12 = *v9;
LABEL_22:
    if ( v12 == 5 && (unsigned __int16)(*(_WORD *)(a2 + 68) - 14) > 1u )
      return 1;
LABEL_23:
    v9 += 40;
    if ( v11 == v9 )
      return 0;
  }
  while ( 1 )
  {
    v28 = *(_DWORD *)(*((_QWORD *)v9 + 3) + 4LL * (*v26 >> 5));
    if ( !_bittest(&v28, *v26) )
      return 1;
    if ( v27 == ++v26 )
      goto LABEL_21;
  }
}
