// Function: sub_2AFFBC0
// Address: 0x2affbc0
//
unsigned __int64 __fastcall sub_2AFFBC0(__int64 a1, __int64 a2, int a3)
{
  __int64 v6; // rcx
  __int64 v7; // rsi
  int v8; // edi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  __int16 v13; // ax
  __int64 v14; // rsi
  unsigned int v15; // edx
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rcx
  unsigned __int64 v19; // rcx
  __int64 v20; // rax
  unsigned int v21; // eax
  unsigned __int64 v22; // r12
  const void **v23; // rsi
  int v24; // eax
  int v25; // r10d
  __int64 *v26; // rdx
  unsigned int v27; // r9d
  __int64 *v28; // rax
  __int64 v29; // r8
  _QWORD *v30; // rax
  unsigned __int64 v31; // rax
  _QWORD *v32; // r14
  unsigned int v33; // r13d
  unsigned __int64 v34; // r15
  _QWORD *v35; // rbx
  unsigned __int64 v36; // rsi
  unsigned __int64 i; // rcx
  unsigned __int64 v38; // rax
  int v39; // eax
  int v40; // r10d
  __int64 v41; // r9
  unsigned int v42; // esi
  int v43; // ecx
  __int64 v44; // r8
  int v45; // r9d
  __int64 *v46; // r8
  unsigned __int64 v47; // r14
  __int64 *v48; // rbx
  unsigned int v49; // r13d
  __int64 v50; // rdi
  __int64 v51; // r15
  unsigned int v52; // edx
  __int64 v53; // rax
  unsigned __int64 v54; // rcx
  unsigned __int64 v55; // r15
  int v56; // eax
  int v57; // r8d
  int v58; // r8d
  __int64 v59; // r9
  __int64 *v60; // rsi
  unsigned int v61; // r13d
  int v62; // eax
  __int64 v63; // rdi
  __int64 v64; // rax
  unsigned __int8 *v65; // rdi
  unsigned __int8 v66; // al
  char v67; // cl
  __int16 v68; // ax
  __int64 v69; // rsi
  __int64 v70; // rax
  __int64 *v71; // rax
  __int64 v72; // rbx
  __int64 v73; // rax
  unsigned __int64 v74; // rax
  unsigned __int64 v75; // rdx
  unsigned __int16 v76; // ax
  int v77; // eax
  __int64 *v78; // rdi
  unsigned __int64 v79; // rcx
  int v80; // r10d
  __int64 v81; // rax
  unsigned int v82; // [rsp-84h] [rbp-84h]
  const void **v83; // [rsp-80h] [rbp-80h]
  __int64 v84; // [rsp-78h] [rbp-78h]
  __int64 *v85; // [rsp-70h] [rbp-70h]
  __int64 v86; // [rsp-70h] [rbp-70h]
  __int64 v87[2]; // [rsp-68h] [rbp-68h] BYREF
  unsigned __int64 *v88; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v89; // [rsp-50h] [rbp-50h]
  __int64 v90; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v91; // [rsp-40h] [rbp-40h]

  if ( !a1 )
    return 0;
  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)v7 )
  {
    v8 = v7 - 1;
    v9 = (v7 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( *v10 == a1 )
    {
LABEL_4:
      if ( v10 != (__int64 *)(v6 + 16LL * (unsigned int)v7) )
        return v10[1];
    }
    else
    {
      v24 = 1;
      while ( v11 != -4096 )
      {
        v45 = v24 + 1;
        v9 = v8 & (v24 + v9);
        v10 = (__int64 *)(v6 + 16LL * v9);
        v11 = *v10;
        if ( *v10 == a1 )
          goto LABEL_4;
        v24 = v45;
      }
    }
    if ( a3 == 10 )
    {
      v25 = 1;
      v26 = 0;
      v27 = v8 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v28 = (__int64 *)(v6 + 16LL * v27);
      v29 = *v28;
      if ( *v28 == a1 )
      {
LABEL_25:
        v30 = v28 + 1;
LABEL_26:
        *v30 = 0;
        return 0;
      }
      while ( v29 != -4096 )
      {
        if ( v29 == -8192 && !v26 )
          v26 = v28;
        v27 = v8 & (v25 + v27);
        v28 = (__int64 *)(v6 + 16LL * v27);
        v29 = *v28;
        if ( *v28 == a1 )
          goto LABEL_25;
        ++v25;
      }
      if ( !v26 )
        v26 = v28;
      v56 = *(_DWORD *)(a2 + 16);
      ++*(_QWORD *)a2;
      v43 = v56 + 1;
      if ( 4 * (v56 + 1) < (unsigned int)(3 * v7) )
      {
        if ( (int)v7 - *(_DWORD *)(a2 + 20) - v43 > (unsigned int)v7 >> 3 )
        {
LABEL_46:
          *(_DWORD *)(a2 + 16) = v43;
          if ( *v26 != -4096 )
            --*(_DWORD *)(a2 + 20);
          *v26 = a1;
          v30 = v26 + 1;
          v26[1] = 0;
          goto LABEL_26;
        }
        sub_2AFF6D0(a2, v7);
        v57 = *(_DWORD *)(a2 + 24);
        if ( v57 )
        {
          v58 = v57 - 1;
          v59 = *(_QWORD *)(a2 + 8);
          v60 = 0;
          v61 = v58 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
          v43 = *(_DWORD *)(a2 + 16) + 1;
          v62 = 1;
          v26 = (__int64 *)(v59 + 16LL * v61);
          v63 = *v26;
          if ( *v26 != a1 )
          {
            while ( v63 != -4096 )
            {
              if ( !v60 && v63 == -8192 )
                v60 = v26;
              v80 = v62 + 1;
              v81 = v58 & (v61 + v62);
              v61 = v81;
              v26 = (__int64 *)(v59 + 16 * v81);
              v63 = *v26;
              if ( *v26 == a1 )
                goto LABEL_46;
              v62 = v80;
            }
            if ( v60 )
              v26 = v60;
          }
          goto LABEL_46;
        }
LABEL_148:
        ++*(_DWORD *)(a2 + 16);
        BUG();
      }
LABEL_44:
      sub_2AFF6D0(a2, 2 * v7);
      v39 = *(_DWORD *)(a2 + 24);
      if ( v39 )
      {
        v40 = v39 - 1;
        v41 = *(_QWORD *)(a2 + 8);
        v42 = (v39 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v43 = *(_DWORD *)(a2 + 16) + 1;
        v26 = (__int64 *)(v41 + 16LL * v42);
        v44 = *v26;
        if ( *v26 != a1 )
        {
          v77 = 1;
          v78 = 0;
          while ( v44 != -4096 )
          {
            if ( !v78 && v44 == -8192 )
              v78 = v26;
            v42 = v40 & (v77 + v42);
            v26 = (__int64 *)(v41 + 16LL * v42);
            v44 = *v26;
            if ( *v26 == a1 )
              goto LABEL_46;
            ++v77;
          }
          if ( v78 )
            v26 = v78;
        }
        goto LABEL_46;
      }
      goto LABEL_148;
    }
  }
  else if ( a3 == 10 )
  {
    ++*(_QWORD *)a2;
    goto LABEL_44;
  }
  v13 = *(_WORD *)(a1 + 24);
  v87[0] = a2;
  v87[1] = a1;
  if ( !v13 )
  {
    v14 = *(_QWORD *)(a1 + 32);
    v15 = *(_DWORD *)(v14 + 32);
    v16 = *(_QWORD *)(v14 + 24);
    v17 = 1LL << ((unsigned __int8)v15 - 1);
    if ( v15 > 0x40 )
    {
      v23 = (const void **)(v14 + 24);
      if ( (*(_QWORD *)(v16 + 8LL * ((v15 - 1) >> 6)) & v17) == 0 )
      {
        v89 = v15;
        sub_C43780((__int64)&v88, v23);
        v21 = v89;
LABEL_16:
        if ( v21 > 0x40 )
        {
          v22 = *v88;
          j_j___libc_free_0_0((unsigned __int64)v88);
          return sub_2AFF8B0(v87, v22);
        }
LABEL_28:
        v22 = (unsigned __int64)v88;
        return sub_2AFF8B0(v87, v22);
      }
      v91 = v15;
      sub_C43780((__int64)&v90, v23);
      v15 = v91;
      if ( v91 > 0x40 )
      {
        sub_C43D10((__int64)&v90);
LABEL_15:
        sub_C46250((__int64)&v90);
        v21 = v91;
        v89 = v91;
        v88 = (unsigned __int64 *)v90;
        goto LABEL_16;
      }
      v18 = v90;
    }
    else
    {
      v18 = *(_QWORD *)(v14 + 24);
      if ( (v17 & v16) == 0 )
      {
        v88 = *(unsigned __int64 **)(v14 + 24);
        goto LABEL_28;
      }
      v91 = *(_DWORD *)(v14 + 32);
    }
    v19 = ~v18 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v15);
    v20 = 0;
    if ( v15 )
      v20 = v19;
    v90 = v20;
    goto LABEL_15;
  }
  if ( v13 == 14 || (unsigned __int16)(v13 - 2) <= 2u )
  {
    v31 = sub_2AFFBC0(*(_QWORD *)(a1 + 32), a2, (unsigned int)(a3 + 1));
    return sub_2AFF8B0(v87, v31);
  }
  if ( v13 == 5 )
  {
    v32 = *(_QWORD **)(a1 + 32);
    v33 = a3 + 1;
    v34 = 0;
    v35 = &v32[*(_QWORD *)(a1 + 40)];
    if ( v32 != v35 )
    {
      while ( 1 )
      {
        v36 = sub_2AFFBC0(*v32, a2, v33);
        if ( !v36 )
          break;
        if ( v34 )
        {
          for ( i = v34 % v36; i; i = v38 % i )
          {
            v38 = v36;
            v36 = i;
          }
        }
        ++v32;
        v34 = v36;
        if ( v35 == v32 )
          return sub_2AFF8B0(v87, v34);
      }
    }
    v34 = 0;
    return sub_2AFF8B0(v87, v34);
  }
  if ( v13 != 6 )
  {
    if ( v13 == 15 )
    {
      v65 = sub_BD3990(*(unsigned __int8 **)(a1 - 8), v7);
      v66 = *v65;
      if ( *v65 == 3 )
      {
        v67 = 0;
        v68 = (*((_WORD *)v65 + 17) >> 1) & 0x3F;
        if ( v68 )
          v67 = v68 - 1;
        v69 = 1LL << v67;
      }
      else if ( v66 == 60 )
      {
        _BitScanReverse64(&v79, 1LL << *((_WORD *)v65 + 1));
        v69 = 0x8000000000000000LL >> ((unsigned __int8)v79 ^ 0x3Fu);
      }
      else
      {
        v69 = 0;
        if ( v66 == 22 && *(_BYTE *)(*((_QWORD *)v65 + 1) + 8LL) == 14 )
        {
          v76 = sub_B2BD00((__int64)v65);
          if ( !HIBYTE(v76) )
            LOBYTE(v76) = 0;
          v69 = 1LL << v76;
        }
      }
      return sub_2AFF8B0(v87, v69);
    }
    if ( v13 == 8 && *(_QWORD *)(a1 + 40) == 2 )
    {
      v70 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL);
      if ( !*(_WORD *)(v70 + 24) )
      {
        sub_9692E0((__int64)&v90, (__int64 *)(*(_QWORD *)(v70 + 32) + 24LL));
        v47 = v91 <= 0x40 ? v90 : *(_QWORD *)v90;
        sub_969240(&v90);
        if ( v47 )
        {
          v71 = *(__int64 **)(a1 + 32);
          v72 = *v71;
          if ( !*(_WORD *)(*v71 + 24) )
          {
            sub_9692E0((__int64)&v90, (__int64 *)(*(_QWORD *)(v72 + 32) + 24LL));
            v73 = v90;
            if ( v91 > 0x40 )
              v73 = *(_QWORD *)v90;
            v86 = v73;
            sub_969240(&v90);
            if ( !v86 )
            {
              v74 = 0x100000000LL;
              goto LABEL_115;
            }
          }
          v74 = sub_2AFFBC0(v72, a2, (unsigned int)(a3 + 1));
          if ( v74 )
          {
LABEL_115:
            while ( 1 )
            {
              v75 = v74 % v47;
              v74 = v47;
              if ( !v75 )
                break;
              v47 = v75;
            }
            return sub_2AFF8B0(v87, v47);
          }
        }
      }
    }
    return sub_2AFF8B0(v87, 0);
  }
  v46 = *(__int64 **)(a1 + 32);
  v47 = 0;
  v85 = &v46[*(_QWORD *)(a1 + 40)];
  if ( v46 == v85 )
    return sub_2AFF8B0(v87, v47);
  v84 = 0;
  v48 = *(__int64 **)(a1 + 32);
  v49 = a3 + 1;
  do
  {
    v50 = *v48;
    if ( *(_WORD *)(*v48 + 24) )
    {
      v64 = sub_2AFFBC0(v50, a2, v49);
      if ( v64 )
      {
        if ( v84 )
          v64 *= v84;
        v84 = v64;
      }
    }
    else
    {
      v51 = *(_QWORD *)(v50 + 32);
      v52 = *(_DWORD *)(v51 + 32);
      if ( v52 > 0x40 )
      {
        v82 = *(_DWORD *)(v51 + 32);
        v83 = (const void **)(v51 + 24);
        if ( (unsigned int)sub_C44630(v51 + 24) != 1 )
          goto LABEL_68;
        if ( (*(_QWORD *)(*(_QWORD *)(v51 + 24) + 8LL * ((v82 - 1) >> 6)) & (1LL << ((unsigned __int8)v82 - 1))) != 0 )
        {
          v91 = v82;
          sub_C43780((__int64)&v90, v83);
          v52 = v91;
          if ( v91 <= 0x40 )
          {
LABEL_60:
            v54 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v52;
            if ( !v52 )
              v54 = 0;
            v90 = v54 & ~v90;
          }
          else
          {
            sub_C43D10((__int64)&v90);
          }
          sub_C46250((__int64)&v90);
          v89 = v91;
          v88 = (unsigned __int64 *)v90;
        }
        else
        {
          v89 = v82;
          sub_C43780((__int64)&v88, v83);
        }
        if ( v89 <= 0x40 )
        {
LABEL_120:
          v55 = (unsigned __int64)v88;
        }
        else
        {
          v55 = *v88;
          j_j___libc_free_0_0((unsigned __int64)v88);
        }
        if ( v47 )
          v47 *= v55;
        else
          v47 = v55;
        goto LABEL_68;
      }
      v53 = *(_QWORD *)(v51 + 24);
      if ( v53 && (v53 & (v53 - 1)) == 0 )
      {
        if ( _bittest64(&v53, v52 - 1) )
        {
          v91 = *(_DWORD *)(v51 + 32);
          v90 = v53;
          goto LABEL_60;
        }
        v88 = *(unsigned __int64 **)(v51 + 24);
        goto LABEL_120;
      }
    }
LABEL_68:
    ++v48;
  }
  while ( v85 != v48 );
  if ( v84 )
    v47 *= v84;
  return sub_2AFF8B0(v87, v47);
}
