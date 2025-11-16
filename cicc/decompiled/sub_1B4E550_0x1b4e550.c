// Function: sub_1B4E550
// Address: 0x1b4e550
//
__int64 __fastcall sub_1B4E550(__int64 a1, __int64 *a2, double a3, double a4, double a5)
{
  __int64 *v5; // r15
  __int64 v7; // rax
  bool v8; // zf
  int v9; // r9d
  int v10; // eax
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r12
  __int64 v15; // rcx
  __int64 v16; // r15
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r13
  unsigned int v20; // ebx
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 *v24; // rax
  __int64 v25; // rsi
  __int64 v26; // r12
  __int64 v27; // rbx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rsi
  _QWORD *v33; // r15
  __int64 i; // r15
  int v35; // ebx
  __int64 v36; // rax
  int v37; // r12d
  unsigned int v38; // ecx
  unsigned int v39; // esi
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 j; // r13
  unsigned int v43; // ebx
  __int64 v44; // rax
  int v45; // r15d
  unsigned int v46; // ecx
  unsigned int v47; // esi
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rdx
  unsigned int v51; // ecx
  __int64 *v52; // rax
  _QWORD *v53; // rax
  __int64 v54; // rdx
  unsigned __int64 v55; // rsi
  __int64 v56; // rcx
  char *v57; // rax
  char v58; // di
  __int64 v59; // r13
  unsigned int v60; // ecx
  char *v61; // r11
  unsigned __int64 v62; // rdx
  __int64 v63; // r9
  __int64 v64; // r8
  __int64 v65; // [rsp-10h] [rbp-1F0h]
  __int64 v66; // [rsp-8h] [rbp-1E8h]
  __int64 v67; // [rsp+0h] [rbp-1E0h]
  void **v68; // [rsp+8h] [rbp-1D8h]
  __int64 v69; // [rsp+10h] [rbp-1D0h]
  __int64 v70; // [rsp+10h] [rbp-1D0h]
  __int64 v71; // [rsp+10h] [rbp-1D0h]
  __int64 v72; // [rsp+28h] [rbp-1B8h]
  __int64 v73; // [rsp+28h] [rbp-1B8h]
  _QWORD v74[2]; // [rsp+30h] [rbp-1B0h] BYREF
  char *v75; // [rsp+40h] [rbp-1A0h] BYREF
  __int64 v76; // [rsp+48h] [rbp-198h]
  _WORD v77[32]; // [rsp+50h] [rbp-190h] BYREF
  _BYTE *v78; // [rsp+90h] [rbp-150h] BYREF
  __int64 v79; // [rsp+98h] [rbp-148h]
  _BYTE v80[128]; // [rsp+A0h] [rbp-140h] BYREF
  void *v81[2]; // [rsp+120h] [rbp-C0h] BYREF
  _BYTE v82[176]; // [rsp+130h] [rbp-B0h] BYREF

  v5 = a2;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v7 = *(_QWORD *)(a1 - 8);
  else
    v7 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v8 = *(_BYTE *)(sub_157ED60(*(_QWORD *)(v7 + 24)) + 16) == 31;
  v10 = *(_DWORD *)(a1 + 20);
  if ( v8 )
  {
    v72 = 0;
    v11 = v10 & 0xFFFFFFF;
  }
  else
  {
    v11 = v10 & 0xFFFFFFF;
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v12 = *(_QWORD *)(a1 - 8);
    else
      v12 = a1 - 24LL * v11;
    v72 = *(_QWORD *)(v12 + 24);
  }
  v13 = 0;
  v78 = v80;
  v81[0] = v82;
  v79 = 0x1000000000LL;
  v81[1] = (void *)0x1000000000LL;
  v69 = 0;
  if ( v11 >> 1 == 1 )
    goto LABEL_70;
  v14 = 1;
  v15 = v72;
  v16 = (v11 >> 1) - 1;
  do
  {
    v17 = 24;
    if ( (_DWORD)v14 != -1 )
      v17 = 24LL * (unsigned int)(2 * v14 + 1);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    {
      v18 = *(_QWORD *)(a1 - 8);
      v19 = *(_QWORD *)(v18 + v17);
      if ( !v15 )
        goto LABEL_25;
    }
    else
    {
      v18 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      v19 = *(_QWORD *)(v18 + v17);
      if ( !v15 )
      {
LABEL_25:
        v70 = v13;
        v73 = v14;
        v75 = *(char **)(v18 + 24LL * (unsigned int)(2 * v14));
        sub_1B47640((__int64)&v78, &v75, 3LL * (unsigned int)(2 * v14), v15, v13, v9);
        v13 = v70;
        v15 = v19;
        goto LABEL_26;
      }
    }
    if ( v19 == v15 )
      goto LABEL_25;
    if ( v13 && v19 != v13 )
    {
      v20 = 0;
      goto LABEL_17;
    }
    v71 = v15;
    v73 = v14;
    v75 = *(char **)(v18 + 24LL * (unsigned int)(2 * v14));
    sub_1B47640((__int64)v81, &v75, 3LL * (unsigned int)(2 * v14), v15, v13, v9);
    v15 = v71;
    v13 = v19;
LABEL_26:
    ++v14;
  }
  while ( v16 != v73 );
  v69 = v13;
  v5 = a2;
  v72 = v15;
  if ( (_DWORD)v79 && (unsigned __int8)sub_1B43AA0((void **)&v78) )
  {
    v22 = v69;
    v23 = v72;
    v68 = (void **)&v78;
    v69 = v72;
    v72 = v22;
    goto LABEL_30;
  }
LABEL_70:
  v68 = v81;
  v20 = sub_1B43AA0(v81);
  if ( (_BYTE)v20 )
  {
LABEL_30:
    v24 = (__int64 *)sub_15A2B90(*((__int64 **)*v68 + *((unsigned int *)v68 + 2) - 1), 0, 0, v23, a3, a4, a5);
    v25 = *((unsigned int *)v68 + 2);
    v67 = (__int64)v24;
    v26 = sub_15A0680(*v24, v25, 0);
    v27 = *(_QWORD *)sub_13CF970(a1);
    if ( !sub_1593BB0(v67, v25, v28, v29) )
    {
      v74[0] = sub_1649960(v27);
      v75 = (char *)v74;
      v74[1] = v50;
      v77[0] = 773;
      v76 = (__int64)".off";
      if ( *(_BYTE *)(v27 + 16) > 0x10u || *(_BYTE *)(v67 + 16) > 0x10u )
      {
        v25 = 11;
        v53 = sub_17D2EF0(v5, 11, (__int64 *)v27, v67, (__int64 *)&v75, 0, 0);
        v30 = v65;
        v31 = v66;
        v27 = (__int64)v53;
      }
      else
      {
        v25 = v67;
        v27 = sub_15A2B30((__int64 *)v27, v67, 0, 0, a3, a4, a5);
      }
    }
    if ( sub_1593BB0(v26, v25, v30, v31) && *((_DWORD *)v68 + 2) )
    {
      v52 = (__int64 *)sub_16498A0(a1);
      v32 = sub_159C4F0(v52);
    }
    else
    {
      v75 = "switch";
      v77[0] = 259;
      v32 = sub_12AA0C0(v5, 0x24u, (_BYTE *)v27, v26, (__int64)&v75);
    }
    v33 = sub_1B48C20(v5, v32, v69, v72, 0, 0);
    if ( sub_1B43680(a1) )
    {
      v76 = 0x800000000LL;
      v75 = (char *)v77;
      sub_1B43970(a1, (__int64)&v75);
      v51 = (*(_DWORD *)(a1 + 20) & 0xFFFFFFFu) >> 1;
      if ( v51 == (_DWORD)v76 )
      {
        v54 = v51;
        if ( v51 )
        {
          v55 = 0;
          v56 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
          v57 = v75;
          v58 = *(_BYTE *)(a1 + 23) & 0x40;
          v59 = v56;
          v60 = 1;
          v61 = &v75[8 * v54];
          v62 = 0;
          do
          {
            v63 = v59;
            if ( v58 )
              v63 = *(_QWORD *)(a1 - 8);
            v64 = *(_QWORD *)v57;
            if ( v69 == *(_QWORD *)(v63 + 24LL * v60) )
              v55 += v64;
            else
              v62 += v64;
            v57 += 8;
            v60 += 2;
          }
          while ( v61 != v57 );
          if ( v55 > 0xFFFFFFFF || v62 > 0xFFFFFFFF )
          {
            do
            {
              do
              {
                v55 >>= 1;
                v62 >>= 1;
              }
              while ( v55 > 0xFFFFFFFF );
            }
            while ( v62 > 0xFFFFFFFF );
          }
        }
        else
        {
          LODWORD(v62) = 0;
          LODWORD(v55) = 0;
        }
        sub_1B423A0((__int64)v33, v55, v62);
      }
      if ( v75 != (char *)v77 )
        _libc_free((unsigned __int64)v75);
    }
    for ( i = *(_QWORD *)(v69 + 48); ; i = *(_QWORD *)(i + 8) )
    {
      if ( !i )
        BUG();
      if ( *(_BYTE *)(i - 8) != 77 )
        break;
      v35 = *((_DWORD *)v68 + 2);
      v36 = *(_QWORD *)(sub_13CF970(a1) + 24);
      if ( v69 != v36 || !v36 )
        --v35;
      v37 = 0;
      if ( v35 )
      {
        do
        {
          v38 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
          if ( v38 )
          {
            v39 = 0;
            v40 = 24LL * *(unsigned int *)(i + 32) + 8;
            while ( 1 )
            {
              v41 = i - 24 - 24LL * v38;
              if ( (*(_BYTE *)(i - 1) & 0x40) != 0 )
                v41 = *(_QWORD *)(i - 32);
              if ( *(_QWORD *)(a1 + 40) == *(_QWORD *)(v41 + v40) )
                break;
              ++v39;
              v40 += 8;
              if ( v38 == v39 )
                goto LABEL_52;
            }
          }
          else
          {
LABEL_52:
            v39 = -1;
          }
          ++v37;
          sub_15F5350(i - 24, v39, 1);
        }
        while ( v37 != v35 );
      }
    }
    for ( j = *(_QWORD *)(v72 + 48); ; j = *(_QWORD *)(j + 8) )
    {
      if ( !j )
        BUG();
      if ( *(_BYTE *)(j - 8) != 77 )
        break;
      v43 = ((*(_DWORD *)(a1 + 20) & 0xFFFFFFFu) >> 1) - 1 - *((_DWORD *)v68 + 2);
      v44 = *(_QWORD *)(sub_13CF970(a1) + 24);
      if ( !v44 || v72 != v44 )
        --v43;
      v45 = 0;
      if ( v43 )
      {
        do
        {
          v46 = *(_DWORD *)(j - 4) & 0xFFFFFFF;
          if ( v46 )
          {
            v47 = 0;
            v48 = 24LL * *(unsigned int *)(j + 32) + 8;
            while ( 1 )
            {
              v49 = j - 24 - 24LL * v46;
              if ( (*(_BYTE *)(j - 1) & 0x40) != 0 )
                v49 = *(_QWORD *)(j - 32);
              if ( *(_QWORD *)(a1 + 40) == *(_QWORD *)(v49 + v48) )
                break;
              ++v47;
              v48 += 8;
              if ( v46 == v47 )
                goto LABEL_68;
            }
          }
          else
          {
LABEL_68:
            v47 = -1;
          }
          ++v45;
          sub_15F5350(j - 24, v47, 1);
        }
        while ( v45 != v43 );
      }
    }
    v20 = 1;
    sub_15F20C0((_QWORD *)a1);
  }
LABEL_17:
  if ( v81[0] != v82 )
    _libc_free((unsigned __int64)v81[0]);
  if ( v78 != v80 )
    _libc_free((unsigned __int64)v78);
  return v20;
}
