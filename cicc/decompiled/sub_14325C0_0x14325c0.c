// Function: sub_14325C0
// Address: 0x14325c0
//
void __fastcall sub_14325C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  int v7; // eax
  _BYTE *v10; // rdi
  __int64 v12; // rdx
  __int64 v13; // r15
  __int64 v14; // r8
  unsigned __int64 *v15; // r9
  __int64 v16; // r15
  __int64 v17; // rax
  unsigned int v18; // esi
  int v19; // r10d
  __int64 v20; // rdi
  unsigned int v21; // eax
  unsigned __int64 *v22; // rdx
  unsigned __int64 v23; // rcx
  __int64 *v24; // r14
  __int64 *v25; // r12
  __int64 v26; // rdi
  __int64 v27; // rsi
  unsigned __int64 v28; // rdi
  _BYTE *v29; // rdi
  __int64 v32; // rdx
  __int64 v33; // r13
  __int64 *v34; // r12
  __int64 *v35; // rbx
  __int64 v36; // rdi
  __int64 v37; // rsi
  __int64 v38; // rax
  unsigned int v39; // esi
  __int64 v40; // r8
  int v41; // ebx
  __int64 *v42; // r10
  __int64 v43; // rax
  unsigned int v44; // edx
  __int64 *v45; // rcx
  __int64 v46; // rdi
  int v47; // edx
  _BYTE *v48; // rsi
  int v49; // r8d
  __int64 v50; // rax
  __int64 v51; // r10
  unsigned int v52; // ecx
  int v53; // edx
  _BYTE *v54; // rsi
  int v55; // r8d
  __int64 v56; // r10
  unsigned int v57; // ecx
  unsigned __int64 v58; // r11
  int v59; // edi
  unsigned __int64 *v60; // rsi
  int v61; // ebx
  __int64 v62; // rcx
  int v63; // ebx
  __int64 v64; // r9
  unsigned int v65; // esi
  int v66; // r8d
  __int64 *v67; // rdi
  int v68; // ebx
  int v69; // ebx
  __int64 v70; // r9
  int v71; // r8d
  unsigned int v72; // esi
  int v73; // edi
  __int64 v74; // [rsp+10h] [rbp-150h]
  unsigned __int64 v75; // [rsp+10h] [rbp-150h]
  __int64 v76; // [rsp+10h] [rbp-150h]
  char v78; // [rsp+27h] [rbp-139h] BYREF
  __int64 v79; // [rsp+28h] [rbp-138h] BYREF
  __int64 v80[2]; // [rsp+30h] [rbp-130h] BYREF
  _BYTE v81[32]; // [rsp+40h] [rbp-120h] BYREF
  _BYTE *v82; // [rsp+60h] [rbp-100h] BYREF
  __int64 v83; // [rsp+68h] [rbp-F8h]
  _BYTE v84[32]; // [rsp+70h] [rbp-F0h] BYREF
  __int64 *v85; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v86; // [rsp+98h] [rbp-C8h]
  _BYTE v87[192]; // [rsp+A0h] [rbp-C0h] BYREF

  v6 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v6 + 16) )
    BUG();
  v7 = *(_DWORD *)(v6 + 36);
  if ( v7 == 207 )
  {
    v29 = *(_BYTE **)(*(_QWORD *)(a1 + 24 * (2LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF))) + 24LL);
    if ( *v29 )
      return;
    v76 = sub_161E970(v29);
    v33 = v32;
    sub_16C1840(&v85);
    sub_16C1A90(&v85, v76, v33);
    sub_16C1AA0(&v85, &v82);
    v79 = (__int64)v82;
    v85 = (__int64 *)v87;
    v82 = v84;
    v86 = 0x400000000LL;
    v80[0] = (__int64)v81;
    v80[1] = 0x400000000LL;
    v83 = 0x400000000LL;
    v78 = 0;
    sub_14A88D0(&v85, v80, &v82, &v78, a1);
    if ( !v78 )
    {
LABEL_18:
      v34 = v85;
      v35 = &v85[2 * (unsigned int)v86];
      if ( v85 != v35 )
      {
        do
        {
          v36 = *v34;
          v37 = v34[1];
          v34 += 2;
          sub_1431F40(v36, v37, v79, a4, a6);
        }
        while ( v35 != v34 );
      }
      if ( v82 != v84 )
        _libc_free((unsigned __int64)v82);
      v28 = v80[0];
      if ( (_BYTE *)v80[0] == v81 )
        goto LABEL_13;
LABEL_12:
      _libc_free(v28);
LABEL_13:
      if ( v85 != (__int64 *)v87 )
        _libc_free((unsigned __int64)v85);
      return;
    }
    v39 = *(_DWORD *)(a2 + 24);
    if ( v39 )
    {
      v40 = *(_QWORD *)(a2 + 8);
      v41 = 1;
      v42 = 0;
      v43 = v79;
      v44 = (v39 - 1) & (37 * v79);
      v45 = (__int64 *)(v40 + 8LL * v44);
      v46 = *v45;
      if ( v79 == *v45 )
        goto LABEL_18;
      while ( v46 != -1 )
      {
        if ( v46 == -2 && !v42 )
          v42 = v45;
        v44 = (v39 - 1) & (v41 + v44);
        v45 = (__int64 *)(v40 + 8LL * v44);
        v46 = *v45;
        if ( v79 == *v45 )
          goto LABEL_18;
        ++v41;
      }
      if ( !v42 )
        v42 = v45;
      ++*(_QWORD *)a2;
      v47 = *(_DWORD *)(a2 + 16) + 1;
      if ( 4 * v47 < 3 * v39 )
      {
        if ( v39 - *(_DWORD *)(a2 + 20) - v47 > v39 >> 3 )
        {
LABEL_35:
          *(_DWORD *)(a2 + 16) = v47;
          if ( *v42 != -1 )
            --*(_DWORD *)(a2 + 20);
          *v42 = v43;
          v48 = *(_BYTE **)(a2 + 40);
          if ( v48 == *(_BYTE **)(a2 + 48) )
          {
            sub_9CA200(a2 + 32, v48, &v79);
          }
          else
          {
            if ( v48 )
            {
              *(_QWORD *)v48 = v79;
              v48 = *(_BYTE **)(a2 + 40);
            }
            *(_QWORD *)(a2 + 40) = v48 + 8;
          }
          goto LABEL_18;
        }
        sub_142F750(a2, v39);
        v68 = *(_DWORD *)(a2 + 24);
        if ( v68 )
        {
          v62 = v79;
          v69 = v68 - 1;
          v70 = *(_QWORD *)(a2 + 8);
          v67 = 0;
          v71 = 1;
          v47 = *(_DWORD *)(a2 + 16) + 1;
          v72 = v69 & (37 * v79);
          v42 = (__int64 *)(v70 + 8LL * v72);
          v43 = *v42;
          if ( v79 == *v42 )
            goto LABEL_35;
          while ( v43 != -1 )
          {
            if ( v43 == -2 && !v67 )
              v67 = v42;
            v72 = v69 & (v71 + v72);
            v42 = (__int64 *)(v70 + 8LL * v72);
            v43 = *v42;
            if ( v79 == *v42 )
              goto LABEL_35;
            ++v71;
          }
          goto LABEL_74;
        }
        goto LABEL_110;
      }
    }
    else
    {
      ++*(_QWORD *)a2;
    }
    sub_142F750(a2, 2 * v39);
    v61 = *(_DWORD *)(a2 + 24);
    if ( v61 )
    {
      v62 = v79;
      v63 = v61 - 1;
      v64 = *(_QWORD *)(a2 + 8);
      v47 = *(_DWORD *)(a2 + 16) + 1;
      v65 = v63 & (37 * v79);
      v42 = (__int64 *)(v64 + 8LL * v65);
      v43 = *v42;
      if ( v79 == *v42 )
        goto LABEL_35;
      v66 = 1;
      v67 = 0;
      while ( v43 != -1 )
      {
        if ( !v67 && v43 == -2 )
          v67 = v42;
        v65 = v63 & (v66 + v65);
        v42 = (__int64 *)(v64 + 8LL * v65);
        v43 = *v42;
        if ( v79 == *v42 )
          goto LABEL_35;
        ++v66;
      }
LABEL_74:
      v43 = v62;
      if ( v67 )
        v42 = v67;
      goto LABEL_35;
    }
LABEL_110:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
  if ( v7 == 208 )
  {
    v10 = *(_BYTE **)(*(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF))) + 24LL);
    if ( !*v10 )
    {
      v74 = sub_161E970(v10);
      v13 = v12;
      sub_16C1840(&v85);
      sub_16C1A90(&v85, v74, v13);
      sub_16C1AA0(&v85, &v82);
      v16 = *(_QWORD *)(a1 + 8);
      v75 = (unsigned __int64)v82;
      v80[0] = (__int64)v82;
      if ( !v16 )
        goto LABEL_9;
      while ( 1 )
      {
        v17 = sub_1648700(v16);
        if ( *(_BYTE *)(v17 + 16) != 78 )
          break;
        v38 = *(_QWORD *)(v17 - 24);
        if ( *(_BYTE *)(v38 + 16) || *(_DWORD *)(v38 + 36) != 4 )
          break;
        v16 = *(_QWORD *)(v16 + 8);
        if ( !v16 )
          goto LABEL_9;
      }
      v18 = *(_DWORD *)(a2 + 24);
      if ( v18 )
      {
        v14 = v18 - 1;
        v15 = 0;
        v19 = 1;
        v20 = *(_QWORD *)(a2 + 8);
        v21 = v14 & (37 * v75);
        v22 = (unsigned __int64 *)(v20 + 8LL * v21);
        v23 = *v22;
        if ( v75 == *v22 )
          goto LABEL_9;
        while ( v23 != -1 )
        {
          if ( !v15 && v23 == -2 )
            v15 = v22;
          v21 = v14 & (v19 + v21);
          v22 = (unsigned __int64 *)(v20 + 8LL * v21);
          v23 = *v22;
          if ( v75 == *v22 )
            goto LABEL_9;
          ++v19;
        }
        if ( !v15 )
          v15 = v22;
        ++*(_QWORD *)a2;
        v53 = *(_DWORD *)(a2 + 16) + 1;
        if ( 4 * v53 < 3 * v18 )
        {
          if ( v18 - *(_DWORD *)(a2 + 20) - v53 > v18 >> 3 )
            goto LABEL_44;
          sub_142F750(a2, v18);
          v55 = *(_DWORD *)(a2 + 24);
          if ( v55 )
          {
            v50 = v80[0];
            v14 = (unsigned int)(v55 - 1);
            v56 = *(_QWORD *)(a2 + 8);
            v57 = v14 & (37 * LODWORD(v80[0]));
            v15 = (unsigned __int64 *)(v56 + 8LL * v57);
            v53 = *(_DWORD *)(a2 + 16) + 1;
            v75 = *v15;
            if ( v80[0] != *v15 )
            {
              v58 = *v15;
              v59 = 1;
              v60 = 0;
              while ( v58 != -1 )
              {
                if ( v58 == -2 && !v60 )
                  v60 = v15;
                v57 = v14 & (v57 + v59);
                v15 = (unsigned __int64 *)(v56 + 8LL * v57);
                v58 = *v15;
                if ( v80[0] == *v15 )
                  goto LABEL_96;
                ++v59;
              }
LABEL_64:
              v75 = v50;
              if ( v60 )
                v15 = v60;
              goto LABEL_44;
            }
            goto LABEL_44;
          }
          goto LABEL_109;
        }
      }
      else
      {
        ++*(_QWORD *)a2;
      }
      sub_142F750(a2, 2 * v18);
      v49 = *(_DWORD *)(a2 + 24);
      if ( v49 )
      {
        v50 = v80[0];
        v14 = (unsigned int)(v49 - 1);
        v51 = *(_QWORD *)(a2 + 8);
        v52 = v14 & (37 * LODWORD(v80[0]));
        v15 = (unsigned __int64 *)(v51 + 8LL * v52);
        v53 = *(_DWORD *)(a2 + 16) + 1;
        v75 = *v15;
        if ( v80[0] != *v15 )
        {
          v58 = *v15;
          v73 = 1;
          v60 = 0;
          while ( v58 != -1 )
          {
            if ( !v60 && v58 == -2 )
              v60 = v15;
            v52 = v14 & (v52 + v73);
            v15 = (unsigned __int64 *)(v51 + 8LL * v52);
            v58 = *v15;
            if ( v80[0] == *v15 )
            {
LABEL_96:
              v75 = v58;
              goto LABEL_44;
            }
            ++v73;
          }
          goto LABEL_64;
        }
LABEL_44:
        *(_DWORD *)(a2 + 16) = v53;
        if ( *v15 != -1 )
          --*(_DWORD *)(a2 + 20);
        *v15 = v75;
        v54 = *(_BYTE **)(a2 + 40);
        if ( v54 == *(_BYTE **)(a2 + 48) )
        {
          sub_9CA200(a2 + 32, v54, v80);
        }
        else
        {
          if ( v54 )
          {
            *(_QWORD *)v54 = v80[0];
            v54 = *(_BYTE **)(a2 + 40);
          }
          *(_QWORD *)(a2 + 40) = v54 + 8;
        }
LABEL_9:
        v85 = (__int64 *)v87;
        v86 = 0x400000000LL;
        v82 = v84;
        v83 = 0x400000000LL;
        sub_14A87C0(&v85, &v82, a1, v84, v14, v15);
        v24 = v85;
        v25 = &v85[2 * (unsigned int)v86];
        if ( v85 != v25 )
        {
          do
          {
            v26 = *v24;
            v27 = v24[1];
            v24 += 2;
            sub_1431F40(v26, v27, v80[0], a3, a5);
          }
          while ( v25 != v24 );
        }
        v28 = (unsigned __int64)v82;
        if ( v82 == v84 )
          goto LABEL_13;
        goto LABEL_12;
      }
LABEL_109:
      ++*(_DWORD *)(a2 + 16);
      BUG();
    }
  }
}
