// Function: sub_1651200
// Address: 0x1651200
//
void __fastcall sub_1651200(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 *v3; // r12
  void *v4; // rdi
  unsigned int v5; // eax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // r13
  _BYTE *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r14
  __int64 v12; // r15
  __int64 v13; // r15
  char *v14; // r14
  _QWORD *v15; // rax
  unsigned __int64 *v16; // rbx
  int v17; // eax
  __int64 v18; // r15
  unsigned __int64 v19; // rax
  unsigned __int64 *v20; // r15
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // rdx
  unsigned __int64 *v23; // rax
  unsigned __int64 *v24; // rsi
  __int64 v25; // r15
  __int64 v26; // rdx
  __int64 v27; // r13
  unsigned int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // r15
  __int64 v31; // r12
  __int64 v32; // rcx
  unsigned __int64 v33; // rbx
  unsigned __int64 v34; // rax
  unsigned __int64 *v35; // rdx
  unsigned __int64 v36; // r15
  unsigned __int64 *v37; // r12
  unsigned __int64 v38; // rax
  unsigned __int64 *v39; // r15
  unsigned __int64 v40; // rcx
  unsigned __int64 v41; // rsi
  unsigned __int64 *i; // rax
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rdx
  __int64 *v45; // rdi
  int v46; // ecx
  unsigned __int64 v47; // rsi
  unsigned __int64 *v48; // rax
  unsigned __int64 v49; // r8
  __int64 v50; // rax
  __int64 *v51; // r12
  __int64 v52; // r15
  const char *v53; // rax
  __int64 v54; // [rsp+0h] [rbp-160h]
  __int64 v55; // [rsp+8h] [rbp-158h]
  __int64 *v56; // [rsp+10h] [rbp-150h]
  unsigned __int64 *v57; // [rsp+18h] [rbp-148h]
  __int64 *v58; // [rsp+18h] [rbp-148h]
  unsigned __int64 v59; // [rsp+18h] [rbp-148h]
  unsigned __int64 *v60; // [rsp+20h] [rbp-140h]
  unsigned __int64 *v61; // [rsp+20h] [rbp-140h]
  __int64 *v62; // [rsp+20h] [rbp-140h]
  _QWORD v63[2]; // [rsp+30h] [rbp-130h] BYREF
  char v64; // [rsp+40h] [rbp-120h]
  char v65; // [rsp+41h] [rbp-11Fh]
  void *src; // [rsp+50h] [rbp-110h] BYREF
  __int64 v67; // [rsp+58h] [rbp-108h]
  _BYTE v68[64]; // [rsp+60h] [rbp-100h] BYREF
  char *v69; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v70; // [rsp+A8h] [rbp-B8h]
  char v71; // [rsp+B0h] [rbp-B0h] BYREF
  char v72; // [rsp+B1h] [rbp-AFh]

  v2 = a2;
  v3 = a1;
  ++a1[20];
  v4 = (void *)a1[22];
  if ( v4 != (void *)v3[21] )
  {
    v5 = 4 * (*((_DWORD *)v3 + 47) - *((_DWORD *)v3 + 48));
    v6 = *((unsigned int *)v3 + 46);
    if ( v5 < 0x20 )
      v5 = 32;
    if ( v5 < (unsigned int)v6 )
    {
      sub_16CC920(v3 + 20);
      goto LABEL_7;
    }
    memset(v4, -1, 8 * v6);
  }
  *(__int64 *)((char *)v3 + 188) = 0;
LABEL_7:
  if ( !sub_157EBA0(a2) )
  {
    v72 = 1;
    v69 = "Basic Block does not have terminator!";
    v71 = 3;
    sub_164FF40(v3, (__int64)&v69);
    if ( *v3 )
      sub_164FA80(v3, a2);
    return;
  }
  v7 = *(_QWORD *)(a2 + 48);
  if ( !v7 )
    BUG();
  if ( *(_BYTE *)(v7 - 8) != 77 )
    goto LABEL_10;
  v10 = *(_QWORD *)(a2 + 8);
  if ( v10 )
  {
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v10) + 16) - 25) > 9u )
    {
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
        goto LABEL_52;
    }
    v11 = v10;
    v12 = 0;
    src = v68;
    v67 = 0x800000000LL;
    while ( 1 )
    {
      v11 = *(_QWORD *)(v11 + 8);
      if ( !v11 )
        break;
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v11) + 16) - 25) <= 9u )
      {
        v11 = *(_QWORD *)(v11 + 8);
        ++v12;
        if ( !v11 )
          goto LABEL_25;
      }
    }
LABEL_25:
    v13 = v12 + 1;
    if ( v13 > 8 )
    {
      sub_16CD150(&src, v68, v13, 8);
      v14 = (char *)src + 8 * (unsigned int)v67;
    }
    else
    {
      v14 = v68;
    }
    v15 = sub_1648700(v10);
LABEL_30:
    if ( v14 )
      *(_QWORD *)v14 = v15[5];
    while ( 1 )
    {
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
        break;
      v15 = sub_1648700(v10);
      if ( (unsigned __int8)(*((_BYTE *)v15 + 16) - 25) <= 9u )
      {
        v14 += 8;
        goto LABEL_30;
      }
    }
    v16 = (unsigned __int64 *)src;
    v69 = &v71;
    v17 = v67 + v13;
    v18 = 8LL * (unsigned int)(v67 + v13);
    LODWORD(v67) = v17;
    v70 = 0x800000000LL;
    if ( src != (char *)src + v18 )
    {
      v60 = (unsigned __int64 *)((char *)src + v18);
      _BitScanReverse64(&v19, v18 >> 3);
      sub_164E230((char *)src, (unsigned __int64 *)((char *)src + v18), 2LL * (int)(63 - (v19 ^ 0x3F)));
      if ( (unsigned __int64)v18 <= 0x80 )
      {
        sub_164E0C0(v16, v60);
      }
      else
      {
        v20 = v16 + 16;
        sub_164E0C0(v16, v16 + 16);
        if ( v16 + 16 != v60 )
        {
          do
          {
            v21 = *v20;
            v22 = *(v20 - 1);
            v23 = v20 - 1;
            if ( *v20 >= v22 )
            {
              v24 = v20;
            }
            else
            {
              do
              {
                v23[1] = v22;
                v24 = v23;
                v22 = *--v23;
              }
              while ( v21 < v22 );
            }
            ++v20;
            *v24 = v21;
          }
          while ( v20 != v60 );
        }
      }
    }
  }
  else
  {
LABEL_52:
    src = v68;
    v67 = 0x800000000LL;
    v69 = &v71;
    v70 = 0x800000000LL;
  }
  v25 = sub_157F280(v2);
  v55 = v26;
  if ( v25 == v26 )
  {
LABEL_80:
    if ( v69 != &v71 )
      _libc_free((unsigned __int64)v69);
    if ( src != v68 )
      _libc_free((unsigned __int64)src);
    v7 = *(_QWORD *)(v2 + 48);
LABEL_10:
    if ( v2 + 40 != v7 )
    {
      while ( 1 )
      {
        if ( !v7 )
          BUG();
        if ( v2 != *(_QWORD *)(v7 + 16) )
          break;
        v7 = *(_QWORD *)(v7 + 8);
        if ( v2 + 40 == v7 )
          return;
      }
      v8 = *v3;
      v72 = 1;
      v69 = "Instruction has bogus parent pointer!";
      v71 = 3;
      if ( v8 )
      {
        sub_16E2CE0(&v69, v8);
        v9 = *(_BYTE **)(v8 + 24);
        if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 16) )
        {
          sub_16E7DE0(v8, 10);
        }
        else
        {
          *(_QWORD *)(v8 + 24) = v9 + 1;
          *v9 = 10;
        }
      }
      *((_BYTE *)v3 + 72) = 1;
    }
    return;
  }
  v56 = v3;
  v54 = v2;
  v27 = v25;
  while ( 1 )
  {
    v28 = *(_DWORD *)(v27 + 20) & 0xFFFFFFF;
    if ( !v28 )
    {
      v65 = 1;
      v51 = v56;
      v52 = v27;
      v53 = "PHI nodes must have at least one entry.  If the block is dead, the PHI should be removed!";
      goto LABEL_100;
    }
    if ( (_DWORD)v67 != v28 )
    {
      v65 = 1;
      v51 = v56;
      v52 = v27;
      v53 = "PHINode should have one entry for each predecessor of its parent basic block!";
LABEL_100:
      v63[0] = v53;
      v64 = 3;
      sub_164FF40(v51, (__int64)v63);
      if ( *v51 )
        sub_164FA80(v51, v52);
      goto LABEL_87;
    }
    v29 = 0;
    LODWORD(v70) = 0;
    if ( v28 <= (unsigned __int64)HIDWORD(v70)
      || (sub_16CD150(&v69, &v71, v28, 16), v29 = (unsigned int)v70, (v28 = *(_DWORD *)(v27 + 20) & 0xFFFFFFF) != 0) )
    {
      v30 = 0;
      v31 = 8LL * v28;
      do
      {
        if ( (*(_BYTE *)(v27 + 23) & 0x40) != 0 )
          v32 = *(_QWORD *)(v27 - 8);
        else
          v32 = v27 - 24LL * (*(_DWORD *)(v27 + 20) & 0xFFFFFFF);
        v33 = *(_QWORD *)(v32 + 3 * v30);
        v34 = *(_QWORD *)(v30 + v32 + 24LL * *(unsigned int *)(v27 + 56) + 8);
        if ( HIDWORD(v70) <= (unsigned int)v29 )
        {
          v59 = *(_QWORD *)(v30 + v32 + 24LL * *(unsigned int *)(v27 + 56) + 8);
          sub_16CD150(&v69, &v71, 0, 16);
          v29 = (unsigned int)v70;
          v34 = v59;
        }
        v35 = (unsigned __int64 *)&v69[16 * v29];
        v30 += 8;
        *v35 = v34;
        v35[1] = v33;
        v29 = (unsigned int)(v70 + 1);
        LODWORD(v70) = v70 + 1;
      }
      while ( v31 != v30 );
    }
    v36 = 16LL * (unsigned int)v29;
    v37 = (unsigned __int64 *)&v69[v36];
    if ( &v69[v36] != v69 )
    {
      v57 = (unsigned __int64 *)v69;
      _BitScanReverse64(&v38, (__int64)v36 >> 4);
      sub_164E3F0((unsigned __int64 *)v69, (unsigned __int64 *)&v69[v36], 2LL * (int)(63 - (v38 ^ 0x3F)));
      if ( v36 <= 0x100 )
      {
        sub_164EB90(v57, v37);
      }
      else
      {
        v39 = v57 + 32;
        sub_164EB90(v57, v57 + 32);
        if ( v37 != v57 + 32 )
        {
          do
          {
            v40 = *v39;
            v41 = v39[1];
            for ( i = v39; ; i[3] = v44 )
            {
              v43 = *(i - 2);
              if ( v40 >= v43 && (v40 != v43 || v41 >= *(i - 1)) )
                break;
              *i = v43;
              v44 = *(i - 1);
              i -= 2;
            }
            v39 += 2;
            *i = v40;
            i[1] = v41;
          }
          while ( v37 != v39 );
        }
      }
      LODWORD(v29) = v70;
    }
    if ( (_DWORD)v29 )
      break;
LABEL_75:
    v50 = *(_QWORD *)(v27 + 32);
    if ( !v50 )
      BUG();
    v27 = 0;
    if ( *(_BYTE *)(v50 - 8) == 77 )
      v27 = v50 - 24;
    if ( v55 == v27 )
    {
      v3 = v56;
      v2 = v54;
      goto LABEL_80;
    }
  }
  v45 = (__int64 *)src;
  v46 = 0;
  v47 = *(_QWORD *)v69;
  v48 = (unsigned __int64 *)(v69 + 16);
  while ( *v45 == v47 )
  {
    if ( ++v46 == (_DWORD)v29 )
      goto LABEL_75;
    v49 = *v48;
    if ( v47 == *v48 && v48[1] != *(v48 - 1) )
    {
      v58 = (__int64 *)v48;
      v61 = v48 - 2;
      v65 = 1;
      v63[0] = "PHI node has multiple entries for the same basic block with different incoming values!";
      v64 = 3;
      sub_164FF40(v56, (__int64)v63);
      if ( *v56 )
      {
        sub_164FA80(v56, v27);
        sub_164FA80(v56, *v58);
        sub_164FA80(v56, v58[1]);
        sub_164FA80(v56, v61[1]);
      }
      goto LABEL_87;
    }
    ++v45;
    v48 += 2;
    v47 = v49;
  }
  v62 = (__int64 *)(v48 - 2);
  v65 = 1;
  v63[0] = "PHI node entries do not match predecessors!";
  v64 = 3;
  sub_164FF40(v56, (__int64)v63);
  if ( *v56 )
  {
    sub_164FA80(v56, v27);
    sub_164FA80(v56, *v62);
    sub_164FA80(v56, *v45);
  }
LABEL_87:
  if ( v69 != &v71 )
    _libc_free((unsigned __int64)v69);
  if ( src != v68 )
    _libc_free((unsigned __int64)src);
}
