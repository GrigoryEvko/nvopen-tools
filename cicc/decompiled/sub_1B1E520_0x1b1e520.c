// Function: sub_1B1E520
// Address: 0x1b1e520
//
__int64 __fastcall sub_1B1E520(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v3; // rsi
  __int64 *v4; // r14
  __int64 v5; // r13
  __int64 result; // rax
  __int64 *v7; // rbx
  __int64 **v8; // rdx
  __int64 v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rbx
  _QWORD *v14; // r12
  _QWORD *v15; // rax
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // rax
  _QWORD *v19; // r15
  __int64 v20; // r13
  __int64 v21; // r14
  _QWORD *v22; // rdx
  __int64 v23; // rax
  __int64 *v24; // r13
  __int64 *v25; // r15
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // r13
  int v32; // eax
  __int64 v33; // rax
  int v34; // edx
  __int64 v35; // rdx
  __int64 **v36; // rax
  __int64 *v37; // rcx
  unsigned __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r12
  __int64 *v43; // rax
  __int64 v44; // rbx
  __int64 v45; // rax
  __int64 v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // r8
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  __int64 v53; // r15
  int v54; // eax
  __int64 v55; // rax
  int v56; // edx
  __int64 v57; // rdx
  _QWORD *v58; // rax
  __int64 v59; // rcx
  unsigned __int64 v60; // rdx
  __int64 v61; // rdx
  __int64 v62; // rdx
  __int64 v63; // r12
  __int64 v64; // rdx
  __int64 v65; // rcx
  _QWORD *v66; // rdx
  int v67; // edx
  int v68; // r9d
  __int64 *v69; // [rsp+8h] [rbp-D8h]
  __int64 v70; // [rsp+10h] [rbp-D0h]
  __int64 v71; // [rsp+18h] [rbp-C8h]
  __int64 v72; // [rsp+20h] [rbp-C0h]
  __int64 v73; // [rsp+28h] [rbp-B8h]
  __int64 v74; // [rsp+38h] [rbp-A8h]
  __int64 *v75; // [rsp+40h] [rbp-A0h]
  _QWORD v77[2]; // [rsp+50h] [rbp-90h] BYREF
  __int64 *v78; // [rsp+60h] [rbp-80h] BYREF
  __int64 v79; // [rsp+68h] [rbp-78h]
  _WORD v80[56]; // [rsp+70h] [rbp-70h] BYREF

  v73 = sub_13FA090(*(_QWORD *)a1);
  v2 = *(__int64 **)a2;
  v3 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  v75 = (__int64 *)v3;
  if ( v2 == (__int64 *)v3 )
  {
    result = v73;
    v5 = *(_QWORD *)(v73 + 48);
    goto LABEL_50;
  }
  v4 = v2;
  v5 = *(_QWORD *)(v73 + 48);
  do
  {
    result = v5;
    v7 = (__int64 *)*v4;
    if ( !v5 )
      goto LABEL_91;
    if ( *(_BYTE *)(v5 - 8) == 77 )
    {
      while ( 1 )
      {
        if ( (*(_BYTE *)(result - 1) & 0x40) != 0 )
          v8 = *(__int64 ***)(result - 32);
        else
          v8 = (__int64 **)(result - 24 - 24LL * (*(_DWORD *)(result - 4) & 0xFFFFFFF));
        if ( v7 == *v8 )
          goto LABEL_49;
        result = *(_QWORD *)(result + 8);
        if ( !result )
          break;
        if ( *(_BYTE *)(result - 8) != 77 )
          goto LABEL_10;
      }
LABEL_91:
      BUG();
    }
LABEL_10:
    v9 = v5 - 24;
    v77[0] = sub_1649960(*v4);
    v80[0] = 773;
    v78 = v77;
    v77[1] = v10;
    v79 = (__int64)".lver";
    v74 = *v7;
    v11 = sub_1648B60(64);
    v3 = v74;
    v12 = v11;
    if ( v11 )
    {
      v72 = v11;
      sub_15F1EA0(v11, v74, 53, 0, 0, v9);
      *(_DWORD *)(v12 + 56) = 2;
      sub_164B780(v12, (__int64 *)&v78);
      v3 = *(unsigned int *)(v12 + 56);
      sub_1648880(v12, v3, 1);
    }
    else
    {
      v72 = 0;
    }
    v78 = (__int64 *)v80;
    v79 = 0x800000000LL;
    if ( !v7[1] )
      goto LABEL_35;
    v71 = (__int64)v7;
    v13 = v7[1];
    v70 = v12;
    v69 = v4;
    do
    {
      while ( 1 )
      {
        v19 = sub_1648700(v13);
        v20 = v19[5];
        v21 = *(_QWORD *)a1;
        v22 = *(_QWORD **)(*(_QWORD *)a1 + 72LL);
        v15 = *(_QWORD **)(*(_QWORD *)a1 + 64LL);
        if ( v22 == v15 )
        {
          v14 = &v15[*(unsigned int *)(v21 + 84)];
          if ( v15 == v14 )
          {
            v66 = *(_QWORD **)(*(_QWORD *)a1 + 64LL);
          }
          else
          {
            do
            {
              if ( v20 == *v15 )
                break;
              ++v15;
            }
            while ( v14 != v15 );
            v66 = v14;
          }
        }
        else
        {
          v3 = v19[5];
          v14 = &v22[*(unsigned int *)(v21 + 80)];
          v15 = sub_16CC9F0(v21 + 56, v20);
          if ( v20 == *v15 )
          {
            v64 = *(_QWORD *)(v21 + 72);
            v65 = v64 == *(_QWORD *)(v21 + 64) ? *(unsigned int *)(v21 + 84) : *(unsigned int *)(v21 + 80);
            v66 = (_QWORD *)(v64 + 8 * v65);
          }
          else
          {
            v18 = *(_QWORD *)(v21 + 72);
            if ( v18 != *(_QWORD *)(v21 + 64) )
            {
              v15 = (_QWORD *)(v18 + 8LL * *(unsigned int *)(v21 + 80));
              goto LABEL_17;
            }
            v15 = (_QWORD *)(v18 + 8LL * *(unsigned int *)(v21 + 84));
            v66 = v15;
          }
        }
        if ( v15 != v66 )
        {
          while ( *v15 >= 0xFFFFFFFFFFFFFFFELL )
          {
            if ( v66 == ++v15 )
            {
              if ( v14 != v15 )
                goto LABEL_18;
              goto LABEL_29;
            }
          }
        }
LABEL_17:
        if ( v14 == v15 )
          break;
LABEL_18:
        v13 = *(_QWORD *)(v13 + 8);
        if ( !v13 )
          goto LABEL_32;
      }
LABEL_29:
      v23 = (unsigned int)v79;
      if ( (unsigned int)v79 >= HIDWORD(v79) )
      {
        v3 = (__int64)v80;
        sub_16CD150((__int64)&v78, v80, 0, 8, v16, v17);
        v23 = (unsigned int)v79;
      }
      v78[v23] = (__int64)v19;
      LODWORD(v79) = v79 + 1;
      v13 = *(_QWORD *)(v13 + 8);
    }
    while ( v13 );
LABEL_32:
    v7 = (__int64 *)v71;
    v12 = v70;
    v24 = &v78[(unsigned int)v79];
    v4 = v69;
    if ( v78 != v24 )
    {
      v25 = v78;
      do
      {
        v26 = *v25;
        v3 = v71;
        ++v25;
        sub_1648780(v26, v71, v70);
      }
      while ( v24 != v25 );
    }
LABEL_35:
    v31 = sub_13F9E70(*(_QWORD *)a1);
    v32 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
    if ( v32 == *(_DWORD *)(v12 + 56) )
    {
      sub_15F55D0(v12, v3, v27, v28, v29, v30);
      v32 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
    }
    v33 = (v32 + 1) & 0xFFFFFFF;
    v34 = v33 | *(_DWORD *)(v12 + 20) & 0xF0000000;
    *(_DWORD *)(v12 + 20) = v34;
    if ( (v34 & 0x40000000) != 0 )
      v35 = *(_QWORD *)(v12 - 8);
    else
      v35 = v72 - 24 * v33;
    v36 = (__int64 **)(v35 + 24LL * (unsigned int)(v33 - 1));
    if ( *v36 )
    {
      v37 = v36[1];
      v38 = (unsigned __int64)v36[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v38 = v37;
      if ( v37 )
      {
        v3 = v37[2] & 3;
        v37[2] = v3 | v38;
      }
    }
    *v36 = v7;
    v39 = v7[1];
    v36[1] = (__int64 *)v39;
    if ( v39 )
    {
      v3 = (unsigned __int64)(v36 + 1) | *(_QWORD *)(v39 + 16) & 3LL;
      *(_QWORD *)(v39 + 16) = v3;
    }
    v36[2] = (__int64 *)((unsigned __int64)(v7 + 1) | (unsigned __int64)v36[2] & 3);
    v7[1] = (__int64)v36;
    v40 = *(_DWORD *)(v12 + 20) & 0xFFFFFFF;
    if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
      v41 = *(_QWORD *)(v12 - 8);
    else
      v41 = v72 - 24 * v40;
    *(_QWORD *)(v41 + 8LL * (unsigned int)(v40 - 1) + 24LL * *(unsigned int *)(v12 + 56) + 8) = v31;
    if ( v78 != (__int64 *)v80 )
      _libc_free((unsigned __int64)v78);
    result = v73;
    v5 = *(_QWORD *)(v73 + 48);
LABEL_49:
    ++v4;
  }
  while ( v75 != v4 );
  while ( 1 )
  {
LABEL_50:
    if ( !v5 )
      BUG();
    if ( *(_BYTE *)(v5 - 8) != 77 )
      return result;
    v42 = v5 - 24;
    if ( (*(_BYTE *)(v5 - 1) & 0x40) != 0 )
      v43 = *(__int64 **)(v5 - 32);
    else
      v43 = (__int64 *)(v42 - 24LL * (*(_DWORD *)(v5 - 4) & 0xFFFFFFF));
    v44 = *v43;
    v45 = *(unsigned int *)(a1 + 40);
    if ( (_DWORD)v45 )
    {
      v46 = *(_QWORD *)(a1 + 24);
      v3 = ((_DWORD)v45 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
      v47 = v46 + (v3 << 6);
      v48 = *(_QWORD *)(v47 + 24);
      if ( v44 == v48 )
      {
LABEL_56:
        if ( v47 != v46 + (v45 << 6) )
          v44 = *(_QWORD *)(v47 + 56);
      }
      else
      {
        v67 = 1;
        while ( v48 != -8 )
        {
          v68 = v67 + 1;
          v3 = ((_DWORD)v45 - 1) & (unsigned int)(v67 + v3);
          v47 = v46 + ((unsigned __int64)(unsigned int)v3 << 6);
          v48 = *(_QWORD *)(v47 + 24);
          if ( v44 == v48 )
            goto LABEL_56;
          v67 = v68;
        }
      }
    }
    v53 = sub_13F9E70(*(_QWORD *)(a1 + 8));
    v54 = *(_DWORD *)(v5 - 4) & 0xFFFFFFF;
    if ( v54 == *(_DWORD *)(v5 + 32) )
    {
      sub_15F55D0(v5 - 24, v3, v49, v50, v51, v52);
      v54 = *(_DWORD *)(v5 - 4) & 0xFFFFFFF;
    }
    v55 = (v54 + 1) & 0xFFFFFFF;
    v56 = v55 | *(_DWORD *)(v5 - 4) & 0xF0000000;
    *(_DWORD *)(v5 - 4) = v56;
    if ( (v56 & 0x40000000) != 0 )
      v57 = *(_QWORD *)(v5 - 32);
    else
      v57 = v42 - 24 * v55;
    v58 = (_QWORD *)(v57 + 24LL * (unsigned int)(v55 - 1));
    if ( *v58 )
    {
      v59 = v58[1];
      v60 = v58[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v60 = v59;
      if ( v59 )
      {
        v3 = *(_QWORD *)(v59 + 16) & 3LL;
        *(_QWORD *)(v59 + 16) = v3 | v60;
      }
    }
    *v58 = v44;
    if ( v44 )
    {
      v61 = *(_QWORD *)(v44 + 8);
      v58[1] = v61;
      if ( v61 )
      {
        v3 = (unsigned __int64)(v58 + 1) | *(_QWORD *)(v61 + 16) & 3LL;
        *(_QWORD *)(v61 + 16) = v3;
      }
      v58[2] = (v44 + 8) | v58[2] & 3LL;
      *(_QWORD *)(v44 + 8) = v58;
    }
    v62 = *(_DWORD *)(v5 - 4) & 0xFFFFFFF;
    if ( (*(_BYTE *)(v5 - 1) & 0x40) != 0 )
      v63 = *(_QWORD *)(v5 - 32);
    else
      v63 = v42 - 24 * v62;
    result = 8LL * (unsigned int)(v62 - 1) + 24LL * *(unsigned int *)(v5 + 32);
    *(_QWORD *)(v63 + result + 8) = v53;
    v5 = *(_QWORD *)(v5 + 8);
  }
}
