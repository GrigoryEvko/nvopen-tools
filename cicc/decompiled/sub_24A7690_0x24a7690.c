// Function: sub_24A7690
// Address: 0x24a7690
//
__int64 __fastcall sub_24A7690(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  unsigned int v9; // esi
  unsigned int v10; // ecx
  __int64 v11; // r10
  int v12; // r15d
  __int64 *v13; // r9
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // r9
  int v18; // r11d
  __int64 *v19; // r15
  unsigned int v20; // edx
  __int64 *v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 *v25; // r13
  __int64 v26; // rax
  unsigned int v28; // r15d
  int v29; // edi
  __int64 v30; // rax
  _QWORD *v31; // rax
  unsigned __int64 v32; // rdi
  __int64 v33; // rdi
  int v34; // edx
  int v35; // eax
  _QWORD *v36; // rax
  unsigned __int64 v37; // rdi
  unsigned __int64 v38; // r14
  char *v39; // rsi
  __int64 v40; // rdi
  __int64 v41; // rax
  bool v42; // cf
  unsigned __int64 v43; // rax
  unsigned __int64 v44; // r12
  __int64 v45; // rax
  unsigned __int64 v46; // r12
  __int64 *v47; // rsi
  _QWORD *v48; // r15
  unsigned __int64 *v49; // r12
  __int64 v50; // rax
  unsigned __int64 v51; // rdi
  int v52; // esi
  int v53; // esi
  __int64 v54; // r10
  unsigned int v55; // edx
  __int64 v56; // rdi
  int v57; // esi
  int v58; // esi
  __int64 *v59; // r11
  unsigned int v60; // [rsp+8h] [rbp-68h]
  __int64 v61; // [rsp+8h] [rbp-68h]
  __int64 v62; // [rsp+10h] [rbp-60h]
  unsigned int v63; // [rsp+10h] [rbp-60h]
  unsigned __int64 v64; // [rsp+10h] [rbp-60h]
  unsigned int v65; // [rsp+10h] [rbp-60h]
  unsigned int v66; // [rsp+10h] [rbp-60h]
  __int64 *v67; // [rsp+18h] [rbp-58h]
  __int64 v68; // [rsp+18h] [rbp-58h]
  unsigned int v69; // [rsp+18h] [rbp-58h]
  __int64 v70; // [rsp+18h] [rbp-58h]
  __int64 v71; // [rsp+18h] [rbp-58h]
  __int64 v72; // [rsp+18h] [rbp-58h]
  int v73; // [rsp+18h] [rbp-58h]
  __int64 *v74; // [rsp+28h] [rbp-48h] BYREF
  __int64 v75; // [rsp+30h] [rbp-40h] BYREF
  __int64 v76; // [rsp+38h] [rbp-38h]

  v4 = a1 + 32;
  v75 = a2;
  v9 = *(_DWORD *)(a1 + 56);
  v76 = 0;
  v10 = *(_DWORD *)(a1 + 48);
  if ( v9 )
  {
    v11 = *(_QWORD *)(a1 + 40);
    v12 = 1;
    v13 = 0;
    v14 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v15 = (__int64 *)(v11 + 16LL * v14);
    v16 = *v15;
    if ( a2 == *v15 )
    {
LABEL_3:
      v75 = a3;
      v76 = 0;
      goto LABEL_4;
    }
    while ( v16 != -4096 )
    {
      if ( v16 != -8192 || v13 )
        v15 = v13;
      v14 = (v9 - 1) & (v12 + v14);
      v16 = *(_QWORD *)(v11 + 16LL * v14);
      if ( a2 == v16 )
        goto LABEL_3;
      ++v12;
      v13 = v15;
      v15 = (__int64 *)(v11 + 16LL * v14);
    }
    v28 = v10 + 1;
    if ( !v13 )
      v13 = v15;
    ++*(_QWORD *)(a1 + 32);
    v29 = v10 + 1;
    v74 = v13;
    if ( 4 * v28 < 3 * v9 )
    {
      v30 = a2;
      if ( v9 - *(_DWORD *)(a1 + 52) - v28 <= v9 >> 3 )
      {
        v66 = v10;
        v72 = v4;
        sub_24A3FB0(v4, v9);
        sub_24A2A40(v72, &v75, &v74);
        v30 = v75;
        v13 = v74;
        v4 = v72;
        v29 = *(_DWORD *)(a1 + 48) + 1;
        v10 = v66;
      }
      goto LABEL_22;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 32);
    v28 = v10 + 1;
    v74 = 0;
  }
  v65 = v10;
  v71 = v4;
  sub_24A3FB0(v4, 2 * v9);
  v52 = *(_DWORD *)(a1 + 56);
  v4 = v71;
  v10 = v65;
  if ( v52 )
  {
    v30 = v75;
    v53 = v52 - 1;
    v54 = *(_QWORD *)(a1 + 40);
    v55 = v53 & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
    v13 = (__int64 *)(v54 + 16LL * v55);
    v56 = *v13;
    if ( v75 == *v13 )
    {
LABEL_74:
      v57 = *(_DWORD *)(a1 + 48);
      v74 = v13;
      v29 = v57 + 1;
    }
    else
    {
      v73 = 1;
      v59 = 0;
      while ( v56 != -4096 )
      {
        if ( !v59 && v56 == -8192 )
          v59 = v13;
        v55 = v53 & (v73 + v55);
        v13 = (__int64 *)(v54 + 16LL * v55);
        v56 = *v13;
        if ( v75 == *v13 )
          goto LABEL_74;
        ++v73;
      }
      if ( !v59 )
        v59 = v13;
      v29 = *(_DWORD *)(a1 + 48) + 1;
      v74 = v59;
      v13 = v59;
    }
  }
  else
  {
    v58 = *(_DWORD *)(a1 + 48);
    v74 = 0;
    v13 = 0;
    v30 = v75;
    v29 = v58 + 1;
  }
LABEL_22:
  *(_DWORD *)(a1 + 48) = v29;
  if ( *v13 != -4096 )
    --*(_DWORD *)(a1 + 52);
  *v13 = v30;
  v62 = v4;
  v13[1] = v76;
  v67 = v13;
  v60 = v10;
  v31 = (_QWORD *)sub_22077B0(0x10u);
  v4 = v62;
  if ( v31 )
  {
    *v31 = v31;
    v31[1] = v60;
  }
  v32 = v67[1];
  v67[1] = (__int64)v31;
  if ( v32 )
  {
    j_j___libc_free_0(v32);
    v4 = v62;
  }
  v9 = *(_DWORD *)(a1 + 56);
  v75 = a3;
  v10 = v28;
  v76 = 0;
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 32);
    v74 = 0;
LABEL_30:
    v63 = v10;
    v9 *= 2;
    goto LABEL_31;
  }
LABEL_4:
  v17 = *(_QWORD *)(a1 + 40);
  v18 = 1;
  v19 = 0;
  v20 = (v9 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v21 = (__int64 *)(v17 + 16LL * v20);
  v22 = *v21;
  if ( a3 == *v21 )
    goto LABEL_5;
  while ( v22 != -4096 )
  {
    if ( v22 != -8192 || v19 )
      v21 = v19;
    v20 = (v9 - 1) & (v18 + v20);
    v22 = *(_QWORD *)(v17 + 16LL * v20);
    if ( a3 == v22 )
      goto LABEL_5;
    ++v18;
    v19 = v21;
    v21 = (__int64 *)(v17 + 16LL * v20);
  }
  if ( !v19 )
    v19 = v21;
  v35 = *(_DWORD *)(a1 + 48);
  ++*(_QWORD *)(a1 + 32);
  v34 = v35 + 1;
  v74 = v19;
  if ( 4 * (v35 + 1) >= 3 * v9 )
    goto LABEL_30;
  v33 = a3;
  if ( v9 - *(_DWORD *)(a1 + 52) - v34 > v9 >> 3 )
    goto LABEL_41;
  v63 = v10;
LABEL_31:
  v68 = v4;
  sub_24A3FB0(v4, v9);
  sub_24A2A40(v68, &v75, &v74);
  v33 = v75;
  v19 = v74;
  v10 = v63;
  v34 = *(_DWORD *)(a1 + 48) + 1;
LABEL_41:
  *(_DWORD *)(a1 + 48) = v34;
  if ( *v19 != -4096 )
    --*(_DWORD *)(a1 + 52);
  *v19 = v33;
  v69 = v10;
  v19[1] = v76;
  v36 = (_QWORD *)sub_22077B0(0x10u);
  if ( v36 )
  {
    *v36 = v36;
    v36[1] = v69;
  }
  v37 = v19[1];
  v19[1] = (__int64)v36;
  if ( v37 )
    j_j___libc_free_0(v37);
LABEL_5:
  v23 = sub_22077B0(0x20u);
  v24 = v23;
  if ( v23 )
  {
    *(_QWORD *)v23 = a2;
    *(_QWORD *)(v23 + 8) = a3;
    *(_QWORD *)(v23 + 16) = a4;
    *(_WORD *)(v23 + 24) = 0;
    *(_BYTE *)(v23 + 26) = 0;
  }
  v25 = *(__int64 **)(a1 + 16);
  if ( v25 != *(__int64 **)(a1 + 24) )
  {
    if ( v25 )
    {
      *v25 = v23;
      v25 = *(__int64 **)(a1 + 16);
    }
    v26 = (__int64)(v25 + 1);
    *(_QWORD *)(a1 + 16) = v25 + 1;
    return *(_QWORD *)(v26 - 8);
  }
  v38 = *(_QWORD *)(a1 + 8);
  v39 = (char *)v25 - v38;
  v40 = (__int64)((__int64)v25 - v38) >> 3;
  if ( v40 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v41 = 1;
  if ( v40 )
    v41 = (__int64)((__int64)v25 - v38) >> 3;
  v42 = __CFADD__(v40, v41);
  v43 = v40 + v41;
  if ( v42 )
  {
    v44 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_55:
    v45 = sub_22077B0(v44);
    v39 = (char *)v25 - v38;
    v46 = v45 + v44;
    v70 = v45;
    v26 = v45 + 8;
    v64 = v46;
    goto LABEL_56;
  }
  if ( v43 )
  {
    if ( v43 > 0xFFFFFFFFFFFFFFFLL )
      v43 = 0xFFFFFFFFFFFFFFFLL;
    v44 = 8 * v43;
    goto LABEL_55;
  }
  v64 = 0;
  v26 = 8;
  v70 = 0;
LABEL_56:
  v47 = (__int64 *)&v39[v70];
  if ( v47 )
    *v47 = v24;
  if ( v25 != (__int64 *)v38 )
  {
    v48 = (_QWORD *)v70;
    v49 = (unsigned __int64 *)v38;
    while ( 1 )
    {
      v51 = *v49;
      if ( v48 )
        break;
      if ( !v51 )
        goto LABEL_61;
      ++v49;
      j_j___libc_free_0(v51);
      v50 = 8;
      if ( v25 == (__int64 *)v49 )
      {
LABEL_66:
        v26 = (__int64)(v48 + 2);
        goto LABEL_67;
      }
LABEL_62:
      v48 = (_QWORD *)v50;
    }
    *v48 = v51;
    *v49 = 0;
LABEL_61:
    ++v49;
    v50 = (__int64)(v48 + 1);
    if ( v25 == (__int64 *)v49 )
      goto LABEL_66;
    goto LABEL_62;
  }
LABEL_67:
  if ( v38 )
  {
    v61 = v26;
    j_j___libc_free_0(v38);
    v26 = v61;
  }
  *(_QWORD *)(a1 + 16) = v26;
  *(_QWORD *)(a1 + 8) = v70;
  *(_QWORD *)(a1 + 24) = v64;
  return *(_QWORD *)(v26 - 8);
}
