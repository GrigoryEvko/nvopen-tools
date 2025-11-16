// Function: sub_28CC6E0
// Address: 0x28cc6e0
//
void __fastcall sub_28CC6E0(__int64 a1, __int64 a2)
{
  _BYTE *v2; // r10
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 *v6; // rcx
  __int64 *v7; // rbx
  __int64 *v8; // r15
  __int64 v9; // rcx
  __int64 v10; // r8
  int v11; // eax
  int v12; // r9d
  unsigned int i; // esi
  _QWORD *v14; // rax
  unsigned int v15; // esi
  __int64 v16; // rsi
  __int64 *v17; // r15
  __int64 v18; // rax
  _BYTE *v19; // r10
  __int64 *v20; // r12
  __int64 v21; // r15
  __int64 v22; // r8
  __int64 v23; // r9
  int v24; // eax
  int v25; // r10d
  unsigned int j; // edi
  _QWORD *v27; // rax
  unsigned int v28; // edi
  _BYTE *v29; // r10
  __int64 *v30; // r15
  __int64 v31; // rax
  _BYTE *v32; // r10
  int v33; // esi
  __int64 v34; // r9
  __int64 v35; // rdi
  __int64 v36; // r8
  int v37; // esi
  unsigned int v38; // edx
  __int64 v39; // rax
  _BYTE *v40; // rcx
  int v41; // eax
  int v42; // ebx
  char v43; // al
  __int64 v44; // r8
  bool v45; // zf
  __int64 v46; // rax
  unsigned int v47; // esi
  int v48; // edx
  int v49; // edx
  _BYTE *v50; // rdx
  __int64 v51; // r13
  int v52; // r14d
  __int64 v53; // r15
  __int64 v54; // r8
  int v55; // edi
  unsigned int k; // esi
  _QWORD *v57; // rax
  unsigned int v58; // esi
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // rax
  int v62; // eax
  int v63; // r11d
  __int64 v64; // rax
  __int64 v65; // [rsp+8h] [rbp-78h]
  __int64 v66; // [rsp+10h] [rbp-70h]
  __int64 v67; // [rsp+18h] [rbp-68h]
  __int64 v68; // [rsp+18h] [rbp-68h]
  __int64 v69; // [rsp+20h] [rbp-60h]
  __int64 v70; // [rsp+20h] [rbp-60h]
  bool v71; // [rsp+20h] [rbp-60h]
  __int64 v72; // [rsp+28h] [rbp-58h]
  int v73; // [rsp+30h] [rbp-50h]
  int v74; // [rsp+30h] [rbp-50h]
  __int64 v75; // [rsp+30h] [rbp-50h]
  _BYTE *v76; // [rsp+38h] [rbp-48h] BYREF
  __int64 v77; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v78[14]; // [rsp+48h] [rbp-38h] BYREF

  v2 = (_BYTE *)a2;
  v3 = a1;
  v4 = *(_QWORD *)(a2 + 64);
  v76 = (_BYTE *)a2;
  v66 = v4;
  v5 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v6 = *(__int64 **)(a2 - 8);
    v7 = &v6[v5];
  }
  else
  {
    v7 = (__int64 *)a2;
    v6 = (__int64 *)(a2 - v5 * 8);
  }
  if ( v6 == v7 )
  {
LABEL_15:
    if ( !(unsigned __int8)sub_28CC2D0(a1, v2, *(_QWORD *)(a1 + 1392)) )
      return;
    goto LABEL_45;
  }
  v8 = v6;
  while ( 1 )
  {
    if ( v2 != (_BYTE *)*v8 && *(_QWORD *)(a1 + 1392) != sub_28C7B90(a1, *v8) )
    {
      v9 = *(_QWORD *)(*((_QWORD *)v2 - 1)
                     + 32LL * *((unsigned int *)v2 + 19)
                     + 8LL * (unsigned int)(((__int64)v8 - *((_QWORD *)v2 - 1)) >> 5));
      v73 = *(_DWORD *)(a1 + 2176);
      if ( v73 )
        break;
    }
LABEL_14:
    v8 += 4;
    if ( v7 == v8 )
      goto LABEL_15;
  }
  v69 = *(_QWORD *)(*((_QWORD *)v2 - 1)
                  + 32LL * *((unsigned int *)v2 + 19)
                  + 8LL * (unsigned int)(((__int64)v8 - *((_QWORD *)v2 - 1)) >> 5));
  v10 = *(_QWORD *)(a1 + 2160);
  LODWORD(v77) = ((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4);
  v67 = v10;
  v78[0] = ((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4);
  v11 = sub_28052C0(v78, (unsigned int *)&v77);
  v12 = 1;
  for ( i = (v73 - 1) & v11; ; i = (v73 - 1) & v15 )
  {
    v14 = (_QWORD *)(v67 + 16LL * i);
    if ( v69 == *v14 && v66 == v14[1] )
      break;
    if ( *v14 == -4096 && v14[1] == -4096 )
    {
      v2 = v76;
      goto LABEL_14;
    }
    v15 = v12 + i;
    ++v12;
  }
  v16 = *v8;
  v17 = v8 + 4;
  v18 = sub_28C7B90(a1, v16);
  v19 = v76;
  v65 = *(_QWORD *)(v18 + 48);
  if ( v17 == v7 )
  {
LABEL_31:
    v30 = v7;
LABEL_32:
    v31 = sub_28C7B90(v3, v65);
    v33 = *(_DWORD *)(v3 + 1976);
    v34 = *(_QWORD *)(v3 + 1960);
    v35 = v3 + 1952;
    v36 = v31;
    if ( !v33 )
    {
      v71 = 1;
      v42 = 2;
      goto LABEL_37;
    }
  }
  else
  {
    v20 = v17;
    v21 = a1;
    while ( 1 )
    {
      if ( (_BYTE *)*v20 != v19 && *(_QWORD *)(v21 + 1392) != sub_28C7B90(v21, *v20) )
      {
        v74 = *(_DWORD *)(v21 + 2176);
        v22 = *(_QWORD *)(*((_QWORD *)v19 - 1)
                        + 32LL * *((unsigned int *)v19 + 19)
                        + 8LL * (unsigned int)(((__int64)v20 - *((_QWORD *)v19 - 1)) >> 5));
        if ( v74 )
          break;
      }
LABEL_51:
      v20 += 4;
      if ( v20 == v7 )
      {
        v3 = v21;
        goto LABEL_31;
      }
    }
    v23 = *(_QWORD *)(v21 + 2160);
    v70 = *(_QWORD *)(*((_QWORD *)v19 - 1)
                    + 32LL * *((unsigned int *)v19 + 19)
                    + 8LL * (unsigned int)(((__int64)v20 - *((_QWORD *)v19 - 1)) >> 5));
    LODWORD(v77) = ((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4);
    v68 = v23;
    v78[0] = ((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4);
    v24 = sub_28052C0(v78, (unsigned int *)&v77);
    v25 = 1;
    for ( j = (v74 - 1) & v24; ; j = (v74 - 1) & v28 )
    {
      v27 = (_QWORD *)(v68 + 16LL * j);
      if ( v70 == *v27 && v66 == v27[1] )
        break;
      if ( *v27 == -4096 && v27[1] == -4096 )
      {
        v19 = v76;
        goto LABEL_51;
      }
      v28 = v25 + j;
      ++v25;
    }
    while ( 1 )
    {
      if ( v20 == v7 )
      {
        v64 = v21;
        v30 = v20;
        v3 = v64;
        goto LABEL_32;
      }
      if ( v65 != *(_QWORD *)(sub_28C7B90(v21, *v20) + 48) )
        break;
      v20 += 4;
      v29 = v76;
      if ( v20 == v7 )
      {
        v3 = v21;
        goto LABEL_31;
      }
      v51 = v21;
      while ( 1 )
      {
        if ( (_BYTE *)*v20 != v29 && *(_QWORD *)(v51 + 1392) != sub_28C7B90(v51, *v20) )
        {
          v52 = *(_DWORD *)(v51 + 2176);
          v53 = *(_QWORD *)(*((_QWORD *)v29 - 1)
                          + 32LL * *((unsigned int *)v29 + 19)
                          + 8LL * (unsigned int)(((__int64)v20 - *((_QWORD *)v29 - 1)) >> 5));
          if ( v52 )
            break;
        }
LABEL_29:
        v20 += 4;
        if ( v20 == v7 )
        {
          v3 = v51;
          goto LABEL_31;
        }
      }
      v54 = *(_QWORD *)(v51 + 2160);
      LODWORD(v77) = ((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4);
      v75 = v54;
      v78[0] = ((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4);
      v55 = 1;
      for ( k = (v52 - 1) & sub_28052C0(v78, (unsigned int *)&v77); ; k = (v52 - 1) & v58 )
      {
        v57 = (_QWORD *)(v75 + 16LL * k);
        if ( v53 == *v57 && v66 == v57[1] )
          break;
        if ( *v57 == -4096 && v57[1] == -4096 )
        {
          v29 = v76;
          goto LABEL_29;
        }
        v58 = v55 + k;
        ++v55;
      }
      v21 = v51;
    }
    v59 = v21;
    v30 = v20;
    v3 = v59;
    v60 = sub_28C7B90(v59, (__int64)v76);
    v32 = *(_BYTE **)(v60 + 48);
    v36 = v60;
    if ( v76 != v32 )
    {
      v61 = sub_28CC470(v3, 0, 0);
      v32 = v76;
      v36 = v61;
      *(_QWORD *)(v61 + 48) = v76;
    }
    v33 = *(_DWORD *)(v3 + 1976);
    v34 = *(_QWORD *)(v3 + 1960);
    v35 = v3 + 1952;
    if ( !v33 )
    {
      v71 = 1;
      v42 = 3;
      goto LABEL_37;
    }
  }
  v37 = v33 - 1;
  v38 = v37 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
  v39 = v34 + 16LL * v38;
  v40 = *(_BYTE **)v39;
  if ( *(_BYTE **)v39 == v32 )
  {
LABEL_34:
    v41 = *(_DWORD *)(v39 + 8);
  }
  else
  {
    v62 = 1;
    while ( v40 != (_BYTE *)-4096LL )
    {
      v63 = v62 + 1;
      v38 = v37 & (v62 + v38);
      v39 = v34 + 16LL * v38;
      v40 = *(_BYTE **)v39;
      if ( *(_BYTE **)v39 == v32 )
        goto LABEL_34;
      v62 = v63;
    }
    v41 = 0;
  }
  if ( v30 == v7 )
  {
    v42 = 2;
    v71 = v41 != 2;
  }
  else
  {
    v42 = 3;
    v71 = v41 != 3;
  }
LABEL_37:
  v72 = v36;
  v43 = sub_28C7770(v35, (__int64 *)&v76, &v77);
  v44 = v72;
  v45 = v43 == 0;
  v46 = v77;
  if ( !v45 )
    goto LABEL_43;
  v47 = *(_DWORD *)(v3 + 1976);
  v48 = *(_DWORD *)(v3 + 1968);
  *(_QWORD *)v78 = v77;
  ++*(_QWORD *)(v3 + 1952);
  v49 = v48 + 1;
  if ( 4 * v49 >= 3 * v47 )
  {
    v47 *= 2;
LABEL_77:
    sub_28C9E70(v35, v47);
    sub_28C7770(v35, (__int64 *)&v76, v78);
    v44 = v72;
    v49 = *(_DWORD *)(v3 + 1968) + 1;
    v46 = *(_QWORD *)v78;
    goto LABEL_40;
  }
  if ( v47 - *(_DWORD *)(v3 + 1972) - v49 <= v47 >> 3 )
    goto LABEL_77;
LABEL_40:
  *(_DWORD *)(v3 + 1968) = v49;
  if ( *(_QWORD *)v46 != -4096 )
    --*(_DWORD *)(v3 + 1972);
  v50 = v76;
  *(_DWORD *)(v46 + 8) = 0;
  *(_QWORD *)v46 = v50;
LABEL_43:
  *(_DWORD *)(v46 + 8) = v42;
  if ( (unsigned __int8)sub_28CC2D0(v3, v76, v44) || v71 )
LABEL_45:
    sub_28CA760(v3, (__int64)v76);
}
