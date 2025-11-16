// Function: sub_1AD6830
// Address: 0x1ad6830
//
__int64 __fastcall sub_1AD6830(__int64 a1, __int64 a2)
{
  bool v3; // zf
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r9
  __int64 result; // rax
  int v10; // edx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // r13
  unsigned __int8 v15; // al
  unsigned int v16; // esi
  unsigned int v17; // ecx
  __int64 v18; // rdi
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r8
  __int64 v22; // rax
  unsigned int v23; // edx
  __int64 *v24; // rax
  __int64 v25; // r8
  __int64 j; // r12
  __int64 *v27; // r9
  unsigned int v28; // r10d
  int v29; // r13d
  unsigned int v30; // ecx
  unsigned int v31; // r15d
  __int64 *v32; // rax
  __int64 v33; // r8
  __int64 v34; // r11
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 *v37; // r12
  __int64 *v38; // r15
  __int64 *v39; // rax
  __int64 v40; // rax
  __int64 i; // r13
  _QWORD *v42; // rax
  int v43; // r8d
  int v44; // r9d
  unsigned __int8 v45; // dl
  __int64 v46; // rdx
  int v47; // r10d
  int v48; // ecx
  __int64 v49; // r13
  _QWORD *v50; // rax
  int v51; // r8d
  int v52; // r9d
  unsigned __int8 v53; // dl
  __int64 *v54; // r13
  __int64 *v55; // r9
  int v56; // r11d
  int v57; // ecx
  __int64 v58; // rcx
  int v59; // edx
  int v60; // eax
  int v61; // r9d
  __int64 *v62; // r9
  int v63; // r10d
  int v64; // ecx
  __int64 v65; // [rsp+0h] [rbp-C0h]
  _QWORD *v66; // [rsp+10h] [rbp-B0h]
  int v67; // [rsp+10h] [rbp-B0h]
  _QWORD *v68; // [rsp+10h] [rbp-B0h]
  __int64 v69; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v70; // [rsp+28h] [rbp-98h] BYREF
  __int64 v71; // [rsp+30h] [rbp-90h] BYREF
  __int64 *v72; // [rsp+38h] [rbp-88h] BYREF
  __int64 *v73; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v74; // [rsp+48h] [rbp-78h]
  unsigned int v75; // [rsp+4Ch] [rbp-74h]
  _QWORD v76[14]; // [rsp+50h] [rbp-70h] BYREF

  v3 = *(_BYTE *)(a1 + 16) == 74;
  v69 = a1;
  if ( v3 )
  {
    a1 = *(_QWORD *)(a1 - 24);
    v69 = a1;
  }
  v4 = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a2 + 8);
    v6 = (v4 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a1 == *v7 )
    {
LABEL_5:
      if ( v7 != (__int64 *)(v5 + 16 * v4) )
        return v7[1];
    }
    else
    {
      v10 = 1;
      while ( v8 != -8 )
      {
        v47 = v10 + 1;
        v6 = (v4 - 1) & (v10 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( *v7 == a1 )
          goto LABEL_5;
        v10 = v47;
      }
    }
  }
  result = sub_1AD5E00(a1, a2);
  if ( result )
    return result;
  v11 = sub_1AD65F0(a2, &v69);
  v12 = v69;
  v11[1] = 0;
  v70 = v12;
  v14 = sub_1AD3470(v12);
  v15 = *(_BYTE *)(v14 + 16);
  if ( v15 <= 0x17u )
  {
    v65 = 0;
    v18 = *(_QWORD *)(a2 + 8);
    v16 = *(_DWORD *)(a2 + 24);
    goto LABEL_22;
  }
  while ( v15 == 74 )
  {
LABEL_20:
    v14 = sub_1AD3470(v14);
    v15 = *(_BYTE *)(v14 + 16);
    if ( v15 <= 0x17u )
    {
      v65 = 0;
      v13 = v70;
      v18 = *(_QWORD *)(a2 + 8);
      v16 = *(_DWORD *)(a2 + 24);
      goto LABEL_22;
    }
  }
  v16 = *(_DWORD *)(a2 + 24);
  if ( v16 )
  {
    v17 = v16 - 1;
    v18 = *(_QWORD *)(a2 + 8);
    v19 = (v16 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v20 = (__int64 *)(v18 + 16LL * v19);
    v21 = *v20;
    if ( v14 == *v20 )
    {
LABEL_15:
      if ( v20 != (__int64 *)(v18 + 16LL * v16) )
      {
        v22 = v20[1];
        if ( v22 )
          goto LABEL_82;
        v70 = v14;
        goto LABEL_18;
      }
    }
    else
    {
      v60 = 1;
      while ( v21 != -8 )
      {
        v61 = v60 + 1;
        v19 = v17 & (v60 + v19);
        v20 = (__int64 *)(v18 + 16LL * v19);
        v21 = *v20;
        if ( *v20 == v14 )
          goto LABEL_15;
        v60 = v61;
      }
    }
  }
  v22 = sub_1AD5E00(v14, a2);
  v18 = *(_QWORD *)(a2 + 8);
  v16 = *(_DWORD *)(a2 + 24);
  if ( !v22 )
  {
    v70 = v14;
    if ( !v16 )
    {
      ++*(_QWORD *)a2;
      goto LABEL_86;
    }
    v17 = v16 - 1;
LABEL_18:
    v23 = v17 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v24 = (__int64 *)(v18 + 16LL * v23);
    v25 = *v24;
    if ( v14 == *v24 )
    {
LABEL_19:
      v24[1] = 0;
      goto LABEL_20;
    }
    v62 = 0;
    v63 = 1;
    while ( v25 != -8 )
    {
      if ( v25 == -16 && !v62 )
        v62 = v24;
      v23 = v17 & (v63 + v23);
      v24 = (__int64 *)(v18 + 16LL * v23);
      v25 = *v24;
      if ( v14 == *v24 )
        goto LABEL_19;
      ++v63;
    }
    v64 = *(_DWORD *)(a2 + 16);
    if ( v62 )
      v24 = v62;
    ++*(_QWORD *)a2;
    v59 = v64 + 1;
    if ( 4 * (v64 + 1) < 3 * v16 )
    {
      v58 = v14;
      if ( v16 - (v59 + *(_DWORD *)(a2 + 20)) > v16 >> 3 )
        goto LABEL_88;
      goto LABEL_87;
    }
LABEL_86:
    v16 *= 2;
LABEL_87:
    sub_19566A0(a2, v16);
    sub_1954890(a2, &v70, &v73);
    v24 = v73;
    v58 = v70;
    v59 = *(_DWORD *)(a2 + 16) + 1;
LABEL_88:
    *(_DWORD *)(a2 + 16) = v59;
    if ( *v24 != -8 )
      --*(_DWORD *)(a2 + 20);
    *v24 = v58;
    v24[1] = 0;
    goto LABEL_19;
  }
LABEL_82:
  v65 = v22;
  v13 = v70;
LABEL_22:
  v76[0] = v13;
  LODWORD(j) = 1;
  v27 = v76;
  v73 = v76;
  v75 = 8;
  while ( 2 )
  {
    LODWORD(j) = j - 1;
    v71 = v13;
    v74 = j;
    if ( !v16 )
    {
      ++*(_QWORD *)a2;
      goto LABEL_52;
    }
    v28 = v16 - 1;
    v29 = 1;
    v30 = (v16 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    v31 = v30;
    v32 = (__int64 *)(v18 + 16LL * v30);
    v33 = *v32;
    v34 = *v32;
    if ( v13 != *v32 )
    {
      while ( v34 != -8 )
      {
        v31 = v28 & (v29 + v31);
        v67 = v29 + 1;
        v54 = (__int64 *)(v18 + 16LL * v31);
        v34 = *v54;
        if ( v13 == *v54 )
        {
          if ( v54 == (__int64 *)(v18 + 16LL * v16) )
            goto LABEL_75;
          v32 = (__int64 *)(v18 + 16LL * v31);
          goto LABEL_28;
        }
        v29 = v67;
      }
LABEL_29:
      v30 = v28 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v32 = (__int64 *)(v18 + 16LL * v30);
      v33 = *v32;
      if ( *v32 == v13 )
        goto LABEL_30;
LABEL_75:
      v55 = 0;
      v56 = 1;
      while ( v33 != -8 )
      {
        if ( !v55 && v33 == -16 )
          v55 = v32;
        v30 = v28 & (v56 + v30);
        v32 = (__int64 *)(v18 + 16LL * v30);
        v33 = *v32;
        if ( v13 == *v32 )
          goto LABEL_30;
        ++v56;
      }
      v57 = *(_DWORD *)(a2 + 16);
      if ( v55 )
        v32 = v55;
      ++*(_QWORD *)a2;
      v48 = v57 + 1;
      if ( 4 * v48 < 3 * v16 )
      {
        if ( v16 - (v48 + *(_DWORD *)(a2 + 20)) > v16 >> 3 )
          goto LABEL_54;
        goto LABEL_53;
      }
LABEL_52:
      v16 *= 2;
LABEL_53:
      sub_19566A0(a2, v16);
      sub_1954890(a2, &v71, &v72);
      v32 = v72;
      v13 = v71;
      v48 = *(_DWORD *)(a2 + 16) + 1;
LABEL_54:
      *(_DWORD *)(a2 + 16) = v48;
      if ( *v32 != -8 )
        --*(_DWORD *)(a2 + 20);
      *v32 = v13;
      v13 = v71;
      v32[1] = 0;
      goto LABEL_30;
    }
    if ( v32 == (__int64 *)(v18 + 16LL * v16) )
    {
LABEL_30:
      v32[1] = v65;
      if ( *(_BYTE *)(v13 + 16) == 34 )
      {
        v35 = 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF);
        if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
        {
          v36 = *(_QWORD *)(v13 - 8);
          v37 = (__int64 *)(v36 + v35);
        }
        else
        {
          v37 = (__int64 *)v13;
          v36 = v13 - v35;
        }
        v38 = (__int64 *)(v36 + 24);
        v39 = (__int64 *)(v36 + 48);
        if ( (*(_BYTE *)(v13 + 18) & 1) != 0 )
          v38 = v39;
        for ( ; v37 != v38; v38 += 3 )
        {
          v40 = sub_15A5110(*v38);
          for ( i = *(_QWORD *)(sub_157ED20(v40) + 8); i; i = *(_QWORD *)(i + 8) )
          {
            while ( 1 )
            {
              v42 = sub_1648700(i);
              v45 = *((_BYTE *)v42 + 16);
              if ( v45 > 0x17u && (v45 == 34 || v45 == 73) )
                break;
              i = *(_QWORD *)(i + 8);
              if ( !i )
                goto LABEL_45;
            }
            v46 = v74;
            if ( v74 >= v75 )
            {
              v66 = v42;
              sub_16CD150((__int64)&v73, v76, 0, 8, v43, v44);
              v46 = v74;
              v42 = v66;
            }
            v73[v46] = (__int64)v42;
            ++v74;
          }
LABEL_45:
          ;
        }
        LODWORD(j) = v74;
      }
      else
      {
        v49 = *(_QWORD *)(v13 + 8);
        for ( j = v74; v49; v49 = *(_QWORD *)(v49 + 8) )
        {
          v50 = sub_1648700(v49);
          v53 = *((_BYTE *)v50 + 16);
          if ( v53 > 0x17u && (v53 == 73 || v53 == 34) )
          {
            if ( (unsigned int)j >= v75 )
            {
              v68 = v50;
              sub_16CD150((__int64)&v73, v76, 0, 8, v51, v52);
              j = v74;
              v50 = v68;
            }
            v73[j] = (__int64)v50;
            j = ++v74;
          }
        }
      }
      v27 = v73;
      goto LABEL_23;
    }
LABEL_28:
    if ( !v32[1] )
      goto LABEL_29;
LABEL_23:
    if ( (_DWORD)j )
    {
      v18 = *(_QWORD *)(a2 + 8);
      v16 = *(_DWORD *)(a2 + 24);
      v13 = v27[(unsigned int)j - 1];
      continue;
    }
    break;
  }
  if ( v27 != v76 )
    _libc_free((unsigned __int64)v27);
  return v65;
}
