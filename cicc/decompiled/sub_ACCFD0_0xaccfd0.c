// Function: sub_ACCFD0
// Address: 0xaccfd0
//
__int64 __fastcall sub_ACCFD0(__int64 *a1, __int64 a2)
{
  unsigned int v3; // r15d
  int v4; // eax
  __int64 v5; // r14
  bool v6; // al
  unsigned int v7; // esi
  __int64 v8; // rcx
  unsigned int *v9; // r9
  int v10; // r11d
  unsigned int v11; // edx
  _DWORD *v12; // rbx
  unsigned int v13; // eax
  __int64 *v14; // rbx
  __int64 result; // rax
  unsigned int v16; // esi
  __int64 v17; // rcx
  int v18; // r11d
  unsigned int v19; // edx
  unsigned int v20; // eax
  int v21; // eax
  int v22; // edx
  __int64 v23; // r12
  __int64 v24; // r12
  __int64 v25; // rdi
  __int64 v26; // rbx
  __int64 v27; // r15
  int v28; // eax
  int v29; // edx
  int v30; // eax
  int v31; // eax
  int v32; // edx
  __int64 v33; // rbx
  unsigned __int64 v34; // rax
  unsigned int v35; // eax
  __int64 v36; // rax
  __int64 v37; // r8
  __int64 v38; // r15
  unsigned __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 j; // rdx
  __int64 v43; // rdx
  __int64 i; // rdx
  __int64 v45; // [rsp+8h] [rbp-48h]
  __int64 v46; // [rsp+8h] [rbp-48h]
  __int64 v47; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v48[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(_DWORD *)(a2 + 8);
  if ( v3 > 0x40 )
  {
    v4 = sub_C444A0(a2);
    v5 = *a1;
    if ( v3 != v4 )
    {
      v6 = v3 - 1 == (unsigned int)sub_C444A0(a2);
      goto LABEL_4;
    }
LABEL_10:
    LODWORD(v47) = v3;
    v16 = *(_DWORD *)(v5 + 232);
    if ( v16 )
    {
      v17 = *(_QWORD *)(v5 + 216);
      v9 = 0;
      v18 = 1;
      v19 = (v16 - 1) & (37 * v3);
      v12 = (_DWORD *)(v17 + 16LL * v19);
      v20 = *v12;
      if ( v3 == *v12 )
        goto LABEL_7;
      while ( v20 != -1 )
      {
        if ( v20 == -2 && !v9 )
          v9 = v12;
        v19 = (v16 - 1) & (v18 + v19);
        v12 = (_DWORD *)(v17 + 16LL * v19);
        v20 = *v12;
        if ( v3 == *v12 )
          goto LABEL_7;
        ++v18;
      }
      if ( !v9 )
        v9 = v12;
      v48[0] = v9;
      v21 = *(_DWORD *)(v5 + 224);
      ++*(_QWORD *)(v5 + 208);
      v22 = v21 + 1;
      if ( 4 * (v21 + 1) < 3 * v16 )
      {
        if ( v16 - *(_DWORD *)(v5 + 228) - v22 > v16 >> 3 )
          goto LABEL_17;
        goto LABEL_56;
      }
    }
    else
    {
      v48[0] = 0;
      ++*(_QWORD *)(v5 + 208);
    }
    v16 *= 2;
LABEL_56:
    sub_AC8740(v5 + 208, v16);
    sub_AC6440(v5 + 208, (int *)&v47, v48);
    v3 = v47;
    v9 = (unsigned int *)v48[0];
    v22 = *(_DWORD *)(v5 + 224) + 1;
LABEL_17:
    *(_DWORD *)(v5 + 224) = v22;
    if ( *v9 != -1 )
      --*(_DWORD *)(v5 + 228);
    goto LABEL_49;
  }
  v5 = *a1;
  if ( !*(_QWORD *)a2 )
    goto LABEL_10;
  v6 = *(_QWORD *)a2 == 1;
LABEL_4:
  if ( v6 )
  {
    LODWORD(v47) = v3;
    v7 = *(_DWORD *)(v5 + 264);
    if ( v7 )
    {
      v8 = *(_QWORD *)(v5 + 248);
      v9 = 0;
      v10 = 1;
      v11 = (v7 - 1) & (37 * v3);
      v12 = (_DWORD *)(v8 + 16LL * v11);
      v13 = *v12;
      if ( v3 == *v12 )
      {
LABEL_7:
        v14 = (__int64 *)(v12 + 2);
        goto LABEL_8;
      }
      while ( v13 != -1 )
      {
        if ( !v9 && v13 == -2 )
          v9 = v12;
        v11 = (v7 - 1) & (v10 + v11);
        v12 = (_DWORD *)(v8 + 16LL * v11);
        v13 = *v12;
        if ( v3 == *v12 )
          goto LABEL_7;
        ++v10;
      }
      if ( !v9 )
        v9 = v12;
      v48[0] = v9;
      v31 = *(_DWORD *)(v5 + 256);
      ++*(_QWORD *)(v5 + 240);
      v32 = v31 + 1;
      if ( 4 * (v31 + 1) < 3 * v7 )
      {
        if ( v7 - *(_DWORD *)(v5 + 260) - v32 > v7 >> 3 )
          goto LABEL_47;
        goto LABEL_53;
      }
    }
    else
    {
      v48[0] = 0;
      ++*(_QWORD *)(v5 + 240);
    }
    v7 *= 2;
LABEL_53:
    sub_AC8740(v5 + 240, v7);
    sub_AC6440(v5 + 240, (int *)&v47, v48);
    v3 = v47;
    v9 = (unsigned int *)v48[0];
    v32 = *(_DWORD *)(v5 + 256) + 1;
LABEL_47:
    *(_DWORD *)(v5 + 256) = v32;
    if ( *v9 != -1 )
      --*(_DWORD *)(v5 + 260);
LABEL_49:
    *v9 = v3;
    v14 = (__int64 *)(v9 + 2);
    *((_QWORD *)v9 + 1) = 0;
    goto LABEL_8;
  }
  if ( (unsigned __int8)sub_AC64E0(v5 + 272, a2, &v47) )
  {
    v14 = (__int64 *)(v47 + 16);
    result = *(_QWORD *)(v47 + 16);
    if ( result )
      return result;
    goto LABEL_23;
  }
  v26 = v47;
  v48[0] = v47;
  v27 = *(unsigned int *)(v5 + 296);
  v28 = *(_DWORD *)(v5 + 288);
  ++*(_QWORD *)(v5 + 272);
  v30 = v28 + 1;
  if ( 4 * v30 >= (unsigned int)(3 * v27) )
  {
    v33 = *(_QWORD *)(v5 + 280);
    v29 = 2 * v27;
    v34 = (((((((((unsigned int)(v29 - 1) | ((unsigned __int64)(unsigned int)(v29 - 1) >> 1)) >> 2)
              | (unsigned int)(v29 - 1)
              | ((unsigned __int64)(unsigned int)(v29 - 1) >> 1)) >> 4)
            | (((unsigned int)(v29 - 1) | ((unsigned __int64)(unsigned int)(v29 - 1) >> 1)) >> 2)
            | (unsigned int)(v29 - 1)
            | ((unsigned __int64)(unsigned int)(v29 - 1) >> 1)) >> 8)
          | (((((unsigned int)(v29 - 1) | ((unsigned __int64)(unsigned int)(v29 - 1) >> 1)) >> 2)
            | (unsigned int)(v29 - 1)
            | ((unsigned __int64)(unsigned int)(v29 - 1) >> 1)) >> 4)
          | (((unsigned int)(v29 - 1) | ((unsigned __int64)(unsigned int)(v29 - 1) >> 1)) >> 2)
          | (unsigned int)(v29 - 1)
          | ((unsigned __int64)(unsigned int)(v29 - 1) >> 1)) >> 16)
        | (((((((unsigned int)(v29 - 1) | ((unsigned __int64)(unsigned int)(v29 - 1) >> 1)) >> 2)
            | (unsigned int)(v29 - 1)
            | ((unsigned __int64)(unsigned int)(v29 - 1) >> 1)) >> 4)
          | (((unsigned int)(v29 - 1) | ((unsigned __int64)(unsigned int)(v29 - 1) >> 1)) >> 2)
          | (unsigned int)(v29 - 1)
          | ((unsigned __int64)(unsigned int)(v29 - 1) >> 1)) >> 8)
        | (((((unsigned int)(v29 - 1) | ((unsigned __int64)(unsigned int)(v29 - 1) >> 1)) >> 2)
          | (unsigned int)(v29 - 1)
          | ((unsigned __int64)(unsigned int)(v29 - 1) >> 1)) >> 4)
        | (((unsigned int)(v29 - 1) | ((unsigned __int64)(unsigned int)(v29 - 1) >> 1)) >> 2)
        | (unsigned int)(v29 - 1)
        | ((unsigned __int64)(unsigned int)(v29 - 1) >> 1);
    v35 = v34 + 1;
    if ( v35 < 0x40 )
      v35 = 64;
    *(_DWORD *)(v5 + 296) = v35;
    v36 = sub_C7D670(24LL * v35, 8);
    v37 = v5 + 272;
    *(_QWORD *)(v5 + 280) = v36;
    if ( !v33 )
    {
      v43 = *(unsigned int *)(v5 + 296);
      *(_QWORD *)(v5 + 288) = 0;
      for ( i = v36 + 24 * v43; i != v36; v36 += 24 )
      {
        if ( v36 )
        {
          *(_DWORD *)(v36 + 8) = 0;
          *(_QWORD *)v36 = -1;
        }
      }
      goto LABEL_63;
    }
  }
  else
  {
    if ( (int)v27 - *(_DWORD *)(v5 + 292) - v30 > (unsigned int)v27 >> 3 )
      goto LABEL_32;
    v33 = *(_QWORD *)(v5 + 280);
    v39 = ((((((((((unsigned int)(v27 - 1) | ((unsigned __int64)(unsigned int)(v27 - 1) >> 1)) >> 2)
               | (unsigned int)(v27 - 1)
               | ((unsigned __int64)(unsigned int)(v27 - 1) >> 1)) >> 4)
             | (((unsigned int)(v27 - 1) | ((unsigned __int64)(unsigned int)(v27 - 1) >> 1)) >> 2)
             | (unsigned int)(v27 - 1)
             | ((unsigned __int64)(unsigned int)(v27 - 1) >> 1)) >> 8)
           | (((((unsigned int)(v27 - 1) | ((unsigned __int64)(unsigned int)(v27 - 1) >> 1)) >> 2)
             | (unsigned int)(v27 - 1)
             | ((unsigned __int64)(unsigned int)(v27 - 1) >> 1)) >> 4)
           | (((unsigned int)(v27 - 1) | ((unsigned __int64)(unsigned int)(v27 - 1) >> 1)) >> 2)
           | (unsigned int)(v27 - 1)
           | ((unsigned __int64)(unsigned int)(v27 - 1) >> 1)) >> 16)
         | (((((((unsigned int)(v27 - 1) | ((unsigned __int64)(unsigned int)(v27 - 1) >> 1)) >> 2)
             | (unsigned int)(v27 - 1)
             | ((unsigned __int64)(unsigned int)(v27 - 1) >> 1)) >> 4)
           | (((unsigned int)(v27 - 1) | ((unsigned __int64)(unsigned int)(v27 - 1) >> 1)) >> 2)
           | (unsigned int)(v27 - 1)
           | ((unsigned __int64)(unsigned int)(v27 - 1) >> 1)) >> 8)
         | (((((unsigned int)(v27 - 1) | ((unsigned __int64)(unsigned int)(v27 - 1) >> 1)) >> 2)
           | (unsigned int)(v27 - 1)
           | ((unsigned __int64)(unsigned int)(v27 - 1) >> 1)) >> 4)
         | (((unsigned int)(v27 - 1) | ((unsigned __int64)(unsigned int)(v27 - 1) >> 1)) >> 2)
         | (unsigned int)(v27 - 1)
         | ((unsigned __int64)(unsigned int)(v27 - 1) >> 1))
        + 1;
    if ( (unsigned int)v39 < 0x40 )
      LODWORD(v39) = 64;
    *(_DWORD *)(v5 + 296) = v39;
    v40 = sub_C7D670(24LL * (unsigned int)v39, 8);
    v37 = v5 + 272;
    *(_QWORD *)(v5 + 280) = v40;
    if ( !v33 )
    {
      v41 = *(unsigned int *)(v5 + 296);
      *(_QWORD *)(v5 + 288) = 0;
      for ( j = v40 + 24 * v41; j != v40; v40 += 24 )
      {
        if ( v40 )
        {
          *(_DWORD *)(v40 + 8) = 0;
          *(_QWORD *)v40 = -1;
        }
      }
      goto LABEL_63;
    }
  }
  v46 = v37;
  v38 = 24 * v27;
  sub_ACCE40(v37, v33, v33 + v38);
  sub_C7D6A0(v33, v38, 8);
  v37 = v46;
LABEL_63:
  sub_AC64E0(v37, a2, v48);
  v26 = v48[0];
  v30 = *(_DWORD *)(v5 + 288) + 1;
LABEL_32:
  *(_DWORD *)(v5 + 288) = v30;
  if ( (!*(_DWORD *)(v26 + 8) && *(_QWORD *)v26 == -1 || (--*(_DWORD *)(v5 + 292), *(_DWORD *)(v26 + 8) <= 0x40u))
    && *(_DWORD *)(a2 + 8) <= 0x40u )
  {
    *(_QWORD *)v26 = *(_QWORD *)a2;
    *(_DWORD *)(v26 + 8) = *(_DWORD *)(a2 + 8);
  }
  else
  {
    sub_C43990(v26, a2);
  }
  *(_QWORD *)(v26 + 16) = 0;
  v14 = (__int64 *)(v26 + 16);
LABEL_8:
  result = *v14;
  if ( *v14 )
    return result;
LABEL_23:
  v23 = sub_BCCE00(a1, *(unsigned int *)(a2 + 8));
  result = sub_BD2C40(40, unk_3F289A4);
  if ( result )
  {
    v45 = result;
    sub_AC2FF0(result, v23, a2);
    result = v45;
  }
  v24 = *v14;
  *v14 = result;
  if ( v24 )
  {
    if ( *(_DWORD *)(v24 + 32) > 0x40u )
    {
      v25 = *(_QWORD *)(v24 + 24);
      if ( v25 )
        j_j___libc_free_0_0(v25);
    }
    sub_BD7260(v24);
    sub_BD2DD0(v24);
    return *v14;
  }
  return result;
}
