// Function: sub_1CDB400
// Address: 0x1cdb400
//
__int64 __fastcall sub_1CDB400(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rcx
  int v5; // r10d
  __int64 v6; // rdi
  int v7; // r11d
  unsigned int v8; // edx
  unsigned int v9; // r12d
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r13
  __int64 v15; // rax
  __int64 *v16; // r8
  __int64 *v17; // r15
  __int64 v18; // rdx
  __int64 *v19; // r10
  int v20; // r11d
  unsigned int v21; // eax
  __int64 *v22; // rdi
  __int64 v23; // rcx
  __int64 v24; // rax
  int v25; // ecx
  __int64 *v26; // r12
  unsigned __int64 v27; // rax
  __int64 v28; // r12
  int v29; // eax
  unsigned int v30; // r14d
  __int64 v31; // rax
  __int64 v32; // rdx
  char v33; // r8
  __int64 *v34; // rax
  int v35; // edi
  unsigned int v36; // esi
  int v37; // edx
  __int64 v38; // rdx
  int v39; // r13d
  __int64 v40; // r11
  __int64 *v41; // r11
  int v42; // r12d
  __int64 *v43; // r9
  int v44; // eax
  int v45; // edx
  __int64 *v46; // r11
  __int64 v47; // [rsp+10h] [rbp-100h]
  int v48; // [rsp+24h] [rbp-ECh]
  __int64 v49; // [rsp+28h] [rbp-E8h] BYREF
  __int64 v50; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v51; // [rsp+38h] [rbp-D8h] BYREF
  __int64 v52; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v53; // [rsp+48h] [rbp-C8h]
  __int64 v54; // [rsp+50h] [rbp-C0h]
  __int64 v55; // [rsp+58h] [rbp-B8h]
  _QWORD v56[6]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v57[2]; // [rsp+90h] [rbp-80h] BYREF
  __int64 *v58; // [rsp+A0h] [rbp-70h]
  __int64 v59; // [rsp+A8h] [rbp-68h]
  __int64 v60; // [rsp+B0h] [rbp-60h]
  __int64 v61; // [rsp+B8h] [rbp-58h]
  __int64 *v62; // [rsp+C0h] [rbp-50h]
  __int64 *v63; // [rsp+C8h] [rbp-48h]
  __int64 v64; // [rsp+D0h] [rbp-40h]
  __int64 **v65; // [rsp+D8h] [rbp-38h]

  v49 = a2;
  v3 = *(unsigned int *)(a1 + 240);
  v47 = a1 + 216;
  if ( !(_DWORD)v3 )
    goto LABEL_8;
  v4 = v49;
  v5 = v3 - 1;
  v6 = *(_QWORD *)(a1 + 224);
  v7 = 1;
  v8 = (v3 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
  v9 = v8;
  v10 = (__int64 *)(v6 + 16LL * v8);
  v11 = *v10;
  v12 = *v10;
  if ( v49 == *v10 )
  {
    if ( v10 != (__int64 *)(v6 + 16 * v3) )
      return v10[1];
LABEL_8:
    v52 = 0;
    v53 = 0;
    v54 = 0;
    v55 = 0;
    v15 = sub_22077B0(32);
    v13 = v15;
    if ( v15 )
    {
      *(_QWORD *)v15 = 0;
      *(_QWORD *)(v15 + 8) = 0;
      *(_QWORD *)(v15 + 16) = 0;
      *(_DWORD *)(v15 + 24) = 0;
    }
    v57[0] = 0;
    v57[1] = 0;
    v50 = v49;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    v61 = 0;
    v62 = 0;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    sub_1C08D60(v57, 0);
    v16 = v62;
    if ( v62 == (__int64 *)(v64 - 8) )
    {
      sub_1B4ECC0(v57, &v50);
      v17 = v62;
    }
    else
    {
      if ( v62 )
      {
        *v62 = v50;
        v16 = v62;
      }
      v17 = v16 + 1;
      v62 = v16 + 1;
    }
    if ( v17 == v58 )
    {
LABEL_46:
      sub_1C08CE0(v57);
      v33 = sub_1CD3720(v47, &v49, v57);
      v34 = (__int64 *)v57[0];
      if ( v33 )
      {
LABEL_52:
        v34[1] = v13;
        j___libc_free_0(v53);
        return v13;
      }
      v35 = *(_DWORD *)(a1 + 232);
      v36 = *(_DWORD *)(a1 + 240);
      ++*(_QWORD *)(a1 + 216);
      v37 = v35 + 1;
      if ( 4 * (v35 + 1) >= 3 * v36 )
      {
        v36 *= 2;
      }
      else if ( v36 - *(_DWORD *)(a1 + 236) - v37 > v36 >> 3 )
      {
LABEL_49:
        *(_DWORD *)(a1 + 232) = v37;
        if ( *v34 != -8 )
          --*(_DWORD *)(a1 + 236);
        v38 = v49;
        v34[1] = 0;
        *v34 = v38;
        goto LABEL_52;
      }
      sub_1CDB270(v47, v36);
      sub_1CD3720(v47, &v49, v57);
      v34 = (__int64 *)v57[0];
      v37 = *(_DWORD *)(a1 + 232) + 1;
      goto LABEL_49;
    }
    while ( 1 )
    {
      if ( v63 == v17 )
      {
        v51 = (*(v65 - 1))[63];
        j_j___libc_free_0(v17, 512);
        v32 = (__int64)(*--v65 + 64);
        v63 = *v65;
        v64 = v32;
        v62 = v63 + 63;
      }
      else
      {
        v24 = *(v17 - 1);
        v62 = v17 - 1;
        v51 = v24;
      }
      if ( !(_DWORD)v55 )
        break;
      v18 = v51;
      v19 = 0;
      v20 = 1;
      v21 = (v55 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
      v22 = (__int64 *)(v53 + 8LL * v21);
      v23 = *v22;
      if ( v51 == *v22 )
        goto LABEL_17;
      while ( v23 != -8 )
      {
        if ( v19 || v23 != -16 )
          v22 = v19;
        v21 = (v55 - 1) & (v20 + v21);
        v26 = (__int64 *)(v53 + 8LL * v21);
        v23 = *v26;
        if ( v51 == *v26 )
          goto LABEL_17;
        ++v20;
        v19 = v22;
        v22 = (__int64 *)(v53 + 8LL * v21);
      }
      if ( !v19 )
        v19 = v22;
      ++v52;
      v25 = v54 + 1;
      if ( 4 * ((int)v54 + 1) >= (unsigned int)(3 * v55) )
        goto LABEL_23;
      if ( (int)v55 - HIDWORD(v54) - v25 <= (unsigned int)v55 >> 3 )
      {
        sub_13B3D40((__int64)&v52, v55);
        goto LABEL_24;
      }
LABEL_34:
      LODWORD(v54) = v25;
      if ( *v19 != -8 )
        --HIDWORD(v54);
      *v19 = v18;
      sub_1C0B2E0((__int64)v56, v13, &v51);
      v27 = sub_157EBA0(v51);
      v28 = v27;
      if ( v27 )
      {
        v29 = sub_15F3BE0(v27);
        v17 = v62;
        v48 = v29;
        if ( v29 )
        {
          v30 = 0;
          do
          {
            v31 = sub_15F3BF0(v28, v30);
            v56[0] = v31;
            if ( v17 == (__int64 *)(v64 - 8) )
            {
              sub_1B4ECC0(v57, v56);
              v17 = v62;
            }
            else
            {
              if ( v17 )
              {
                *v17 = v31;
                v17 = v62;
              }
              v62 = ++v17;
            }
            ++v30;
          }
          while ( v48 != v30 );
        }
        goto LABEL_18;
      }
LABEL_17:
      v17 = v62;
LABEL_18:
      if ( v58 == v17 )
        goto LABEL_46;
    }
    ++v52;
LABEL_23:
    sub_13B3D40((__int64)&v52, 2 * v55);
LABEL_24:
    sub_1898220((__int64)&v52, &v51, v56);
    v19 = (__int64 *)v56[0];
    v18 = v51;
    v25 = v54 + 1;
    goto LABEL_34;
  }
  while ( 1 )
  {
    if ( v12 == -8 )
      goto LABEL_8;
    v39 = v7 + 1;
    v40 = v5 & (v9 + v7);
    v9 = v40;
    v41 = (__int64 *)(v6 + 16 * v40);
    v12 = *v41;
    if ( v49 == *v41 )
      break;
    v7 = v39;
  }
  if ( v41 == (__int64 *)(v6 + 16LL * (unsigned int)v3) )
    goto LABEL_8;
  v42 = 1;
  v43 = 0;
  while ( v11 != -8 )
  {
    if ( v11 != -16 || v43 )
      v10 = v43;
    v8 = v5 & (v42 + v8);
    v46 = (__int64 *)(v6 + 16LL * v8);
    v11 = *v46;
    if ( v49 == *v46 )
      return v46[1];
    ++v42;
    v43 = v10;
    v10 = (__int64 *)(v6 + 16LL * v8);
  }
  if ( !v43 )
    v43 = v10;
  v44 = *(_DWORD *)(a1 + 232);
  ++*(_QWORD *)(a1 + 216);
  v45 = v44 + 1;
  if ( 4 * (v44 + 1) >= (unsigned int)(3 * v3) )
  {
    LODWORD(v3) = 2 * v3;
    goto LABEL_65;
  }
  if ( (int)v3 - *(_DWORD *)(a1 + 236) - v45 <= (unsigned int)v3 >> 3 )
  {
LABEL_65:
    sub_1CDB270(v47, v3);
    sub_1CD3720(v47, &v49, v57);
    v43 = (__int64 *)v57[0];
    v4 = v49;
    v45 = *(_DWORD *)(a1 + 232) + 1;
  }
  *(_DWORD *)(a1 + 232) = v45;
  if ( *v43 != -8 )
    --*(_DWORD *)(a1 + 236);
  *v43 = v4;
  v13 = 0;
  v43[1] = 0;
  return v13;
}
