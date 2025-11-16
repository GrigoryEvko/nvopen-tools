// Function: sub_2B4A4D0
// Address: 0x2b4a4d0
//
__int64 __fastcall sub_2B4A4D0(__int64 *a1, unsigned int a2, unsigned int a3, unsigned int a4, unsigned __int64 *a5)
{
  __int64 v5; // r14
  __int64 v6; // rsi
  _BYTE *v7; // rbx
  __int64 *v8; // rax
  __int64 v9; // rax
  __int64 v10; // r11
  char v11; // si
  __int64 v12; // rbx
  __int64 v13; // r15
  __int64 *v14; // rdi
  int v15; // edx
  unsigned int v16; // ecx
  __int64 *v17; // rax
  _BYTE *v18; // r10
  int v19; // r13d
  char v20; // r8
  _BYTE *v21; // r12
  unsigned int v22; // edx
  __int64 result; // rax
  unsigned int v24; // eax
  __int64 *v25; // r9
  unsigned int v26; // ecx
  unsigned int v27; // edi
  unsigned int v28; // r9d
  __int64 v29; // rax
  __int64 *v30; // rcx
  int v31; // esi
  unsigned int v32; // eax
  __int64 *v33; // rdx
  __int64 v34; // rdi
  __int64 v35; // rbx
  __int64 *v36; // rsi
  int v37; // r11d
  unsigned int v38; // r10d
  unsigned int v39; // edi
  __int64 *v40; // rax
  __int64 v41; // r13
  unsigned int v42; // ecx
  unsigned int v43; // edi
  unsigned int v44; // eax
  unsigned int v45; // eax
  unsigned int v46; // ecx
  int v47; // edx
  unsigned int v48; // eax
  __int64 *v49; // rsi
  int v50; // ecx
  __int64 v51; // rdx
  __int64 v52; // rdi
  int v53; // r10d
  __int64 *v54; // r8
  __int64 *v55; // rsi
  int v56; // ecx
  int v57; // r10d
  __int64 v58; // rdx
  __int64 v59; // rdi
  __int64 v60; // rax
  int v61; // edx
  int v62; // eax
  int v63; // r12d
  int v64; // r10d
  int v65; // [rsp+14h] [rbp-BCh]
  _BYTE *v67; // [rsp+28h] [rbp-A8h]
  __int64 v69; // [rsp+40h] [rbp-90h]
  unsigned int v70; // [rsp+48h] [rbp-88h]
  __int64 v71; // [rsp+50h] [rbp-80h] BYREF
  __int64 v72; // [rsp+58h] [rbp-78h]
  __int64 *v73; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v74; // [rsp+68h] [rbp-68h]
  char v75; // [rsp+A0h] [rbp-30h] BYREF

  v5 = a2;
  v6 = *a1;
  v7 = *(_BYTE **)(*(_QWORD *)(*a1 + 48LL * a4) + 16 * v5);
  v67 = v7;
  if ( *v7 <= 0x1Cu )
    return 0;
  v69 = 48LL * a3;
  if ( *(_BYTE **)(*(_QWORD *)(v6 + v69) + 16 * v5) == v7 || *v7 == 90 )
    return 0;
  v8 = (__int64 *)&v73;
  v71 = 0;
  v72 = 1;
  do
  {
    *v8 = -4096;
    v8 += 2;
  }
  while ( v8 != (__int64 *)&v75 );
  v9 = *(unsigned int *)(v6 + 8);
  v10 = *(_QWORD *)(v6 + 48LL * a3);
  v11 = v72;
  if ( v9 )
  {
    v12 = 0;
    v13 = v9;
    while ( 1 )
    {
      v19 = v12;
      v20 = v11 & 1;
      if ( (_DWORD)v5 == (_DWORD)v12 )
        goto LABEL_10;
      v21 = *(_BYTE **)(v10 + 16LL * (unsigned int)v12);
      if ( *v21 <= 0x1Cu )
      {
        result = 0;
        goto LABEL_44;
      }
      if ( v20 )
      {
        v14 = (__int64 *)&v73;
        v15 = 3;
      }
      else
      {
        v22 = v74;
        v14 = v73;
        if ( !v74 )
        {
          v24 = v72;
          ++v71;
          v25 = 0;
          v26 = ((unsigned int)v72 >> 1) + 1;
LABEL_19:
          v27 = 3 * v22;
          goto LABEL_20;
        }
        v15 = v74 - 1;
      }
      v16 = v15 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v17 = &v14[2 * v16];
      v18 = (_BYTE *)*v17;
      if ( v21 == (_BYTE *)*v17 )
      {
LABEL_10:
        if ( v13 == ++v12 )
          goto LABEL_25;
      }
      else
      {
        v65 = 1;
        v25 = 0;
        while ( v18 != (_BYTE *)-4096LL )
        {
          if ( v25 || v18 != (_BYTE *)-8192LL )
            v17 = v25;
          v16 = v15 & (v65 + v16);
          v18 = (_BYTE *)v14[2 * v16];
          if ( v21 == v18 )
            goto LABEL_10;
          ++v65;
          v25 = v17;
          v17 = &v14[2 * v16];
        }
        if ( !v25 )
          v25 = v17;
        v24 = v72;
        ++v71;
        v26 = ((unsigned int)v72 >> 1) + 1;
        if ( !v20 )
        {
          v22 = v74;
          goto LABEL_19;
        }
        v27 = 12;
        v22 = 4;
LABEL_20:
        if ( 4 * v26 >= v27 )
        {
          sub_BB64D0((__int64)&v71, 2 * v22);
          if ( (v72 & 1) != 0 )
          {
            v49 = (__int64 *)&v73;
            v50 = 3;
          }
          else
          {
            v49 = v73;
            if ( !v74 )
              goto LABEL_110;
            v50 = v74 - 1;
          }
          v24 = v72;
          LODWORD(v51) = v50 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v25 = &v49[2 * (unsigned int)v51];
          v52 = *v25;
          if ( v21 == (_BYTE *)*v25 )
            goto LABEL_22;
          v53 = 1;
          v54 = 0;
          while ( v52 != -4096 )
          {
            if ( v52 == -8192 && !v54 )
              v54 = v25;
            v51 = v50 & (unsigned int)(v51 + v53);
            v25 = &v49[2 * v51];
            v52 = *v25;
            if ( v21 == (_BYTE *)*v25 )
              goto LABEL_67;
            ++v53;
          }
        }
        else
        {
          if ( v22 - HIDWORD(v72) - v26 > v22 >> 3 )
            goto LABEL_22;
          sub_BB64D0((__int64)&v71, v22);
          if ( (v72 & 1) != 0 )
          {
            v55 = (__int64 *)&v73;
            v56 = 3;
          }
          else
          {
            v55 = v73;
            if ( !v74 )
            {
LABEL_110:
              LODWORD(v72) = (2 * ((unsigned int)v72 >> 1) + 2) | v72 & 1;
              BUG();
            }
            v56 = v74 - 1;
          }
          v57 = 1;
          v54 = 0;
          v24 = v72;
          LODWORD(v58) = v56 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v25 = &v55[2 * (unsigned int)v58];
          v59 = *v25;
          if ( v21 == (_BYTE *)*v25 )
            goto LABEL_22;
          while ( v59 != -4096 )
          {
            if ( v59 == -8192 && !v54 )
              v54 = v25;
            v58 = v56 & (unsigned int)(v58 + v57);
            v25 = &v55[2 * v58];
            v59 = *v25;
            if ( v21 == (_BYTE *)*v25 )
              goto LABEL_67;
            ++v57;
          }
        }
        if ( v54 )
          v25 = v54;
LABEL_67:
        v24 = v72;
LABEL_22:
        LODWORD(v72) = (2 * (v24 >> 1) + 2) | v24 & 1;
        if ( *v25 != -4096 )
          --HIDWORD(v72);
        *v25 = (__int64)v21;
        ++v12;
        *((_DWORD *)v25 + 2) = v19;
        v10 = *(_QWORD *)(*a1 + v69);
        v11 = v72;
        v20 = v72 & 1;
        if ( v13 == v12 )
          goto LABEL_25;
      }
    }
  }
  v20 = v72 & 1;
LABEL_25:
  v28 = (unsigned int)v72 >> 1;
  if ( v20 )
  {
    v30 = (__int64 *)&v73;
    v31 = 3;
  }
  else
  {
    v29 = v74;
    v30 = v73;
    if ( !v74 )
      goto LABEL_87;
    v31 = v74 - 1;
  }
  v32 = v31 & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
  v33 = &v30[2 * v32];
  v34 = *v33;
  if ( v67 == (_BYTE *)*v33 )
    goto LABEL_29;
  v61 = 1;
  while ( v34 != -4096 )
  {
    v64 = v61 + 1;
    v32 = v31 & (v61 + v32);
    v33 = &v30[2 * v32];
    v34 = *v33;
    if ( v67 == (_BYTE *)*v33 )
      goto LABEL_29;
    v61 = v64;
  }
  if ( v20 )
  {
    v60 = 8;
    goto LABEL_88;
  }
  v29 = v74;
LABEL_87:
  v60 = 2 * v29;
LABEL_88:
  v33 = &v30[v60];
LABEL_29:
  v35 = *(_QWORD *)(v10 + 16 * v5);
  if ( v20 )
  {
    v36 = v30 + 8;
    v37 = 3;
    v38 = v28 + (v33 == v30 + 8);
  }
  else
  {
    v36 = &v30[2 * v74];
    v38 = v28 + (v33 == v36);
    if ( !v74 )
      goto LABEL_78;
    v37 = v74 - 1;
  }
  v39 = v37 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
  v40 = &v30[2 * v39];
  v41 = *v40;
  if ( v35 != *v40 )
  {
    v62 = 1;
    while ( v41 != -4096 )
    {
      v63 = v62 + 1;
      v39 = v37 & (v62 + v39);
      v40 = &v30[2 * v39];
      v41 = *v40;
      if ( v35 == *v40 )
        goto LABEL_32;
      v62 = v63;
    }
LABEL_78:
    ++v28;
    result = 0;
    if ( v28 == v38 )
      goto LABEL_44;
    goto LABEL_35;
  }
LABEL_32:
  if ( v40 == v36 )
    goto LABEL_78;
  result = 0;
  if ( v28 != v38 )
  {
    if ( !v28 )
    {
LABEL_39:
      if ( v33 == v36 || !(unsigned __int8)sub_2B0D930(*a5, *((_DWORD *)v33 + 2)) )
      {
        v47 = 1;
        if ( v38 > 1 )
        {
          _BitScanReverse(&v48, v38 - 1);
          v47 = 1 << (32 - (v48 ^ 0x1F));
        }
        v38 = v47 - v38;
      }
      else if ( v38 )
      {
        _BitScanReverse(&v46, v38);
        v38 -= 0x80000000 >> (v46 ^ 0x1F);
      }
      result = v28 - v38;
      goto LABEL_44;
    }
LABEL_35:
    _BitScanReverse(&v42, v28);
    v43 = v28 - (0x80000000 >> (v42 ^ 0x1F));
    if ( v28 <= 1 )
    {
      if ( 1 - v28 <= v43 )
        v43 = 1 - v28;
      v28 = v43;
    }
    else
    {
      _BitScanReverse(&v44, v28 - 1);
      v45 = (1 << (32 - (v44 ^ 0x1F))) - v28;
      if ( v45 <= v43 )
        v43 = v45;
      v28 = v43;
    }
    goto LABEL_39;
  }
LABEL_44:
  if ( !v20 )
  {
    v70 = result;
    sub_C7D6A0((__int64)v73, 16LL * v74, 8);
    return v70;
  }
  return result;
}
