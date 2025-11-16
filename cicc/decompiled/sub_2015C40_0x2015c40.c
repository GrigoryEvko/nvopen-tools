// Function: sub_2015C40
// Address: 0x2015c40
//
__int64 __fastcall sub_2015C40(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128i *a5,
        const __m128i *a6,
        unsigned __int64 a7,
        __int64 a8)
{
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __m128i *v14; // r8
  const __m128i *v15; // r9
  bool v16; // zf
  __int64 *v17; // rax
  _BYTE *v18; // rax
  __int64 v19; // r14
  __int64 v20; // rax
  char v21; // di
  __int64 v22; // rax
  unsigned int v23; // eax
  __int64 v24; // r14
  __int64 v25; // rax
  char v26; // di
  __int64 v27; // rax
  unsigned int v28; // r15d
  __int64 v29; // rax
  char v30; // di
  __int64 v31; // rax
  unsigned int v32; // r9d
  __int64 v33; // rax
  char v34; // di
  __int64 v35; // rax
  unsigned int v36; // eax
  __int64 v37; // r14
  __int64 v38; // rax
  char v39; // di
  __int64 v40; // rax
  unsigned int v41; // r15d
  __int64 v42; // rax
  char v43; // di
  __int64 v44; // rax
  unsigned int v45; // r9d
  int v46; // r13d
  char v47; // al
  __int64 v48; // rdi
  int v49; // esi
  unsigned int v50; // edx
  __int64 v51; // r14
  int v52; // ecx
  int v53; // eax
  unsigned __int64 v54; // rdx
  __int64 result; // rax
  unsigned int v56; // esi
  unsigned int v57; // edi
  int v58; // edx
  unsigned int v59; // ecx
  int v60; // r9d
  __int64 v61; // r8
  __int64 v62; // rsi
  int v63; // eax
  unsigned int v64; // edx
  int v65; // ecx
  __int64 v66; // rsi
  int v67; // edx
  unsigned int v68; // ecx
  int v69; // eax
  int v70; // r8d
  __int64 v71; // rdi
  int v72; // eax
  int v73; // edx
  int v74; // r8d
  unsigned __int64 v75; // [rsp+0h] [rbp-50h] BYREF
  __m128i *v76; // [rsp+8h] [rbp-48h]
  char v77[8]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v78; // [rsp+18h] [rbp-38h]

  v75 = a4;
  v76 = a5;
  v11 = sub_2010420(a1, a4, a3, a4, a5, a6);
  v16 = *((_DWORD *)v11 + 7) == -3;
  v75 = (unsigned __int64)v11;
  if ( v16 )
    sub_2010110(a1, (__int64)&v75);
  v17 = sub_2010420(a1, a7, v12, v13, v14, v15);
  v16 = *((_DWORD *)v17 + 7) == -3;
  a7 = (unsigned __int64)v17;
  if ( v16 )
    sub_2010110(a1, (__int64)&a7);
  v18 = (_BYTE *)sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL));
  v19 = *(_QWORD *)(a1 + 8);
  if ( *v18 )
  {
    v20 = *(_QWORD *)(a7 + 40) + 16LL * (unsigned int)a8;
    v21 = *(_BYTE *)v20;
    v22 = *(_QWORD *)(v20 + 8);
    v77[0] = v21;
    v78 = v22;
    if ( v21 )
      v23 = sub_200D0E0(v21);
    else
      v23 = sub_1F58D40((__int64)v77);
    sub_1D306C0(v19, a2, a3, a7, a8, 0, v23, 0);
    v24 = *(_QWORD *)(a1 + 8);
    v25 = *(_QWORD *)(v75 + 40) + 16LL * (unsigned int)v76;
    v26 = *(_BYTE *)v25;
    v27 = *(_QWORD *)(v25 + 8);
    v77[0] = v26;
    v78 = v27;
    if ( v26 )
      v28 = sub_200D0E0(v26);
    else
      v28 = sub_1F58D40((__int64)v77);
    v29 = *(_QWORD *)(a7 + 40) + 16LL * (unsigned int)a8;
    v30 = *(_BYTE *)v29;
    v31 = *(_QWORD *)(v29 + 8);
    v77[0] = v30;
    v78 = v31;
    if ( v30 )
      v32 = sub_200D0E0(v30);
    else
      v32 = sub_1F58D40((__int64)v77);
    sub_1D306C0(v24, a2, a3, v75, (int)v76, v32, v28, 1);
  }
  else
  {
    v33 = *(_QWORD *)(v75 + 40) + 16LL * (unsigned int)v76;
    v34 = *(_BYTE *)v33;
    v35 = *(_QWORD *)(v33 + 8);
    v77[0] = v34;
    v78 = v35;
    if ( v34 )
      v36 = sub_200D0E0(v34);
    else
      v36 = sub_1F58D40((__int64)v77);
    sub_1D306C0(v19, a2, a3, v75, (int)v76, 0, v36, 0);
    v37 = *(_QWORD *)(a1 + 8);
    v38 = *(_QWORD *)(a7 + 40) + 16LL * (unsigned int)a8;
    v39 = *(_BYTE *)v38;
    v40 = *(_QWORD *)(v38 + 8);
    v77[0] = v39;
    v78 = v40;
    if ( v39 )
      v41 = sub_200D0E0(v39);
    else
      v41 = sub_1F58D40((__int64)v77);
    v42 = *(_QWORD *)(v75 + 40) + 16LL * (unsigned int)v76;
    v43 = *(_BYTE *)v42;
    v44 = *(_QWORD *)(v42 + 8);
    v77[0] = v43;
    v78 = v44;
    if ( v43 )
      v45 = sub_200D0E0(v43);
    else
      v45 = sub_1F58D40((__int64)v77);
    sub_1D306C0(v37, a2, a3, a7, a8, v45, v41, 1);
  }
  v46 = sub_200F8F0(a1, a2, a3);
  v47 = *(_BYTE *)(a1 + 640) & 1;
  if ( v47 )
  {
    v48 = a1 + 648;
    v49 = 7;
  }
  else
  {
    v56 = *(_DWORD *)(a1 + 656);
    v48 = *(_QWORD *)(a1 + 648);
    if ( !v56 )
    {
      v57 = *(_DWORD *)(a1 + 640);
      v51 = 0;
      ++*(_QWORD *)(a1 + 632);
      v58 = (v57 >> 1) + 1;
LABEL_33:
      v59 = 3 * v56;
      goto LABEL_34;
    }
    v49 = v56 - 1;
  }
  v50 = v49 & (37 * v46);
  v51 = v48 + 12LL * v50;
  v52 = *(_DWORD *)v51;
  if ( v46 == *(_DWORD *)v51 )
    goto LABEL_23;
  v60 = 1;
  v61 = 0;
  while ( v52 != -1 )
  {
    if ( !v61 && v52 == -2 )
      v61 = v51;
    v50 = v49 & (v60 + v50);
    v51 = v48 + 12LL * v50;
    v52 = *(_DWORD *)v51;
    if ( v46 == *(_DWORD *)v51 )
      goto LABEL_23;
    ++v60;
  }
  v57 = *(_DWORD *)(a1 + 640);
  v59 = 24;
  v56 = 8;
  if ( v61 )
    v51 = v61;
  ++*(_QWORD *)(a1 + 632);
  v58 = (v57 >> 1) + 1;
  if ( !v47 )
  {
    v56 = *(_DWORD *)(a1 + 656);
    goto LABEL_33;
  }
LABEL_34:
  if ( v59 <= 4 * v58 )
  {
    sub_2015860(a1 + 632, 2 * v56);
    if ( (*(_BYTE *)(a1 + 640) & 1) != 0 )
    {
      v62 = a1 + 648;
      v63 = 7;
    }
    else
    {
      v72 = *(_DWORD *)(a1 + 656);
      v62 = *(_QWORD *)(a1 + 648);
      if ( !v72 )
        goto LABEL_77;
      v63 = v72 - 1;
    }
    v64 = v63 & (37 * v46);
    v51 = v62 + 12LL * v64;
    v65 = *(_DWORD *)v51;
    if ( v46 != *(_DWORD *)v51 )
    {
      v74 = 1;
      v71 = 0;
      while ( v65 != -1 )
      {
        if ( !v71 && v65 == -2 )
          v71 = v51;
        v64 = v63 & (v74 + v64);
        v51 = v62 + 12LL * v64;
        v65 = *(_DWORD *)v51;
        if ( v46 == *(_DWORD *)v51 )
          goto LABEL_48;
        ++v74;
      }
      goto LABEL_54;
    }
LABEL_48:
    v57 = *(_DWORD *)(a1 + 640);
    goto LABEL_36;
  }
  if ( v56 - *(_DWORD *)(a1 + 644) - v58 <= v56 >> 3 )
  {
    sub_2015860(a1 + 632, v56);
    if ( (*(_BYTE *)(a1 + 640) & 1) != 0 )
    {
      v66 = a1 + 648;
      v67 = 7;
      goto LABEL_51;
    }
    v73 = *(_DWORD *)(a1 + 656);
    v66 = *(_QWORD *)(a1 + 648);
    if ( v73 )
    {
      v67 = v73 - 1;
LABEL_51:
      v68 = v67 & (37 * v46);
      v51 = v66 + 12LL * v68;
      v69 = *(_DWORD *)v51;
      if ( v46 != *(_DWORD *)v51 )
      {
        v70 = 1;
        v71 = 0;
        while ( v69 != -1 )
        {
          if ( !v71 && v69 == -2 )
            v71 = v51;
          v68 = v67 & (v70 + v68);
          v51 = v66 + 12LL * v68;
          v69 = *(_DWORD *)v51;
          if ( v46 == *(_DWORD *)v51 )
            goto LABEL_48;
          ++v70;
        }
LABEL_54:
        if ( v71 )
          v51 = v71;
        goto LABEL_48;
      }
      goto LABEL_48;
    }
LABEL_77:
    *(_DWORD *)(a1 + 640) = (2 * (*(_DWORD *)(a1 + 640) >> 1) + 2) | *(_DWORD *)(a1 + 640) & 1;
    BUG();
  }
LABEL_36:
  *(_DWORD *)(a1 + 640) = (2 * (v57 >> 1) + 2) | v57 & 1;
  if ( *(_DWORD *)v51 != -1 )
    --*(_DWORD *)(a1 + 644);
  *(_DWORD *)v51 = v46;
  *(_QWORD *)(v51 + 4) = 0;
LABEL_23:
  *(_DWORD *)(v51 + 4) = sub_200F8F0(a1, v75, (__int64)v76);
  v53 = sub_200F8F0(a1, a7, a8);
  v54 = v75;
  *(_DWORD *)(v51 + 8) = v53;
  result = *(unsigned int *)(a2 + 64);
  *(_DWORD *)(v54 + 64) = result;
  *(_DWORD *)(a7 + 64) = result;
  return result;
}
