// Function: sub_1D4FC30
// Address: 0x1d4fc30
//
void __fastcall sub_1D4FC30(__int64 a1)
{
  __int64 *v1; // rax
  __int64 *v2; // rsi
  _QWORD *v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // rdx
  unsigned int v7; // edx
  char v8; // dl
  __int64 v9; // r15
  __int64 *v10; // rsi
  __int64 *v11; // rcx
  unsigned int *v12; // r14
  unsigned int *i; // r9
  __int64 v14; // r8
  __int64 v15; // rax
  _QWORD *v16; // rax
  int v17; // r15d
  __int64 v18; // rsi
  __int64 v19; // r14
  __int64 v20; // rax
  char v21; // dl
  __int64 v22; // rax
  unsigned __int8 v23; // al
  __int64 v24; // r14
  bool v25; // al
  unsigned int v26; // r15d
  unsigned __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // r15
  bool v30; // cc
  unsigned __int64 v31; // rdx
  __int64 v32; // rax
  unsigned __int64 v33; // rsi
  __int64 v34; // rsi
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // rax
  unsigned int v39; // ecx
  unsigned int v40; // ecx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdi
  __int64 v45; // rdi
  __int64 v46; // [rsp+0h] [rbp-550h]
  __int64 v47; // [rsp+0h] [rbp-550h]
  __int64 v48; // [rsp+8h] [rbp-548h]
  __int64 v49; // [rsp+8h] [rbp-548h]
  __int64 v50; // [rsp+10h] [rbp-540h]
  __int64 v51; // [rsp+10h] [rbp-540h]
  int v52; // [rsp+20h] [rbp-530h]
  int v53; // [rsp+20h] [rbp-530h]
  unsigned int v54; // [rsp+20h] [rbp-530h]
  __int64 v55; // [rsp+20h] [rbp-530h]
  int v56; // [rsp+28h] [rbp-528h]
  unsigned int *v57; // [rsp+28h] [rbp-528h]
  char v58[8]; // [rsp+30h] [rbp-520h] BYREF
  __int64 v59; // [rsp+38h] [rbp-518h]
  unsigned __int64 v60; // [rsp+40h] [rbp-510h] BYREF
  __int64 v61; // [rsp+48h] [rbp-508h]
  __int64 v62; // [rsp+50h] [rbp-500h] BYREF
  __int64 v63; // [rsp+58h] [rbp-4F8h]
  __int64 v64; // [rsp+60h] [rbp-4F0h] BYREF
  __int64 *v65; // [rsp+68h] [rbp-4E8h]
  __int64 *v66; // [rsp+70h] [rbp-4E0h]
  __int64 v67; // [rsp+78h] [rbp-4D8h]
  int v68; // [rsp+80h] [rbp-4D0h]
  _BYTE v69[136]; // [rsp+88h] [rbp-4C8h] BYREF
  _QWORD *v70; // [rsp+110h] [rbp-440h] BYREF
  __int64 v71; // [rsp+118h] [rbp-438h]
  _QWORD v72[134]; // [rsp+120h] [rbp-430h] BYREF

  v1 = (__int64 *)v69;
  v2 = (__int64 *)v69;
  v4 = v72;
  v5 = *(_QWORD *)(a1 + 272);
  v64 = 0;
  v65 = (__int64 *)v69;
  v6 = *(_QWORD *)(v5 + 176);
  v66 = (__int64 *)v69;
  v67 = 16;
  v72[0] = v6;
  v7 = 1;
  v68 = 0;
  v70 = v72;
  v71 = 0x8000000001LL;
  v60 = 0;
  v61 = 1;
  v62 = 0;
  v63 = 1;
  while ( 1 )
  {
    v9 = v4[v7 - 1];
    LODWORD(v71) = v7 - 1;
    if ( v1 != v2 )
      goto LABEL_2;
    v10 = &v1[HIDWORD(v67)];
    if ( v10 != v1 )
    {
      v11 = 0;
      while ( v9 != *v1 )
      {
        if ( *v1 == -2 )
          v11 = v1;
        if ( v10 == ++v1 )
        {
          if ( !v11 )
            goto LABEL_51;
          *v11 = v9;
          --v68;
          ++v64;
          goto LABEL_14;
        }
      }
      goto LABEL_3;
    }
LABEL_51:
    if ( HIDWORD(v67) < (unsigned int)v67 )
    {
      ++HIDWORD(v67);
      *v10 = v9;
      ++v64;
    }
    else
    {
LABEL_2:
      sub_16CCBA0((__int64)&v64, v9);
      if ( !v8 )
        goto LABEL_3;
    }
LABEL_14:
    v12 = *(unsigned int **)(v9 + 32);
    for ( i = &v12[10 * *(unsigned int *)(v9 + 56)]; i != v12; v12 += 10 )
    {
      v14 = *(_QWORD *)v12;
      if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v12 + 40LL) + 16LL * v12[2]) == 1 )
      {
        v15 = (unsigned int)v71;
        if ( (unsigned int)v71 >= HIDWORD(v71) )
        {
          v55 = *(_QWORD *)v12;
          v57 = i;
          sub_16CD150((__int64)&v70, v72, 0, 8, v14, (int)i);
          v15 = (unsigned int)v71;
          v14 = v55;
          i = v57;
        }
        v70[v15] = v14;
        LODWORD(v71) = v71 + 1;
      }
    }
    if ( *(_WORD *)(v9 + 24) != 46 )
      goto LABEL_3;
    v16 = *(_QWORD **)(v9 + 32);
    v17 = *(_DWORD *)(v16[5] + 84LL);
    if ( v17 >= 0 )
      goto LABEL_3;
    v18 = v16[10];
    v19 = v16[11];
    v20 = *(_QWORD *)(v18 + 40) + 16LL * *((unsigned int *)v16 + 22);
    v21 = *(_BYTE *)v20;
    v22 = *(_QWORD *)(v20 + 8);
    v58[0] = v21;
    v59 = v22;
    if ( v21 )
      break;
    if ( (unsigned __int8)sub_1F58CF0(v58) && !(unsigned __int8)sub_1F58D20(v58) )
      goto LABEL_26;
LABEL_3:
    v7 = v71;
    if ( !(_DWORD)v71 )
      goto LABEL_40;
LABEL_4:
    v4 = v70;
    v2 = v66;
    v1 = v65;
  }
  v23 = v21 - 14;
  if ( (unsigned __int8)(v21 - 2) > 5u && v23 > 0x47u || v23 <= 0x5Fu )
    goto LABEL_3;
LABEL_26:
  v56 = sub_1D23330(*(_QWORD *)(a1 + 272), v18, v19, 0);
  sub_1D1F820(*(_QWORD *)(a1 + 272), v18, v19, &v60, 0);
  v24 = *(_QWORD *)(a1 + 248);
  if ( v56 == 1 )
  {
    if ( (unsigned int)v61 <= 0x40 )
    {
      v25 = v60 == 0;
    }
    else
    {
      v52 = v61;
      v25 = v52 == (unsigned int)sub_16A57B0((__int64)&v60);
    }
    if ( v25 )
    {
      if ( (unsigned int)v63 <= 0x40 )
      {
        if ( !v62 )
          goto LABEL_3;
      }
      else
      {
        v53 = v63;
        if ( v53 == (unsigned int)sub_16A57B0((__int64)&v62) )
          goto LABEL_3;
      }
    }
  }
  v26 = v17 & 0x7FFFFFFF;
  v54 = v26 + 1;
  v27 = *(unsigned int *)(v24 + 952);
  if ( v26 + 1 > (unsigned int)v27 )
  {
    v37 = v54;
    if ( v54 < v27 )
    {
      v28 = *(_QWORD *)(v24 + 944);
      v43 = v28 + 40 * v27;
      v51 = v28 + 40LL * v54;
      if ( v43 == v51 )
      {
LABEL_85:
        *(_DWORD *)(v24 + 952) = v54;
        goto LABEL_34;
      }
      do
      {
        v43 -= 40;
        if ( *(_DWORD *)(v43 + 32) > 0x40u )
        {
          v44 = *(_QWORD *)(v43 + 24);
          if ( v44 )
          {
            v48 = v43;
            j_j___libc_free_0_0(v44);
            v43 = v48;
          }
        }
        if ( *(_DWORD *)(v43 + 16) > 0x40u )
        {
          v45 = *(_QWORD *)(v43 + 8);
          if ( v45 )
          {
            v49 = v43;
            j_j___libc_free_0_0(v45);
            v43 = v49;
          }
        }
      }
      while ( v51 != v43 );
    }
    else
    {
      if ( v54 <= v27 )
        goto LABEL_33;
      if ( v54 > (unsigned __int64)*(unsigned int *)(v24 + 956) )
      {
        sub_1D4FA80(v24 + 944, v54);
        v27 = *(unsigned int *)(v24 + 952);
        v37 = v54;
      }
      v28 = *(_QWORD *)(v24 + 944);
      v38 = v28 + 40 * v27;
      v50 = v28 + 40 * v37;
      if ( v50 == v38 )
        goto LABEL_85;
      do
      {
        if ( v38 )
        {
          *(_DWORD *)v38 = *(_DWORD *)(v24 + 960);
          v40 = *(_DWORD *)(v24 + 976);
          *(_DWORD *)(v38 + 16) = v40;
          if ( v40 <= 0x40 )
          {
            *(_QWORD *)(v38 + 8) = *(_QWORD *)(v24 + 968);
          }
          else
          {
            v46 = v38;
            sub_16A4FD0(v38 + 8, (const void **)(v24 + 968));
            v38 = v46;
          }
          v39 = *(_DWORD *)(v24 + 992);
          *(_DWORD *)(v38 + 32) = v39;
          if ( v39 > 0x40 )
          {
            v47 = v38;
            sub_16A4FD0(v38 + 24, (const void **)(v24 + 984));
            v38 = v47;
          }
          else
          {
            *(_QWORD *)(v38 + 24) = *(_QWORD *)(v24 + 984);
          }
        }
        v38 += 40;
      }
      while ( v50 != v38 );
    }
    v28 = *(_QWORD *)(v24 + 944);
    goto LABEL_85;
  }
LABEL_33:
  v28 = *(_QWORD *)(v24 + 944);
LABEL_34:
  v29 = v28 + 40LL * v26;
  v30 = *(_DWORD *)(v29 + 32) <= 0x40u;
  *(_DWORD *)v29 = v56 & 0x7FFFFFFF | *(_DWORD *)v29 & 0x80000000;
  if ( v30 && (unsigned int)v63 <= 0x40 )
  {
    v34 = v62;
    *(_QWORD *)(v29 + 24) = v62;
    v35 = (unsigned int)v63;
    *(_DWORD *)(v29 + 32) = v63;
    v36 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v35;
    if ( (unsigned int)v35 > 0x40 )
    {
      v42 = (unsigned int)((unsigned __int64)(v35 + 63) >> 6) - 1;
      *(_QWORD *)(v34 + 8 * v42) &= v36;
    }
    else
    {
      *(_QWORD *)(v29 + 24) = v34 & v36;
    }
  }
  else
  {
    sub_16A51C0(v29 + 24, (__int64)&v62);
  }
  if ( *(_DWORD *)(v29 + 16) <= 0x40u && (unsigned int)v61 <= 0x40 )
  {
    v31 = v60;
    *(_QWORD *)(v29 + 8) = v60;
    v32 = (unsigned int)v61;
    *(_DWORD *)(v29 + 16) = v61;
    v33 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v32;
    if ( (unsigned int)v32 > 0x40 )
    {
      v41 = (unsigned int)((unsigned __int64)(v32 + 63) >> 6) - 1;
      *(_QWORD *)(v31 + 8 * v41) &= v33;
    }
    else
    {
      *(_QWORD *)(v29 + 8) = v33 & v31;
    }
    goto LABEL_3;
  }
  sub_16A51C0(v29 + 8, (__int64)&v60);
  v7 = v71;
  if ( (_DWORD)v71 )
    goto LABEL_4;
LABEL_40:
  if ( (unsigned int)v63 > 0x40 && v62 )
    j_j___libc_free_0_0(v62);
  if ( (unsigned int)v61 > 0x40 && v60 )
    j_j___libc_free_0_0(v60);
  if ( v70 != v72 )
    _libc_free((unsigned __int64)v70);
  if ( v66 != v65 )
    _libc_free((unsigned __int64)v66);
}
