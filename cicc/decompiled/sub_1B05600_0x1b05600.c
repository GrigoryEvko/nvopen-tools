// Function: sub_1B05600
// Address: 0x1b05600
//
__int64 __fastcall sub_1B05600(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v6; // rdx
  unsigned int v7; // r14d
  __int64 v9; // r13
  __int64 v10; // r14
  unsigned int v11; // edx
  __int64 v12; // rbx
  __int64 v13; // rax
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // rdx
  __int64 v17; // r12
  char v18; // si
  unsigned int v19; // edx
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rdi
  __int64 v23; // rcx
  __int64 v24; // rbx
  __int64 v25; // rax
  __int64 v26; // r12
  _QWORD *v27; // rdx
  _QWORD *v28; // rax
  _QWORD *v29; // rbx
  _QWORD *v30; // rsi
  __int64 v31; // rdx
  _QWORD *v32; // rdx
  unsigned __int64 v33; // rdi
  int v34; // r8d
  int v35; // r9d
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 i; // rbx
  __int64 v39; // r12
  __int64 v40; // rsi
  _QWORD *j; // rax
  unsigned int v42; // [rsp+14h] [rbp-23Ch]
  __int64 v43; // [rsp+28h] [rbp-228h]
  __int64 v44; // [rsp+30h] [rbp-220h]
  __int64 v45; // [rsp+30h] [rbp-220h]
  __int64 v46; // [rsp+38h] [rbp-218h]
  __int64 v47; // [rsp+40h] [rbp-210h]
  __int64 v48; // [rsp+40h] [rbp-210h]
  __int64 v49; // [rsp+40h] [rbp-210h]
  __int16 v51; // [rsp+50h] [rbp-200h] BYREF
  _QWORD v52[3]; // [rsp+58h] [rbp-1F8h] BYREF
  int v53; // [rsp+70h] [rbp-1E0h]
  _QWORD v54[2]; // [rsp+80h] [rbp-1D0h] BYREF
  _BYTE v55[32]; // [rsp+90h] [rbp-1C0h] BYREF
  unsigned __int64 v56[2]; // [rsp+B0h] [rbp-1A0h] BYREF
  _BYTE v57[32]; // [rsp+C0h] [rbp-190h] BYREF
  __int64 v58; // [rsp+E0h] [rbp-170h] BYREF
  _BYTE *v59; // [rsp+E8h] [rbp-168h]
  _BYTE *v60; // [rsp+F0h] [rbp-160h]
  __int64 v61; // [rsp+F8h] [rbp-158h]
  int v62; // [rsp+100h] [rbp-150h]
  _BYTE v63[40]; // [rsp+108h] [rbp-148h] BYREF
  __int64 v64; // [rsp+130h] [rbp-120h] BYREF
  _BYTE *v65; // [rsp+138h] [rbp-118h]
  _BYTE *v66; // [rsp+140h] [rbp-110h]
  __int64 v67; // [rsp+148h] [rbp-108h]
  int v68; // [rsp+150h] [rbp-100h]
  _BYTE v69[40]; // [rsp+158h] [rbp-F8h] BYREF
  __int64 v70; // [rsp+180h] [rbp-D0h] BYREF
  _BYTE *v71; // [rsp+188h] [rbp-C8h]
  _BYTE *v72; // [rsp+190h] [rbp-C0h]
  __int64 v73; // [rsp+198h] [rbp-B8h]
  int v74; // [rsp+1A0h] [rbp-B0h]
  _BYTE v75[40]; // [rsp+1A8h] [rbp-A8h] BYREF
  _BYTE *v76; // [rsp+1D0h] [rbp-80h] BYREF
  __int64 v77; // [rsp+1D8h] [rbp-78h]
  _BYTE v78[112]; // [rsp+1E0h] [rbp-70h] BYREF

  if ( !(unsigned __int8)sub_13FCBF0(a1) )
    return 0;
  v6 = *(__int64 **)(a1 + 8);
  if ( *(_QWORD *)(a1 + 16) - (_QWORD)v6 != 8 )
    return 0;
  v9 = *v6;
  if ( !(unsigned __int8)sub_13FCBF0(*v6) )
    return 0;
  v44 = **(_QWORD **)(a1 + 32);
  v47 = sub_13FCB50(a1);
  v10 = sub_13F9E70(a1);
  v43 = **(_QWORD **)(v9 + 32);
  v46 = sub_13FCB50(v9);
  LOBYTE(v11) = (v47 != v10) | (v46 != sub_13F9E70(v9));
  v7 = v11;
  if ( (_BYTE)v11 || *(_WORD *)(v44 + 18) || *(_WORD *)(v43 + 18) )
    return 0;
  v59 = v63;
  v60 = v63;
  v65 = v69;
  v66 = v69;
  v58 = 0;
  v61 = 4;
  v62 = 0;
  v64 = 0;
  v67 = 4;
  v68 = 0;
  v70 = 0;
  v71 = v75;
  v72 = v75;
  v73 = 4;
  v74 = 0;
  if ( !(unsigned __int8)sub_1B04FC0(a1, v9, (__int64)&v64, (__int64)&v58, (__int64)&v70, a3) )
    goto LABEL_11;
  if ( HIDWORD(v73) - v74 != 1 )
    goto LABEL_11;
  v12 = sub_1474160(a2, v9, v46);
  if ( sub_14562D0(v12) )
    goto LABEL_11;
  if ( *(_BYTE *)(sub_1456040(v12) + 8) != 11 )
    goto LABEL_11;
  v42 = sub_146CB30(a2, v12, a1);
  if ( v42 != 1 )
    goto LABEL_11;
  v51 = 0;
  memset(v52, 0, sizeof(v52));
  v53 = 0;
  sub_1436EA0((__int64)&v51, a1);
  v7 = (unsigned __int8)v51;
  if ( (_BYTE)v51 )
  {
    v7 = 0;
    goto LABEL_52;
  }
  v76 = v78;
  v77 = 0x800000000LL;
  v13 = sub_157F280(v44);
  v45 = v16;
  v17 = v13;
  while ( v17 != v45 )
  {
    v18 = *(_BYTE *)(v17 + 23) & 0x40;
    v19 = *(_DWORD *)(v17 + 20) & 0xFFFFFFF;
    if ( v19 )
    {
      v15 = v47;
      v14 = v17 - 24 * v19;
      v20 = 0;
      v21 = 24LL * *(unsigned int *)(v17 + 56) + 8;
      while ( 1 )
      {
        v22 = v17 - 24LL * v19;
        if ( v18 )
          v22 = *(_QWORD *)(v17 - 8);
        if ( v47 == *(_QWORD *)(v22 + v21) )
          break;
        v20 = (unsigned int)(v20 + 1);
        v21 += 8;
        if ( v19 == (_DWORD)v20 )
          goto LABEL_39;
      }
    }
    else
    {
LABEL_39:
      v20 = 0xFFFFFFFFLL;
    }
    if ( v18 )
      v23 = *(_QWORD *)(v17 - 8);
    else
      v23 = v17 - 24LL * v19;
    v24 = *(_QWORD *)(v23 + 24 * v20);
    if ( *(_BYTE *)(v24 + 16) > 0x17u )
    {
      if ( (unsigned int)v77 >= HIDWORD(v77) )
        sub_16CD150((__int64)&v76, v78, 0, 8, v14, v15);
      *(_QWORD *)&v76[8 * (unsigned int)v77] = v24;
      LODWORD(v77) = v77 + 1;
    }
    v25 = *(_QWORD *)(v17 + 32);
    if ( !v25 )
      BUG();
    v17 = 0;
    if ( *(_BYTE *)(v25 - 8) == 77 )
      v17 = v25 - 24;
  }
  while ( (_DWORD)v77 )
  {
    v26 = *(_QWORD *)&v76[8 * (unsigned int)v77 - 8];
    LODWORD(v77) = v77 - 1;
    v27 = *(_QWORD **)(v9 + 72);
    v28 = *(_QWORD **)(v9 + 64);
    if ( v27 == v28 )
    {
      v40 = *(unsigned int *)(v9 + 84);
      v29 = &v27[v40];
      if ( v27 != v29 )
      {
        do
        {
          if ( *(_QWORD *)(v26 + 40) == *v28 )
            break;
          ++v28;
        }
        while ( v29 != v28 );
      }
    }
    else
    {
      v48 = *(_QWORD *)(v26 + 40);
      v29 = &v27[*(unsigned int *)(v9 + 80)];
      v28 = sub_16CC9F0(v9 + 56, v48);
      if ( v48 == *v28 )
      {
        v30 = *(_QWORD **)(v9 + 72);
        v27 = *(_QWORD **)(v9 + 64);
        if ( v30 != v27 )
        {
          v31 = *(unsigned int *)(v9 + 80);
          goto LABEL_45;
        }
        v40 = *(unsigned int *)(v9 + 84);
      }
      else
      {
        v27 = *(_QWORD **)(v9 + 72);
        v30 = v27;
        if ( v27 != *(_QWORD **)(v9 + 64) )
        {
          v31 = *(unsigned int *)(v9 + 80);
          v28 = &v30[v31];
LABEL_45:
          v32 = &v30[v31];
          goto LABEL_46;
        }
        v40 = *(unsigned int *)(v9 + 84);
        v28 = &v27[v40];
      }
    }
    v32 = &v27[v40];
LABEL_46:
    while ( v32 != v28 )
    {
      if ( *v28 < 0xFFFFFFFFFFFFFFFELL )
        break;
      ++v28;
    }
    if ( v28 != v29
      || sub_183E920((__int64)&v70, *(_QWORD *)(v26 + 40))
      && (*(_BYTE *)(v26 + 16) == 77
       || (unsigned __int8)sub_15F3040(v26)
       || sub_15F3330(v26)
       || (unsigned __int8)sub_15F2ED0(v26)
       || (unsigned __int8)sub_15F3040(v26)) )
    {
      v33 = (unsigned __int64)v76;
      if ( v76 == v78 )
        goto LABEL_52;
      goto LABEL_51;
    }
    if ( sub_183E920((__int64)&v70, *(_QWORD *)(v26 + 40)) )
    {
      v36 = 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF);
      v37 = v26 - v36;
      if ( (*(_BYTE *)(v26 + 23) & 0x40) != 0 )
        v37 = *(_QWORD *)(v26 - 8);
      v49 = v37 + v36;
      for ( i = v37; v49 != i; i += 24 )
      {
        v39 = *(_QWORD *)i;
        if ( *(_BYTE *)(*(_QWORD *)i + 16LL) > 0x17u )
        {
          if ( HIDWORD(v77) <= (unsigned int)v77 )
            sub_16CD150((__int64)&v76, v78, 0, 8, v34, v35);
          *(_QWORD *)&v76[8 * (unsigned int)v77] = v39;
          LODWORD(v77) = v77 + 1;
        }
      }
    }
  }
  if ( v76 != v78 )
    _libc_free((unsigned __int64)v76);
  v54[0] = v55;
  v54[1] = 0x400000000LL;
  v56[0] = (unsigned __int64)v57;
  v56[1] = 0x400000000LL;
  v76 = v78;
  v77 = 0x400000000LL;
  if ( (unsigned __int8)sub_1B04DA0((__int64)&v64, (__int64)v54) )
  {
    if ( (unsigned __int8)sub_1B04DA0((__int64)&v58, (__int64)v56) )
    {
      if ( (unsigned __int8)sub_1B04DA0((__int64)&v70, (__int64)&v76) )
      {
        for ( j = *(_QWORD **)a1; j; j = (_QWORD *)*j )
          ++v42;
        if ( (unsigned __int8)sub_1B04A80((__int64)v54, (__int64)v56, v42, 0, a4)
          && (unsigned __int8)sub_1B04A80((__int64)v54, (__int64)&v76, v42, 0, a4)
          && (unsigned __int8)sub_1B04A80((__int64)v56, (__int64)&v76, v42, 0, a4) )
        {
          v7 = sub_1B04A80((__int64)v56, (__int64)v56, v42, 1, a4);
        }
      }
    }
  }
  if ( v76 != v78 )
    _libc_free((unsigned __int64)v76);
  if ( (_BYTE *)v56[0] != v57 )
    _libc_free(v56[0]);
  v33 = v54[0];
  if ( (_BYTE *)v54[0] != v55 )
LABEL_51:
    _libc_free(v33);
LABEL_52:
  sub_1B05570((__int64)v52);
LABEL_11:
  if ( v72 != v71 )
    _libc_free((unsigned __int64)v72);
  if ( v66 != v65 )
    _libc_free((unsigned __int64)v66);
  if ( v60 != v59 )
    _libc_free((unsigned __int64)v60);
  return v7;
}
