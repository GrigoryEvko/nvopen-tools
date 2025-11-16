// Function: sub_1707490
// Address: 0x1707490
//
_QWORD *__fastcall sub_1707490(__int64 a1, unsigned __int8 *a2, double a3, double a4, double a5)
{
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r14
  unsigned __int8 v9; // cl
  __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rdx
  _QWORD *result; // rax
  __int64 **v14; // rdi
  __int64 v15; // rax
  int v16; // r8d
  int v17; // r9d
  __int64 *v18; // rcx
  __int64 v19; // rbx
  __int64 *v20; // rsi
  __int64 v21; // r15
  __int64 v22; // rax
  _BYTE *v23; // rdi
  __int64 *v24; // rdx
  __int64 v25; // rcx
  _QWORD *v26; // r11
  __int64 v27; // rbx
  _QWORD *v28; // r10
  __int64 v29; // rsi
  __int64 v30; // rcx
  __int64 v31; // rdx
  int v32; // esi
  __int64 v33; // rdi
  unsigned __int8 *v34; // r12
  __int64 v35; // rax
  __int64 v36; // r14
  __int64 v37; // rax
  int v38; // edx
  __int64 v39; // r14
  unsigned int v40; // r11d
  unsigned __int8 v41; // r15
  __int64 **v42; // rsi
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // rbx
  int v46; // r9d
  __int64 *v47; // rdi
  __int64 v48; // r8
  __int64 v49; // r12
  __int64 v50; // rax
  _QWORD *v51; // rax
  unsigned __int8 *v52; // r12
  __int64 v53; // r13
  unsigned int v54; // r11d
  __int64 v55; // r8
  int v56; // [rsp+8h] [rbp-1E8h]
  __int64 v57; // [rsp+8h] [rbp-1E8h]
  int v58; // [rsp+8h] [rbp-1E8h]
  __int64 v59; // [rsp+18h] [rbp-1D8h]
  __int64 **v60; // [rsp+18h] [rbp-1D8h]
  unsigned __int8 *v61; // [rsp+20h] [rbp-1D0h]
  unsigned int v62; // [rsp+30h] [rbp-1C0h]
  unsigned int v63; // [rsp+30h] [rbp-1C0h]
  __int64 v64; // [rsp+30h] [rbp-1C0h]
  _QWORD *v65; // [rsp+40h] [rbp-1B0h]
  _QWORD *v67; // [rsp+48h] [rbp-1A8h]
  _QWORD *v68; // [rsp+48h] [rbp-1A8h]
  _QWORD *v69; // [rsp+48h] [rbp-1A8h]
  _QWORD *v70; // [rsp+48h] [rbp-1A8h]
  unsigned __int64 v71[2]; // [rsp+50h] [rbp-1A0h] BYREF
  _BYTE v72[64]; // [rsp+60h] [rbp-190h] BYREF
  __int64 *v73; // [rsp+A0h] [rbp-150h] BYREF
  __int64 v74; // [rsp+A8h] [rbp-148h]
  _BYTE v75[128]; // [rsp+B0h] [rbp-140h] BYREF
  __int64 *v76; // [rsp+130h] [rbp-C0h] BYREF
  __int64 v77; // [rsp+138h] [rbp-B8h]
  _WORD s[88]; // [rsp+140h] [rbp-B0h] BYREF

  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 16 || !(unsigned __int8)sub_14AF470((__int64)a2, 0, 0, 0) )
    return 0;
  v6 = *((_QWORD *)a2 - 6);
  v7 = *(_QWORD *)a2;
  v8 = *((_QWORD *)a2 - 3);
  v9 = *(_BYTE *)(v6 + 16);
  v10 = *(_QWORD *)(*(_QWORD *)a2 + 32LL);
  v11 = *(_QWORD *)(v6 + 8);
  if ( v9 == 85 )
  {
    v26 = *(_QWORD **)(v6 - 72);
    if ( v26 )
    {
      if ( *(_BYTE *)(*(_QWORD *)(v6 - 48) + 16LL) == 9 )
      {
        v27 = *(_QWORD *)(v6 - 24);
        if ( *(_BYTE *)(v27 + 16) <= 0x10u && *(_BYTE *)(v8 + 16) == 85 )
        {
          v28 = *(_QWORD **)(v8 - 72);
          if ( v28 )
          {
            if ( *(_BYTE *)(*(_QWORD *)(v8 - 48) + 16LL) == 9
              && v27 == *(_QWORD *)(v8 - 24)
              && *v28 == *v26
              && (v11 && !*(_QWORD *)(v11 + 8) || (v29 = *(_QWORD *)(v8 + 8)) != 0 && !*(_QWORD *)(v29 + 8) || v6 == v8) )
            {
              v30 = *(_QWORD *)(v8 - 72);
              v31 = *(_QWORD *)(v6 - 72);
              v32 = a2[16];
              v33 = *(_QWORD *)(a1 + 8);
              s[0] = 257;
              v34 = (unsigned __int8 *)sub_17066B0(v33, v32 - 24, v31, v30, (__int64 *)&v76, 0, a3, a4, a5);
              if ( (unsigned __int8)(v34[16] - 35) <= 0x11u )
                sub_15F2530(v34, (__int64)a2, 1);
              v35 = sub_1599EF0(*(__int64 ***)v34);
              s[0] = 257;
              v36 = v35;
              result = sub_1648A60(56, 3u);
              if ( result )
              {
                v67 = result;
                sub_15FA660((__int64)result, v34, v36, (_QWORD *)v27, (__int64)&v76, 0);
                return v67;
              }
              return result;
            }
          }
        }
      }
    }
  }
  if ( !v11
    || *(_QWORD *)(v11 + 8)
    || v9 != 85
    || (v65 = *(_QWORD **)(v6 - 72)) == 0
    || *(_BYTE *)(*(_QWORD *)(v6 - 48) + 16LL) != 9
    || (v61 = *(unsigned __int8 **)(v6 - 24), v61[16] > 0x10u)
    || *(_BYTE *)(v8 + 16) > 0x10u )
  {
    v12 = *(_QWORD *)(v8 + 8);
    if ( !v12 )
      return 0;
    if ( *(_QWORD *)(v12 + 8) )
      return 0;
    if ( *(_BYTE *)(v8 + 16) != 85 )
      return 0;
    v65 = *(_QWORD **)(v8 - 72);
    if ( !v65 )
      return 0;
    if ( *(_BYTE *)(*(_QWORD *)(v8 - 48) + 16LL) != 9 )
      return 0;
    v61 = *(unsigned __int8 **)(v8 - 24);
    if ( v61[16] > 0x10u || v9 > 0x10u )
      return 0;
    v8 = v6;
  }
  if ( v7 != *v65 )
    return 0;
  v71[0] = (unsigned __int64)v72;
  v71[1] = 0x1000000000LL;
  sub_15FAA20(v61, (__int64)v71);
  v14 = *(__int64 ***)v8;
  if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) == 16 )
    v14 = (__int64 **)*v14[2];
  v15 = sub_1599EF0(v14);
  v18 = (__int64 *)v75;
  v19 = (unsigned int)v10;
  v73 = (__int64 *)v75;
  v74 = 0x1000000000LL;
  if ( (unsigned int)v10 > 0x10 )
  {
    v64 = v15;
    sub_16CD150((__int64)&v73, v75, (unsigned int)v10, 8, v16, v17);
    v18 = v73;
    v15 = v64;
  }
  v20 = &v18[(unsigned int)v10];
  for ( LODWORD(v74) = v10; v20 != v18; ++v18 )
    *v18 = v15;
  if ( !(_DWORD)v10 )
  {
LABEL_53:
    v37 = sub_15A01B0(v73, (unsigned int)v74);
    v38 = a2[16];
    v39 = v37;
    v40 = v38 - 24;
    v41 = *(_BYTE *)(*((_QWORD *)a2 - 3) + 16LL);
    if ( (unsigned __int8)(a2[16] - 44) > 1u && (unsigned int)(v38 - 41) > 1 )
    {
      if ( (unsigned int)(v38 - 47) > 2 || v41 > 0x10u )
      {
LABEL_69:
        if ( *(_BYTE *)(v6 + 16) <= 0x10u )
        {
          v51 = (_QWORD *)v39;
          v39 = (__int64)v65;
          v65 = v51;
        }
        s[0] = 257;
        v52 = (unsigned __int8 *)sub_17066B0(
                                   *(_QWORD *)(a1 + 8),
                                   v40,
                                   (__int64)v65,
                                   v39,
                                   (__int64 *)&v76,
                                   0,
                                   a3,
                                   a4,
                                   a5);
        if ( (unsigned __int8)(v52[16] - 35) <= 0x11u )
          sub_15F2530(v52, (__int64)a2, 1);
        v53 = sub_1599EF0(*(__int64 ***)v52);
        s[0] = 257;
        result = sub_1648A60(56, 3u);
        if ( result )
        {
          v68 = result;
          sub_15FA660((__int64)result, v52, v53, v61, (__int64)&v76, 0);
          result = v68;
        }
        if ( v73 != (__int64 *)v75 )
        {
          v69 = result;
          _libc_free((unsigned __int64)v73);
          result = v69;
        }
        if ( (_BYTE *)v71[0] != v72 )
        {
          v70 = result;
          _libc_free(v71[0]);
          return v70;
        }
        return result;
      }
      v63 = v38 - 24;
      v60 = **(__int64 ****)(*(_QWORD *)v37 + 16LL);
      v45 = sub_15A14F0(v40, v60, 1);
      if ( v45 )
      {
LABEL_56:
        v47 = (__int64 *)s;
        v48 = *(_QWORD *)(*(_QWORD *)v39 + 32LL);
        v76 = (__int64 *)s;
        v59 = (unsigned int)v48;
        v77 = 0x1000000000LL;
        if ( (unsigned int)v48 > 0x10 )
        {
          v58 = v48;
          sub_16CD150((__int64)&v76, s, (unsigned int)v48, 8, v48, v46);
          v47 = v76;
          LODWORD(v48) = v58;
        }
        LODWORD(v77) = v48;
        if ( 8 * v59 )
        {
          v56 = v48;
          memset(v47, 0, 8 * v59);
          LODWORD(v48) = v56;
        }
        if ( (_DWORD)v48 )
        {
          v57 = v6;
          v49 = 0;
          do
          {
            v50 = sub_15A0A60(v39, v49);
            if ( *(_BYTE *)(v50 + 16) == 9 )
              v50 = v45;
            v76[v49++] = v50;
          }
          while ( v49 != v59 );
          v6 = v57;
        }
        v39 = sub_15A01B0(v76, (unsigned int)v77);
        if ( v76 != (__int64 *)s )
          _libc_free((unsigned __int64)v76);
        v40 = a2[16] - 24;
        goto LABEL_69;
      }
      v54 = v63;
      v55 = (__int64)v60;
    }
    else
    {
      v62 = v38 - 24;
      v42 = **(__int64 ****)(*(_QWORD *)v37 + 16LL);
      v45 = sub_15A14F0(v40, v42, v41 <= 0x10u);
      if ( v45 )
        goto LABEL_56;
      v54 = v62;
      v55 = (__int64)v42;
      if ( v41 > 0x10u )
      {
        v45 = sub_15A06D0(v42, (__int64)v42, v43, v44);
        goto LABEL_56;
      }
    }
    if ( v54 > 0x15 )
    {
      a3 = 1.0;
      v45 = sub_15A10B0(v55, 1.0);
    }
    else
    {
      v45 = sub_15A0680(v55, 1, 0);
    }
    goto LABEL_56;
  }
  v21 = 0;
  while ( *(int *)(v71[0] + 4 * v21) < 0 )
  {
LABEL_25:
    if ( v19 == ++v21 )
      goto LABEL_53;
  }
  v22 = sub_15A0A60(v8, v21);
  v23 = (_BYTE *)v71[0];
  v24 = &v73[*(int *)(v71[0] + 4 * v21)];
  v25 = *v24;
  if ( v22 && (*(_BYTE *)(v25 + 16) == 9 || v25 == v22) )
  {
    *v24 = v22;
    goto LABEL_25;
  }
  if ( v73 != (__int64 *)v75 )
  {
    _libc_free((unsigned __int64)v73);
    v23 = (_BYTE *)v71[0];
  }
  if ( v23 != v72 )
    _libc_free((unsigned __int64)v23);
  return 0;
}
