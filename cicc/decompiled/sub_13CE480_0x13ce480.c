// Function: sub_13CE480
// Address: 0x13ce480
//
__int64 __fastcall sub_13CE480(__int64 a1, _QWORD *a2, __int64 a3, __int64 *a4)
{
  unsigned int v5; // eax
  _QWORD *v6; // rdx
  __int64 v7; // r13
  unsigned int v9; // ebx
  bool v10; // al
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned int v18; // eax
  __int64 *v19; // rsi
  unsigned __int64 v20; // rsi
  __int64 v21; // r12
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rax
  __int16 v25; // dx
  __int64 v26; // rcx
  __int64 v27; // r13
  __int64 v28; // rdi
  unsigned __int8 v29; // al
  __int64 v30; // rax
  bool v31; // al
  unsigned __int8 v32; // al
  __int64 v33; // r12
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // rbx
  __int64 v37; // rax
  __int64 v38; // rbx
  __int64 v39; // rbx
  unsigned int v40; // r14d
  int v41; // r13d
  __int64 v42; // rax
  __int64 v43; // rbx
  char v44; // al
  __int64 v45; // rbx
  int v46; // r13d
  unsigned int v47; // r14d
  __int64 v48; // rax
  __int64 v49; // rbx
  char v50; // al
  __int64 v51; // rbx
  __int64 v52; // [rsp+0h] [rbp-70h]
  __int64 v53; // [rsp+8h] [rbp-68h]
  __int64 v54; // [rsp+10h] [rbp-60h] BYREF
  __int64 v55; // [rsp+18h] [rbp-58h] BYREF
  __int64 v56; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v57; // [rsp+28h] [rbp-48h]
  __int64 v58; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v59; // [rsp+38h] [rbp-38h]

  v5 = *(_DWORD *)(a1 + 36);
  v6 = *(_QWORD **)(*(_QWORD *)(a1 + 24) + 16LL);
  v7 = *v6;
  if ( v5 > 0xD3 )
    return 0;
  if ( v5 > 0xBC )
  {
    switch ( v5 )
    {
      case 0xBDu:
      case 0xD1u:
        goto LABEL_14;
      case 0xC3u:
      case 0xD2u:
        if ( !sub_13CD190((__int64)a2) && !sub_13CD190(a3) && *((_BYTE *)a2 + 16) != 9 && *(_BYTE *)(a3 + 16) != 9 )
          return 0;
        return sub_15A06D0(v7);
      case 0xC6u:
      case 0xD3u:
        if ( a2 == (_QWORD *)a3 )
          return sub_15A06D0(v7);
LABEL_14:
        if ( *((_BYTE *)a2 + 16) == 9 || *(_BYTE *)(a3 + 16) == 9 )
          return sub_1599EF0(*v6);
        break;
      default:
        return 0;
    }
    return 0;
  }
  if ( v5 == 139 )
    goto LABEL_40;
  if ( v5 <= 0x8B )
  {
    if ( v5 == 118 )
    {
      if ( *((_BYTE *)a2 + 16) > 0x10u || *(_BYTE *)(a3 + 16) > 0x10u )
        return 0;
      v12 = *a4;
      v57 = 1;
      v56 = 0;
      if ( !(unsigned __int8)sub_14D5D40(a2, &v54, &v56, v12) )
        goto LABEL_63;
      v13 = sub_16498A0(a2);
      v52 = sub_16471D0(v13, 0);
      v14 = sub_16498A0(a2);
      v15 = sub_1643350(v14);
      v53 = sub_1647190(v15, 0);
      v16 = sub_16498A0(a2);
      v17 = sub_1643360(v16);
      if ( *(_BYTE *)(a3 + 16) != 13 || *(_DWORD *)(*(_QWORD *)a3 + 8LL) > 0x40FFu )
        goto LABEL_63;
      v18 = *(_DWORD *)(a3 + 32);
      v19 = *(__int64 **)(a3 + 24);
      v20 = v18 > 0x40 ? *v19 : (__int64)((_QWORD)v19 << (64 - (unsigned __int8)v18)) >> (64 - (unsigned __int8)v18);
      if ( (v20 & 3) != 0 )
        goto LABEL_63;
      v21 = sub_15A0680(v17, v20 >> 2, 0);
      v22 = sub_15A4510(a2, v53, 0);
      v55 = v21;
      BYTE4(v58) = 0;
      v23 = sub_15A2E80(v15, v22, (unsigned int)&v55, 1, 0, (unsigned int)&v58, 0);
      v24 = sub_14D8290(v23, v15, v12);
      if ( !v24 || *(_BYTE *)(v24 + 16) != 5 )
        goto LABEL_63;
      v25 = *(_WORD *)(v24 + 18);
      if ( v25 == 36 )
      {
        v24 = *(_QWORD *)(v24 - 24LL * (*(_DWORD *)(v24 + 20) & 0xFFFFFFF));
        if ( *(_BYTE *)(v24 + 16) != 5 )
          goto LABEL_63;
        v25 = *(_WORD *)(v24 + 18);
      }
      if ( v25 == 13 )
      {
        v26 = *(_QWORD *)(v24 - 24LL * (*(_DWORD *)(v24 + 20) & 0xFFFFFFF));
        if ( *(_BYTE *)(v26 + 16) == 5 && *(_WORD *)(v26 + 18) == 45 )
        {
          v27 = *(_QWORD *)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
          a3 = 0;
          v28 = *(_QWORD *)(v24 + 24 * (1LL - (*(_DWORD *)(v24 + 20) & 0xFFFFFFF)));
          v59 = 1;
          v58 = 0;
          if ( !(unsigned __int8)sub_14D5D40(v28, &v55, &v58, v12) || v54 != v55 )
          {
LABEL_35:
            if ( v59 > 0x40 && v58 )
              j_j___libc_free_0_0(v58);
            goto LABEL_64;
          }
          if ( v57 <= 0x40 )
          {
            if ( v56 != v58 )
              goto LABEL_35;
          }
          else if ( !(unsigned __int8)sub_16A5220(&v56, &v58) )
          {
            goto LABEL_35;
          }
          a3 = sub_15A4510(v27, v52, 0);
          goto LABEL_35;
        }
      }
LABEL_63:
      a3 = 0;
LABEL_64:
      if ( v57 > 0x40 && v56 )
        j_j___libc_free_0_0(v56);
      return a3;
    }
    if ( v5 != 132 )
      return 0;
LABEL_40:
    v29 = *((_BYTE *)a2 + 16);
    if ( v29 == 14 )
    {
      if ( a2[4] == sub_16982C0() )
        v30 = a2[5] + 8LL;
      else
        v30 = (__int64)(a2 + 4);
      v31 = (*(_BYTE *)(v30 + 18) & 7) == 1;
    }
    else
    {
      if ( *(_BYTE *)(*a2 + 8LL) != 16 || v29 > 0x10u )
        goto LABEL_45;
      v34 = sub_15A1020(a2);
      v35 = v34;
      if ( !v34 || *(_BYTE *)(v34 + 16) != 14 )
      {
        v40 = 0;
        v41 = *(_QWORD *)(*a2 + 32LL);
        if ( !v41 )
          return a3;
        while ( 1 )
        {
          v42 = sub_15A0A60(a2, v40);
          v43 = v42;
          if ( !v42 )
            break;
          v44 = *(_BYTE *)(v42 + 16);
          if ( v44 != 9 )
          {
            if ( v44 != 14 )
              break;
            v45 = *(_QWORD *)(v43 + 32) == sub_16982C0() ? *(_QWORD *)(v43 + 40) + 8LL : v43 + 32;
            if ( (*(_BYTE *)(v45 + 18) & 7) != 1 )
              break;
          }
          if ( v41 == ++v40 )
            return a3;
        }
        goto LABEL_45;
      }
      if ( *(_QWORD *)(v34 + 32) == sub_16982C0() )
        v36 = *(_QWORD *)(v35 + 40) + 8LL;
      else
        v36 = v35 + 32;
      v31 = (*(_BYTE *)(v36 + 18) & 7) == 1;
    }
    if ( v31 )
      return a3;
LABEL_45:
    v32 = *(_BYTE *)(a3 + 16);
    if ( v32 == 14 )
    {
      if ( *(_QWORD *)(a3 + 32) == sub_16982C0() )
        v33 = *(_QWORD *)(a3 + 40) + 8LL;
      else
        v33 = a3 + 32;
      if ( (*(_BYTE *)(v33 + 18) & 7) == 1 )
        return (__int64)a2;
    }
    else if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 && v32 <= 0x10u )
    {
      v37 = sub_15A1020(a3);
      v38 = v37;
      if ( v37 && *(_BYTE *)(v37 + 16) == 14 )
      {
        if ( *(_QWORD *)(v37 + 32) == sub_16982C0() )
          v39 = *(_QWORD *)(v38 + 40) + 8LL;
        else
          v39 = v38 + 32;
        if ( (*(_BYTE *)(v39 + 18) & 7) == 1 )
          return (__int64)a2;
      }
      else
      {
        v46 = *(_QWORD *)(*(_QWORD *)a3 + 32LL);
        if ( !v46 )
          return (__int64)a2;
        v47 = 0;
        while ( 1 )
        {
          v48 = sub_15A0A60(a3, v47);
          v49 = v48;
          if ( !v48 )
            break;
          v50 = *(_BYTE *)(v48 + 16);
          if ( v50 != 9 )
          {
            if ( v50 != 14 )
              break;
            v51 = *(_QWORD *)(v49 + 32) == sub_16982C0() ? *(_QWORD *)(v49 + 40) + 8LL : v49 + 32;
            if ( (*(_BYTE *)(v51 + 18) & 7) != 1 )
              break;
          }
          if ( v46 == ++v47 )
            return (__int64)a2;
        }
      }
    }
    return 0;
  }
  if ( v5 != 147 || *(_BYTE *)(a3 + 16) != 13 )
    return 0;
  v9 = *(_DWORD *)(a3 + 32);
  if ( v9 <= 0x40 )
    v10 = *(_QWORD *)(a3 + 24) == 0;
  else
    v10 = v9 == (unsigned int)sub_16A57B0(a3 + 24);
  if ( !v10 )
  {
    if ( v9 <= 0x40 )
    {
      if ( *(_QWORD *)(a3 + 24) == 1 )
        return (__int64)a2;
    }
    else if ( (unsigned int)sub_16A57B0(a3 + 24) == v9 - 1 )
    {
      return (__int64)a2;
    }
    return 0;
  }
  return sub_15A10B0(*a2, 1.0);
}
