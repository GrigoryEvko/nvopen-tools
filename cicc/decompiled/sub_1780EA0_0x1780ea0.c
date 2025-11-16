// Function: sub_1780EA0
// Address: 0x1780ea0
//
unsigned __int8 *__fastcall sub_1780EA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        double a5,
        double a6,
        double a7)
{
  _QWORD *v8; // r12
  char v10; // r13
  __int64 v11; // r12
  __int64 v13; // rax
  __int64 v14; // r10
  unsigned int v15; // r13d
  int v16; // eax
  bool v17; // al
  __int64 v18; // rdi
  char v19; // r13
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rcx
  unsigned __int64 v24; // rdx
  __int64 v25; // rcx
  char v26; // al
  __int64 v27; // rax
  char v28; // al
  __int64 v29; // rax
  _BYTE *v30; // r14
  __int64 v31; // r15
  unsigned __int8 v32; // al
  int v33; // eax
  bool v34; // al
  __int64 v35; // r13
  __int64 v36; // r15
  __int64 v37; // rax
  __int64 v38; // r13
  __int64 v39; // rax
  __int64 v40; // rdx
  char v41; // dl
  __int64 v42; // rdx
  unsigned int v43; // r13d
  __int64 v44; // rax
  char v45; // dl
  int v46; // eax
  __int64 v47; // [rsp+0h] [rbp-70h]
  __int64 v48; // [rsp+8h] [rbp-68h]
  int v49; // [rsp+8h] [rbp-68h]
  __int64 v50; // [rsp+10h] [rbp-60h]
  int v51; // [rsp+10h] [rbp-60h]
  __int64 v52; // [rsp+10h] [rbp-60h]
  int v53; // [rsp+10h] [rbp-60h]
  __int64 v54; // [rsp+18h] [rbp-58h]
  __int64 v55; // [rsp+18h] [rbp-58h]
  __int64 v56; // [rsp+18h] [rbp-58h]
  __int64 v57; // [rsp+18h] [rbp-58h]
  __int64 v58[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v59; // [rsp+30h] [rbp-40h]

  v8 = (_QWORD *)a2;
  v10 = *(_BYTE *)(a1 + 16);
  if ( v10 == 48 )
  {
    v18 = *(_QWORD *)(a1 - 48);
    v27 = *(_QWORD *)(v18 + 8);
    if ( !v27 || *(_QWORD *)(v27 + 8) )
      goto LABEL_16;
    v28 = *(_BYTE *)(v18 + 16);
    if ( v28 == 47 )
    {
      v30 = *(_BYTE **)(v18 - 48);
      if ( !v30 )
        goto LABEL_16;
      v31 = *(_QWORD *)(v18 - 24);
      if ( !v31 )
        goto LABEL_16;
    }
    else
    {
      if ( v28 != 5 )
        goto LABEL_16;
      if ( *(_WORD *)(v18 + 18) != 23 )
        goto LABEL_16;
      v29 = *(_DWORD *)(v18 + 20) & 0xFFFFFFF;
      a4 = 4 * v29;
      v30 = *(_BYTE **)(v18 - 24 * v29);
      if ( !v30 )
        goto LABEL_16;
      v31 = *(_QWORD *)(v18 + 24 * (1 - v29));
      if ( !v31 )
        goto LABEL_16;
    }
    v14 = *(_QWORD *)(a1 - 24);
    if ( !v14 )
      goto LABEL_16;
  }
  else
  {
    if ( v10 != 5 )
      goto LABEL_14;
    if ( *(_WORD *)(a1 + 18) != 24 )
      return 0;
    a4 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v40 = *(_QWORD *)(a4 + 8);
    if ( !v40 || *(_QWORD *)(v40 + 8) )
      return 0;
    v41 = *(_BYTE *)(a4 + 16);
    if ( v41 == 47 )
    {
      v30 = *(_BYTE **)(a4 - 48);
      if ( !v30 )
        return 0;
      v31 = *(_QWORD *)(a4 - 24);
      if ( !v31 )
        return 0;
    }
    else
    {
      if ( v41 != 5 )
        return 0;
      if ( *(_WORD *)(a4 + 18) != 23 )
        return 0;
      v42 = *(_DWORD *)(a4 + 20) & 0xFFFFFFF;
      v30 = *(_BYTE **)(a4 - 24 * v42);
      if ( !v30 )
        return 0;
      a2 = 1 - v42;
      v31 = *(_QWORD *)(a4 + 24 * (1 - v42));
      if ( !v31 )
        return 0;
    }
    v14 = *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
    if ( !v14 )
      return 0;
  }
  v32 = v30[16];
  if ( v32 == 13 )
  {
    if ( *((_DWORD *)v30 + 8) <= 0x40u )
    {
      v34 = *((_QWORD *)v30 + 3) == 1;
    }
    else
    {
      v48 = a3;
      v51 = *((_DWORD *)v30 + 8);
      v56 = v14;
      v33 = sub_16A57B0((__int64)(v30 + 24));
      v14 = v56;
      a3 = v48;
      v34 = v51 - 1 == v33;
    }
    if ( v34 )
      goto LABEL_42;
    goto LABEL_14;
  }
  if ( *(_BYTE *)(*(_QWORD *)v30 + 8LL) != 16 || v32 > 0x10u )
  {
LABEL_14:
    if ( (unsigned __int8)(v10 - 47) > 1u )
      return 0;
    v18 = *(_QWORD *)(a1 - 48);
LABEL_16:
    v55 = a3;
    v19 = sub_14BDFF0(v18, v8[333], 0, 0, v8[330], a3, v8[332]);
    if ( !v19 )
      return 0;
    v20 = *(_QWORD *)(a1 - 48);
    v21 = *(_QWORD *)(v20 + 8);
    if ( v21 && !*(_QWORD *)(v21 + 8) && (v22 = sub_1780EA0(v20, v8, v55)) != 0 )
    {
      if ( *(_QWORD *)(a1 - 48) )
      {
        v23 = *(_QWORD *)(a1 - 40);
        v24 = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v24 = v23;
        if ( v23 )
          *(_QWORD *)(v23 + 16) = *(_QWORD *)(v23 + 16) & 3LL | v24;
      }
      *(_QWORD *)(a1 - 48) = v22;
      v25 = *(_QWORD *)(v22 + 8);
      *(_QWORD *)(a1 - 40) = v25;
      if ( v25 )
        *(_QWORD *)(v25 + 16) = (a1 - 40) | *(_QWORD *)(v25 + 16) & 3LL;
      *(_QWORD *)(a1 - 32) = *(_QWORD *)(a1 - 32) & 3LL | (v22 + 8);
      *(_QWORD *)(v22 + 8) = a1 - 48;
    }
    else
    {
      v19 = 0;
    }
    v26 = *(_BYTE *)(a1 + 16);
    if ( v26 == 48 )
    {
      if ( !sub_15F23D0(a1) )
      {
        sub_15F2350(a1, 1);
        if ( *(_BYTE *)(a1 + 16) != 47 || sub_15F2370(a1) )
          return (unsigned __int8 *)a1;
        goto LABEL_79;
      }
      v26 = *(_BYTE *)(a1 + 16);
    }
    if ( v26 != 47 || sub_15F2370(a1) )
    {
      if ( !v19 )
        return 0;
      return (unsigned __int8 *)a1;
    }
LABEL_79:
    sub_15F2310(a1, 1);
    return (unsigned __int8 *)a1;
  }
  v50 = a3;
  v54 = v14;
  v13 = sub_15A1020(v30, a2, *(_QWORD *)v30, a4);
  v14 = v54;
  a3 = v50;
  if ( v13 && *(_BYTE *)(v13 + 16) == 13 )
  {
    v15 = *(_DWORD *)(v13 + 32);
    if ( v15 <= 0x40 )
    {
      v17 = *(_QWORD *)(v13 + 24) == 1;
    }
    else
    {
      v16 = sub_16A57B0(v13 + 24);
      v14 = v54;
      a3 = v50;
      v17 = v15 - 1 == v16;
    }
    if ( v17 )
      goto LABEL_42;
    goto LABEL_13;
  }
  v49 = *(_QWORD *)(*(_QWORD *)v30 + 32LL);
  if ( v49 )
  {
    v43 = 0;
    while ( 1 )
    {
      v52 = a3;
      v57 = v14;
      v44 = sub_15A0A60((__int64)v30, v43);
      v14 = v57;
      a3 = v52;
      if ( !v44 )
        break;
      v45 = *(_BYTE *)(v44 + 16);
      if ( v45 != 9 )
      {
        if ( v45 != 13 )
          break;
        if ( *(_DWORD *)(v44 + 32) <= 0x40u )
        {
          if ( *(_QWORD *)(v44 + 24) != 1 )
            break;
        }
        else
        {
          v47 = v52;
          v53 = *(_DWORD *)(v44 + 32);
          v46 = sub_16A57B0(v44 + 24);
          v14 = v57;
          a3 = v47;
          if ( v46 != v53 - 1 )
            break;
        }
      }
      if ( v49 == ++v43 )
        goto LABEL_42;
    }
LABEL_13:
    v10 = *(_BYTE *)(a1 + 16);
    goto LABEL_14;
  }
LABEL_42:
  v35 = v8[1];
  v59 = 257;
  if ( *(_BYTE *)(v31 + 16) > 0x10u || *(_BYTE *)(v14 + 16) > 0x10u )
  {
    v36 = (__int64)sub_170A2B0(v35, 13, (__int64 *)v31, v14, v58, 0, 0);
  }
  else
  {
    v36 = sub_15A2B60((__int64 *)v31, v14, 0, 0, a5, a6, a7);
    v37 = sub_14DBA30(v36, *(_QWORD *)(v35 + 96), 0);
    if ( v37 )
      v36 = v37;
  }
  v38 = v8[1];
  v59 = 257;
  if ( v30[16] > 0x10u || *(_BYTE *)(v36 + 16) > 0x10u )
    return sub_170A2B0(v38, 23, (__int64 *)v30, v36, v58, 0, 0);
  v11 = sub_15A2D50((__int64 *)v30, v36, 0, 0, a5, a6, a7);
  v39 = sub_14DBA30(v11, *(_QWORD *)(v38 + 96), 0);
  if ( v39 )
    return (unsigned __int8 *)v39;
  return (unsigned __int8 *)v11;
}
