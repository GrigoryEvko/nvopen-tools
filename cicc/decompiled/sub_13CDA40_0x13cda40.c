// Function: sub_13CDA40
// Address: 0x13cda40
//
__int64 __fastcall sub_13CDA40(unsigned __int8 *a1, _QWORD *a2)
{
  unsigned __int8 *v2; // r12
  unsigned __int8 v3; // al
  __int64 v4; // rax
  bool v5; // al
  __int64 v6; // rdi
  unsigned __int8 v7; // al
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned __int8 v10; // al
  __int64 v11; // rax
  bool v12; // al
  unsigned __int8 v14; // al
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // r13
  int v25; // r14d
  unsigned int v26; // r15d
  __int64 v27; // rax
  __int64 v28; // r13
  char v29; // al
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // r12
  __int64 v33; // r12
  int v34; // r14d
  unsigned int v35; // r15d
  __int64 v36; // rax
  __int64 v37; // r13
  char v38; // al
  __int64 v39; // r13
  unsigned int v40; // r15d
  int v41; // r14d
  __int64 v42; // rax
  __int64 v43; // r13
  char v44; // al
  __int64 v45; // r13
  int v46; // r13d
  unsigned int v47; // r14d
  __int64 v48; // rax
  __int64 v49; // r12
  char v50; // al
  __int64 v51; // r12

  v2 = a1;
  v3 = a1[16];
  if ( v3 == 14 )
  {
    if ( *((_QWORD *)a1 + 4) == sub_16982C0() )
      v4 = *((_QWORD *)a1 + 5) + 8LL;
    else
      v4 = (__int64)(a1 + 32);
    v5 = (*(_BYTE *)(v4 + 18) & 7) == 1;
LABEL_5:
    if ( !v5 )
      goto LABEL_8;
LABEL_6:
    v6 = *(_QWORD *)v2;
    goto LABEL_7;
  }
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 16 || v3 > 0x10u )
    goto LABEL_8;
  v16 = sub_15A1020(a1);
  v17 = v16;
  if ( v16 && *(_BYTE *)(v16 + 16) == 14 )
  {
    if ( *(_QWORD *)(v16 + 32) == sub_16982C0() )
      v18 = *(_QWORD *)(v17 + 40) + 8LL;
    else
      v18 = v17 + 32;
    v5 = (*(_BYTE *)(v18 + 18) & 7) == 1;
    goto LABEL_5;
  }
  v6 = *(_QWORD *)a1;
  v25 = *(_QWORD *)(*(_QWORD *)v2 + 32LL);
  if ( v25 )
  {
    v26 = 0;
    while ( 1 )
    {
      v27 = sub_15A0A60(v2, v26);
      v28 = v27;
      if ( !v27 )
        goto LABEL_8;
      v29 = *(_BYTE *)(v27 + 16);
      if ( v29 != 9 )
      {
        if ( v29 != 14 )
          goto LABEL_8;
        v30 = *(_QWORD *)(v28 + 32) == sub_16982C0() ? *(_QWORD *)(v28 + 40) + 8LL : v28 + 32;
        if ( (*(_BYTE *)(v30 + 18) & 7) != 1 )
          goto LABEL_8;
      }
      if ( v25 == ++v26 )
        goto LABEL_6;
    }
  }
LABEL_7:
  if ( (unsigned int)sub_16431D0(v6) <= 0x20 )
    return 0;
LABEL_8:
  v7 = *((_BYTE *)a2 + 16);
  if ( v7 == 14 )
  {
    if ( a2[4] == sub_16982C0() )
      v8 = a2[5] + 8LL;
    else
      v8 = (__int64)(a2 + 4);
    if ( (*(_BYTE *)(v8 + 18) & 7) != 1 )
      goto LABEL_14;
LABEL_12:
    v9 = *a2;
    goto LABEL_13;
  }
  if ( *(_BYTE *)(*a2 + 8LL) != 16 || v7 > 0x10u )
    goto LABEL_14;
  v19 = sub_15A1020(a2);
  v20 = v19;
  if ( v19 && *(_BYTE *)(v19 + 16) == 14 )
  {
    if ( *(_QWORD *)(v19 + 32) == sub_16982C0() )
      v21 = *(_QWORD *)(v20 + 40) + 8LL;
    else
      v21 = v20 + 32;
    if ( (*(_BYTE *)(v21 + 18) & 7) != 1 )
      goto LABEL_14;
    goto LABEL_12;
  }
  v9 = *a2;
  v34 = *(_QWORD *)(*a2 + 32LL);
  if ( v34 )
  {
    v35 = 0;
    while ( 1 )
    {
      v36 = sub_15A0A60(a2, v35);
      v37 = v36;
      if ( !v36 )
        goto LABEL_14;
      v38 = *(_BYTE *)(v36 + 16);
      if ( v38 != 9 )
      {
        if ( v38 != 14 )
          goto LABEL_14;
        v39 = *(_QWORD *)(v37 + 32) == sub_16982C0() ? *(_QWORD *)(v37 + 40) + 8LL : v37 + 32;
        if ( (*(_BYTE *)(v39 + 18) & 7) != 1 )
          goto LABEL_14;
      }
      if ( v34 == ++v35 )
        goto LABEL_12;
    }
  }
LABEL_13:
  if ( (unsigned int)sub_16431D0(v9) <= 0x20 )
    return 0;
LABEL_14:
  v10 = v2[16];
  if ( v10 == 14 )
  {
    if ( *((_QWORD *)v2 + 4) == sub_16982C0() )
      v11 = *((_QWORD *)v2 + 5) + 8LL;
    else
      v11 = (__int64)(v2 + 32);
    v12 = (*(_BYTE *)(v11 + 18) & 7) == 1;
    goto LABEL_18;
  }
  if ( *(_BYTE *)(*(_QWORD *)v2 + 8LL) != 16 || v10 > 0x10u )
  {
LABEL_24:
    v14 = *((_BYTE *)a2 + 16);
    if ( v14 == 14 )
    {
      if ( a2[4] == sub_16982C0() )
        v15 = a2[5] + 8LL;
      else
        v15 = (__int64)(a2 + 4);
      if ( (*(_BYTE *)(v15 + 18) & 7) == 1 )
        goto LABEL_28;
    }
    else if ( *(_BYTE *)(*a2 + 8LL) == 16 && v14 <= 0x10u )
    {
      v31 = sub_15A1020(a2);
      v32 = v31;
      if ( v31 && *(_BYTE *)(v31 + 16) == 14 )
      {
        if ( *(_QWORD *)(v31 + 32) == sub_16982C0() )
          v33 = *(_QWORD *)(v32 + 40) + 8LL;
        else
          v33 = v32 + 32;
        if ( (*(_BYTE *)(v33 + 18) & 7) == 1 )
          goto LABEL_28;
      }
      else
      {
        v46 = *(_QWORD *)(*a2 + 32LL);
        if ( !v46 )
        {
LABEL_28:
          v2 = (unsigned __int8 *)a2;
          if ( (unsigned __int8)sub_15A0E50(a2) )
            return (__int64)v2;
          return sub_15A11D0(*a2, 0, 0);
        }
        v47 = 0;
        while ( 1 )
        {
          v48 = sub_15A0A60(a2, v47);
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
            goto LABEL_28;
        }
      }
    }
    return 0;
  }
  v22 = sub_15A1020(v2);
  v23 = v22;
  if ( !v22 || *(_BYTE *)(v22 + 16) != 14 )
  {
    v40 = 0;
    v41 = *(_QWORD *)(*(_QWORD *)v2 + 32LL);
    if ( !v41 )
      goto LABEL_19;
    while ( 1 )
    {
      v42 = sub_15A0A60(v2, v40);
      v43 = v42;
      if ( !v42 )
        goto LABEL_24;
      v44 = *(_BYTE *)(v42 + 16);
      if ( v44 != 9 )
      {
        if ( v44 != 14 )
          goto LABEL_24;
        v45 = *(_QWORD *)(v43 + 32) == sub_16982C0() ? *(_QWORD *)(v43 + 40) + 8LL : v43 + 32;
        if ( (*(_BYTE *)(v45 + 18) & 7) != 1 )
          goto LABEL_24;
      }
      if ( v41 == ++v40 )
        goto LABEL_19;
    }
  }
  if ( *(_QWORD *)(v22 + 32) == sub_16982C0() )
    v24 = *(_QWORD *)(v23 + 40) + 8LL;
  else
    v24 = v23 + 32;
  v12 = (*(_BYTE *)(v24 + 18) & 7) == 1;
LABEL_18:
  if ( !v12 )
    goto LABEL_24;
LABEL_19:
  if ( (unsigned __int8)sub_15A0E50(v2) )
    return (__int64)v2;
  return sub_15A11D0(*(_QWORD *)v2, 0, 0);
}
