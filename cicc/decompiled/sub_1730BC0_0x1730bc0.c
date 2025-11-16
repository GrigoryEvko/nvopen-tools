// Function: sub_1730BC0
// Address: 0x1730bc0
//
char __fastcall sub_1730BC0(__int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  char v4; // al
  __int64 v6; // rdx
  __int64 v7; // r12
  char v8; // cl
  unsigned __int8 v9; // al
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  char v15; // cl
  __int64 v16; // rdi
  __int64 v17; // r8
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rdx
  char v22; // cl
  __int64 v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rdi
  __int64 v30; // r8
  __int64 v31; // rdx
  unsigned int v32; // ebx
  __int64 v33; // rax
  unsigned int v34; // r12d
  bool v35; // al
  __int64 v36; // rdx
  __int64 v37; // rdx
  unsigned int v38; // r15d
  int v39; // r14d
  __int64 v40; // rax
  char v41; // cl
  unsigned int v42; // esi
  bool v43; // al
  unsigned int v44; // [rsp+Ch] [rbp-34h]

  v3 = a2;
  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 == 52 )
  {
    v6 = *(_QWORD *)(a2 - 48);
    v7 = *(_QWORD *)(a2 - 24);
    v8 = *(_BYTE *)(v6 + 16);
    v9 = *(_BYTE *)(v7 + 16);
    if ( v8 == 51 )
    {
      a2 = *(_QWORD *)(v6 - 48);
      v27 = *a1;
      v28 = *(_QWORD *)(v6 - 24);
      if ( (a2 != *a1 || a1[1] != v28) && (v27 != v28 || a2 != a1[1]) )
        goto LABEL_8;
    }
    else
    {
      if ( v8 != 5 || *(_WORD *)(v6 + 18) != 27 )
        goto LABEL_8;
      v29 = *a1;
      v30 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
      a2 = 1LL - (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
      v27 = 3 * a2;
      v31 = *(_QWORD *)(v6 + 24 * a2);
      if ( (v30 != *a1 || a1[1] != v31) && (v29 != v31 || v30 != a1[1]) )
        goto LABEL_8;
    }
    if ( v9 == 13 )
    {
      v32 = *(_DWORD *)(v7 + 32);
      if ( v32 <= 0x40 )
        return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v32) == *(_QWORD *)(v7 + 24);
      else
        return v32 == (unsigned int)sub_16A58F0(v7 + 24);
    }
    if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) == 16 && v9 <= 0x10u )
    {
      v33 = sub_15A1020((_BYTE *)v7, a2, *(_QWORD *)v7, v27);
      if ( v33 && *(_BYTE *)(v33 + 16) == 13 )
      {
        v34 = *(_DWORD *)(v33 + 32);
        if ( v34 <= 0x40 )
          v35 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v34) == *(_QWORD *)(v33 + 24);
        else
          v35 = v34 == (unsigned int)sub_16A58F0(v33 + 24);
        if ( v35 )
          return 1;
      }
      else
      {
        v38 = 0;
        v39 = *(_QWORD *)(*(_QWORD *)v7 + 32LL);
        if ( !v39 )
          return 1;
        while ( 1 )
        {
          a2 = v38;
          v40 = sub_15A0A60(v7, v38);
          if ( !v40 )
            break;
          v41 = *(_BYTE *)(v40 + 16);
          if ( v41 != 9 )
          {
            if ( v41 != 13 )
              break;
            v42 = *(_DWORD *)(v40 + 32);
            if ( v42 <= 0x40 )
            {
              a2 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v42);
              v43 = a2 == *(_QWORD *)(v40 + 24);
            }
            else
            {
              v44 = *(_DWORD *)(v40 + 32);
              a2 = v44;
              v43 = v44 == (unsigned int)sub_16A58F0(v40 + 24);
            }
            if ( !v43 )
              break;
          }
          if ( v39 == ++v38 )
            return 1;
        }
      }
      v7 = *(_QWORD *)(v3 - 24);
      v9 = *(_BYTE *)(v7 + 16);
    }
LABEL_8:
    if ( v9 == 51 )
    {
      v11 = *(_QWORD *)(v7 - 48);
      v10 = *(_QWORD *)(v7 - 24);
      if ( (v11 != *a1 || a1[1] != v10) && (*a1 != v10 || v11 != a1[1]) )
        return 0;
    }
    else
    {
      if ( v9 != 5 || *(_WORD *)(v7 + 18) != 27 )
        return 0;
      v10 = *a1;
      a2 = *(_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
      v11 = 1LL - (*(_DWORD *)(v7 + 20) & 0xFFFFFFF);
      v12 = *(_QWORD *)(v7 + 24 * v11);
      if ( (a2 != *a1 || a1[1] != v12) && (v10 != v12 || a2 != a1[1]) )
        return 0;
    }
    return sub_17279D0(*(_BYTE **)(v3 - 48), a2, v11, v10);
  }
  if ( v4 != 5 || *(_WORD *)(a2 + 18) != 28 )
    return 0;
  v13 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v14 = *(_QWORD *)(a2 - 24 * v13);
  v15 = *(_BYTE *)(v14 + 16);
  if ( v15 == 51 )
  {
    v18 = *(_QWORD *)(v14 - 48);
    v19 = *a1;
    v36 = *(_QWORD *)(v14 - 24);
    if ( (v18 != *a1 || v36 != a1[1]) && (v36 != v19 || v18 != a1[1]) )
      goto LABEL_23;
  }
  else
  {
    if ( v15 != 5 || *(_WORD *)(v14 + 18) != 27 )
      goto LABEL_23;
    v16 = *a1;
    v17 = *(_QWORD *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
    v18 = 1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
    v19 = 3 * v18;
    v20 = *(_QWORD *)(v14 + 24 * v18);
    if ( (v17 != *a1 || a1[1] != v20) && (v16 != v20 || v17 != a1[1]) )
      goto LABEL_23;
  }
  if ( sub_1727B40(*(_BYTE **)(v3 + 24 * (1 - v13)), v18, 1 - v13, v19) )
    return 1;
  v13 = *(_DWORD *)(v3 + 20) & 0xFFFFFFF;
LABEL_23:
  v21 = *(_QWORD *)(v3 + 24 * (1 - v13));
  v22 = *(_BYTE *)(v21 + 16);
  if ( v22 == 51 )
  {
    v23 = *(_QWORD *)(v21 - 48);
    v24 = *a1;
    v37 = *(_QWORD *)(v21 - 24);
    if ( (v23 != *a1 || a1[1] != v37) && (v24 != v37 || v23 != a1[1]) )
      return 0;
  }
  else
  {
    if ( v22 != 5 || *(_WORD *)(v21 + 18) != 27 )
      return 0;
    v23 = 1LL - (*(_DWORD *)(v21 + 20) & 0xFFFFFFF);
    v24 = 3 * v23;
    v25 = *(_QWORD *)(v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF));
    v26 = *(_QWORD *)(v21 + 24 * v23);
    if ( (v25 != *a1 || a1[1] != v26) && (*a1 != v26 || v25 != a1[1]) )
      return 0;
  }
  return sub_1727B40(*(_BYTE **)(v3 - 24 * v13), v23, 4 * v13, v24);
}
