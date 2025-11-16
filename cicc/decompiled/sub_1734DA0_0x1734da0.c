// Function: sub_1734DA0
// Address: 0x1734da0
//
char __fastcall sub_1734DA0(__int64 **a1, unsigned __int64 a2)
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
  __int64 v16; // r8
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rdx
  char v22; // cl
  __int64 v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rdx
  __int64 v27; // rdi
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // r8
  __int64 v32; // rdi
  __int64 v33; // rdx
  unsigned int v34; // ebx
  __int64 v35; // rax
  unsigned int v36; // r12d
  bool v37; // al
  __int64 v38; // rdx
  __int64 v39; // rdx
  unsigned int v40; // r15d
  int v41; // r14d
  __int64 v42; // rax
  char v43; // cl
  unsigned int v44; // esi
  bool v45; // al
  unsigned int v46; // [rsp+Ch] [rbp-34h]

  v3 = a2;
  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 == 52 )
  {
    v6 = *(_QWORD *)(a2 - 48);
    v7 = *(_QWORD *)(a2 - 24);
    v8 = *(_BYTE *)(v6 + 16);
    v9 = *(_BYTE *)(v7 + 16);
    if ( v8 == 50 )
    {
      a2 = *(_QWORD *)(v6 - 48);
      v28 = *(_QWORD *)(v6 - 24);
      v29 = **a1;
      if ( (a2 != v29 || *a1[1] != v28) && (v29 != v28 || a2 != *a1[1]) )
        goto LABEL_8;
    }
    else
    {
      if ( v8 != 5 || *(_WORD *)(v6 + 18) != 26 )
        goto LABEL_8;
      v31 = *(_QWORD *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
      v32 = **a1;
      a2 = 1LL - (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
      v29 = 3 * a2;
      v33 = *(_QWORD *)(v6 + 24 * a2);
      if ( v31 != v32 || (v29 = (__int64)a1[1], *(_QWORD *)v29 != v33) )
      {
        if ( v32 != v33 || v31 != *a1[1] )
          goto LABEL_8;
      }
    }
    if ( v9 == 13 )
    {
      v34 = *(_DWORD *)(v7 + 32);
      if ( v34 <= 0x40 )
        return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v34) == *(_QWORD *)(v7 + 24);
      else
        return v34 == (unsigned int)sub_16A58F0(v7 + 24);
    }
    if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) == 16 && v9 <= 0x10u )
    {
      v35 = sub_15A1020((_BYTE *)v7, a2, *(_QWORD *)v7, v29);
      if ( v35 && *(_BYTE *)(v35 + 16) == 13 )
      {
        v36 = *(_DWORD *)(v35 + 32);
        if ( v36 <= 0x40 )
          v37 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v36) == *(_QWORD *)(v35 + 24);
        else
          v37 = v36 == (unsigned int)sub_16A58F0(v35 + 24);
        if ( v37 )
          return 1;
      }
      else
      {
        v40 = 0;
        v41 = *(_QWORD *)(*(_QWORD *)v7 + 32LL);
        if ( !v41 )
          return 1;
        while ( 1 )
        {
          a2 = v40;
          v42 = sub_15A0A60(v7, v40);
          if ( !v42 )
            break;
          v43 = *(_BYTE *)(v42 + 16);
          if ( v43 != 9 )
          {
            if ( v43 != 13 )
              break;
            v44 = *(_DWORD *)(v42 + 32);
            if ( v44 <= 0x40 )
            {
              a2 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v44);
              v45 = a2 == *(_QWORD *)(v42 + 24);
            }
            else
            {
              v46 = *(_DWORD *)(v42 + 32);
              a2 = v46;
              v45 = v46 == (unsigned int)sub_16A58F0(v42 + 24);
            }
            if ( !v45 )
              break;
          }
          if ( v41 == ++v40 )
            return 1;
        }
      }
      v7 = *(_QWORD *)(v3 - 24);
      v9 = *(_BYTE *)(v7 + 16);
    }
LABEL_8:
    if ( v9 == 50 )
    {
      v11 = *(_QWORD *)(v7 - 48);
      v10 = *(_QWORD *)(v7 - 24);
      v30 = **a1;
      if ( v11 != v30 || (a2 = (unsigned __int64)a1[1], *(_QWORD *)a2 != v10) )
      {
        if ( v30 != v10 || v11 != *a1[1] )
          return 0;
      }
    }
    else
    {
      if ( v9 != 5 || *(_WORD *)(v7 + 18) != 26 )
        return 0;
      a2 = *(_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
      v10 = **a1;
      v11 = 1LL - (*(_DWORD *)(v7 + 20) & 0xFFFFFFF);
      v12 = *(_QWORD *)(v7 + 24 * v11);
      if ( a2 != v10 || (v11 = (__int64)a1[1], *(_QWORD *)v11 != v12) )
      {
        if ( v10 != v12 || a2 != *a1[1] )
          return 0;
      }
    }
    return sub_17279D0(*(_BYTE **)(v3 - 48), a2, v11, v10);
  }
  if ( v4 != 5 || *(_WORD *)(a2 + 18) != 28 )
    return 0;
  v13 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v14 = *(_QWORD *)(a2 - 24 * v13);
  v15 = *(_BYTE *)(v14 + 16);
  if ( v15 == 50 )
  {
    v18 = *(_QWORD *)(v14 - 48);
    v38 = *(_QWORD *)(v14 - 24);
    v19 = **a1;
    if ( (v18 != v19 || v38 != *a1[1]) && (v38 != v19 || v18 != *a1[1]) )
      goto LABEL_23;
  }
  else
  {
    if ( v15 != 5 || *(_WORD *)(v14 + 18) != 26 )
      goto LABEL_23;
    v16 = *(_QWORD *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
    v17 = **a1;
    v18 = 1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
    v19 = 3 * v18;
    v20 = *(_QWORD *)(v14 + 24 * v18);
    if ( v16 != v17 || (v19 = (__int64)a1[1], *(_QWORD *)v19 != v20) )
    {
      if ( v17 != v20 || v16 != *a1[1] )
        goto LABEL_23;
    }
  }
  if ( sub_1727B40(*(_BYTE **)(v3 + 24 * (1 - v13)), v18, 1 - v13, v19) )
    return 1;
  v13 = *(_DWORD *)(v3 + 20) & 0xFFFFFFF;
LABEL_23:
  v21 = *(_QWORD *)(v3 + 24 * (1 - v13));
  v22 = *(_BYTE *)(v21 + 16);
  if ( v22 == 50 )
  {
    v23 = *(_QWORD *)(v21 - 48);
    v39 = *(_QWORD *)(v21 - 24);
    v24 = **a1;
    if ( (v23 != v24 || *a1[1] != v39) && (v24 != v39 || v23 != *a1[1]) )
      return 0;
  }
  else
  {
    if ( v22 != 5 || *(_WORD *)(v21 + 18) != 26 )
      return 0;
    v23 = 1LL - (*(_DWORD *)(v21 + 20) & 0xFFFFFFF);
    v24 = 3 * v23;
    v25 = *(_QWORD *)(v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF));
    v26 = *(_QWORD *)(v21 + 24 * v23);
    v27 = **a1;
    if ( v25 != v27 || (v24 = (__int64)a1[1], *(_QWORD *)v24 != v26) )
    {
      if ( v27 != v26 || v25 != *a1[1] )
        return 0;
    }
  }
  return sub_1727B40(*(_BYTE **)(v3 - 24 * v13), v23, 4 * v13, v24);
}
