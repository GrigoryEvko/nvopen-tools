// Function: sub_1D03130
// Address: 0x1d03130
//
bool __fastcall sub_1D03130(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // r14d
  char v7; // r8
  char v8; // r9
  unsigned int v9; // r15d
  unsigned int v10; // ecx
  unsigned int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rsi
  unsigned int v14; // edx
  unsigned int v15; // esi
  int v16; // eax
  unsigned int v17; // eax
  bool v18; // cf
  __int64 *v19; // rdx
  __int64 *v20; // rsi
  __int64 *v21; // r9
  __int64 *v22; // rdi
  unsigned int v23; // r8d
  __int64 *v24; // rdx
  unsigned int v25; // esi
  bool result; // al
  unsigned __int8 v27; // dl
  unsigned __int8 v28; // al
  unsigned int v29; // edx
  unsigned int v30; // esi
  bool v31; // cc
  int v32; // r12d
  unsigned int v33; // r12d
  int v34; // r12d
  unsigned int v35; // r12d
  int v36; // eax
  unsigned int v37; // [rsp+Ch] [rbp-34h]

  if ( !byte_4FC1020 )
  {
    v27 = (*(_BYTE *)(a1 + 228) & 0x40) != 0;
    v28 = (*(_BYTE *)(a2 + 228) & 0x40) != 0;
    v18 = v27 < v28;
    if ( v27 != v28 )
      return v18;
  }
  v6 = sub_1D01080(a3, (unsigned int *)a1);
  v9 = sub_1D01080(a3, (unsigned int *)a2);
  if ( (v7 & 2) != 0 && (v8 & 4) != 0 )
  {
    v10 = 0;
    v11 = *(_DWORD *)(*(_QWORD *)a2 + 60LL);
    if ( v9 > v11 )
      v10 = v9 - v11;
    v9 = v10;
  }
  if ( (v8 & 2) != 0 )
  {
    if ( (v7 & 4) != 0 )
    {
      v12 = *(_QWORD *)a1;
      v29 = *(_DWORD *)(*(_QWORD *)a1 + 60LL);
      v30 = v6 - v29;
      v31 = v6 <= v29;
      v6 = 0;
      if ( !v31 )
        v6 = v30;
      if ( v9 == v6 )
      {
        v13 = *(_QWORD *)a2;
        goto LABEL_11;
      }
    }
    else if ( v9 == v6 )
    {
      goto LABEL_10;
    }
    return v9 < v6;
  }
  if ( v9 != v6 )
    return v9 < v6;
  if ( (v7 & 2) == 0 )
    goto LABEL_15;
LABEL_10:
  v12 = *(_QWORD *)a1;
  v13 = *(_QWORD *)a2;
  if ( !*(_QWORD *)a1 )
  {
    if ( !v13 )
      goto LABEL_15;
    v14 = 0;
    goto LABEL_12;
  }
LABEL_11:
  v14 = *(_DWORD *)(v12 + 64);
  if ( !v13 )
  {
    v16 = *(_DWORD *)(v12 + 64);
    v15 = 0;
    goto LABEL_13;
  }
LABEL_12:
  v15 = *(_DWORD *)(v13 + 64);
  v16 = v14 | v15;
LABEL_13:
  if ( !v16 || v15 == v14 )
  {
LABEL_15:
    v37 = sub_1D00F50(a1);
    v17 = sub_1D00F50(a2);
    v18 = v37 < v17;
    if ( v37 != v17 )
      return v18;
    v19 = *(__int64 **)(a1 + 32);
    v20 = *(__int64 **)(a2 + 32);
    v21 = &v19[2 * *(unsigned int *)(a1 + 40)];
    v22 = &v20[2 * *(unsigned int *)(a2 + 40)];
    if ( v19 == v21 )
    {
      if ( v20 == v22 )
      {
LABEL_23:
        if ( (!v9 || (*(_BYTE *)(a1 + 228) & 2) == 0) && (!v6 || (*(_BYTE *)(a2 + 228) & 2) == 0) )
        {
          if ( (unsigned __int8)byte_4FC13A0 | ((*(_BYTE *)(a2 + 228) & 2) != 0) || (*(_BYTE *)(a1 + 228) & 2) != 0 )
          {
            if ( (*(_BYTE *)(a1 + 236) & 2) == 0 )
              sub_1F01F70(a1);
            v32 = *(_DWORD *)(a1 + 244);
            if ( (*(_BYTE *)(a2 + 236) & 2) == 0 )
              sub_1F01F70(a2);
            if ( *(_DWORD *)(a2 + 244) != v32 )
            {
              if ( (*(_BYTE *)(a1 + 236) & 2) == 0 )
                sub_1F01F70(a1);
              v33 = *(_DWORD *)(a1 + 244);
              if ( (*(_BYTE *)(a2 + 236) & 2) == 0 )
                sub_1F01F70(a2);
              return *(_DWORD *)(a2 + 244) < v33;
            }
            if ( (*(_BYTE *)(a1 + 236) & 1) == 0 )
              sub_1F01DD0(a1);
            v34 = *(_DWORD *)(a1 + 240);
            if ( (*(_BYTE *)(a2 + 236) & 1) == 0 )
              sub_1F01DD0(a2);
            if ( *(_DWORD *)(a2 + 240) != v34 )
            {
              if ( (*(_BYTE *)(a1 + 236) & 1) == 0 )
                sub_1F01DD0(a1);
              v35 = *(_DWORD *)(a1 + 240);
              if ( (*(_BYTE *)(a2 + 236) & 1) == 0 )
                sub_1F01DD0(a2);
              return *(_DWORD *)(a2 + 240) > v35;
            }
          }
          else
          {
            v36 = sub_1D02CA0(a1, a2, 0, a3);
            if ( v36 )
              return v36 > 0;
          }
        }
        return *(_DWORD *)(a1 + 196) > *(_DWORD *)(a2 + 196);
      }
      v23 = 0;
    }
    else
    {
      v23 = 0;
      do
      {
        v23 += ((*v19 >> 1) & 3) == 0;
        v19 += 2;
      }
      while ( v21 != v19 );
      if ( v20 == v22 )
      {
        v25 = 0;
        goto LABEL_22;
      }
    }
    v24 = *(__int64 **)(a2 + 32);
    v25 = 0;
    do
    {
      v25 += ((*v24 >> 1) & 3) == 0;
      v24 += 2;
    }
    while ( v22 != v24 );
LABEL_22:
    v18 = v25 < v23;
    if ( v25 == v23 )
      goto LABEL_23;
    return v18;
  }
  result = 0;
  if ( v14 )
    return v15 == 0 || v15 > v14;
  return result;
}
