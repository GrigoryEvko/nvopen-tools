// Function: sub_1CD2540
// Address: 0x1cd2540
//
unsigned __int64 __fastcall sub_1CD2540(__int64 a1)
{
  int v1; // ecx
  __int64 v2; // rdx
  _QWORD *v3; // rax
  _QWORD *i; // rdx
  int v5; // eax
  __int64 v6; // rdx
  _QWORD *v7; // rax
  _QWORD *j; // rdx
  int v9; // eax
  __int64 v10; // rdx
  _QWORD *v11; // rax
  _QWORD *k; // rdx
  int v13; // eax
  unsigned __int64 result; // rax
  __int64 v15; // rdx
  unsigned __int64 m; // rdx
  unsigned int v17; // ecx
  unsigned int v18; // eax
  int v19; // r14d
  unsigned int v20; // ecx
  unsigned int v21; // eax
  int v22; // r14d
  unsigned int v23; // eax
  unsigned int v24; // ecx
  unsigned int v25; // eax
  int v26; // r14d
  unsigned int v27; // eax
  _QWORD *v28; // rax
  _QWORD *v29; // r12
  _QWORD *v30; // rbx
  __int64 v31; // r14
  unsigned int v32; // eax
  unsigned int v33; // eax
  int v34; // r14d
  unsigned int v35; // eax

  v1 = *(_DWORD *)(a1 + 232);
  if ( !v1 )
  {
    ++*(_QWORD *)(a1 + 216);
LABEL_3:
    if ( !*(_DWORD *)(a1 + 236) )
      goto LABEL_8;
    v2 = *(unsigned int *)(a1 + 240);
    if ( (unsigned int)v2 <= 0x40 )
      goto LABEL_5;
    j___libc_free_0(*(_QWORD *)(a1 + 224));
    *(_DWORD *)(a1 + 240) = 0;
LABEL_89:
    *(_QWORD *)(a1 + 224) = 0;
LABEL_7:
    *(_QWORD *)(a1 + 232) = 0;
    goto LABEL_8;
  }
  v28 = *(_QWORD **)(a1 + 224);
  v29 = &v28[2 * *(unsigned int *)(a1 + 240)];
  if ( v28 == v29 )
    goto LABEL_61;
  while ( 1 )
  {
    v30 = v28;
    if ( *v28 != -8 && *v28 != -16 )
      break;
    v28 += 2;
    if ( v29 == v28 )
      goto LABEL_61;
  }
  if ( v29 == v28 )
  {
LABEL_61:
    ++*(_QWORD *)(a1 + 216);
  }
  else
  {
    do
    {
      v31 = v30[1];
      if ( v31 )
      {
        j___libc_free_0(*(_QWORD *)(v31 + 8));
        j_j___libc_free_0(v31, 32);
      }
      v30 += 2;
      if ( v30 == v29 )
        break;
      while ( *v30 == -8 || *v30 == -16 )
      {
        v30 += 2;
        if ( v29 == v30 )
          goto LABEL_69;
      }
    }
    while ( v29 != v30 );
LABEL_69:
    v1 = *(_DWORD *)(a1 + 232);
    ++*(_QWORD *)(a1 + 216);
    if ( !v1 )
      goto LABEL_3;
  }
  v32 = 4 * v1;
  v2 = *(unsigned int *)(a1 + 240);
  if ( (unsigned int)(4 * v1) < 0x40 )
    v32 = 64;
  if ( (unsigned int)v2 <= v32 )
  {
LABEL_5:
    v3 = *(_QWORD **)(a1 + 224);
    for ( i = &v3[2 * v2]; i != v3; v3 += 2 )
      *v3 = -8;
    goto LABEL_7;
  }
  if ( v1 == 1 )
  {
    v34 = 64;
  }
  else
  {
    _BitScanReverse(&v33, v1 - 1);
    v34 = 1 << (33 - (v33 ^ 0x1F));
    if ( v34 < 64 )
      v34 = 64;
    if ( (_DWORD)v2 == v34 )
      goto LABEL_79;
  }
  j___libc_free_0(*(_QWORD *)(a1 + 224));
  v35 = sub_1CD0670(v34);
  *(_DWORD *)(a1 + 240) = v35;
  if ( !v35 )
    goto LABEL_89;
  *(_QWORD *)(a1 + 224) = sub_22077B0(16LL * v35);
LABEL_79:
  sub_1CD2480(a1 + 216);
LABEL_8:
  v5 = *(_DWORD *)(a1 + 296);
  ++*(_QWORD *)(a1 + 280);
  if ( !v5 )
  {
    if ( !*(_DWORD *)(a1 + 300) )
      goto LABEL_14;
    v6 = *(unsigned int *)(a1 + 304);
    if ( (unsigned int)v6 <= 0x40 )
      goto LABEL_11;
    j___libc_free_0(*(_QWORD *)(a1 + 288));
    *(_DWORD *)(a1 + 304) = 0;
LABEL_85:
    *(_QWORD *)(a1 + 288) = 0;
LABEL_13:
    *(_QWORD *)(a1 + 296) = 0;
    goto LABEL_14;
  }
  v24 = 4 * v5;
  v6 = *(unsigned int *)(a1 + 304);
  if ( (unsigned int)(4 * v5) < 0x40 )
    v24 = 64;
  if ( v24 >= (unsigned int)v6 )
  {
LABEL_11:
    v7 = *(_QWORD **)(a1 + 288);
    for ( j = &v7[2 * v6]; j != v7; v7 += 2 )
      *v7 = -8;
    goto LABEL_13;
  }
  v25 = v5 - 1;
  if ( v25 )
  {
    _BitScanReverse(&v25, v25);
    v26 = 1 << (33 - (v25 ^ 0x1F));
    if ( v26 < 64 )
      v26 = 64;
    if ( (_DWORD)v6 == v26 )
      goto LABEL_56;
  }
  else
  {
    v26 = 64;
  }
  j___libc_free_0(*(_QWORD *)(a1 + 288));
  v27 = sub_1CD0670(v26);
  *(_DWORD *)(a1 + 304) = v27;
  if ( !v27 )
    goto LABEL_85;
  *(_QWORD *)(a1 + 288) = sub_22077B0(16LL * v27);
LABEL_56:
  sub_1CD24C0(a1 + 280);
LABEL_14:
  v9 = *(_DWORD *)(a1 + 328);
  ++*(_QWORD *)(a1 + 312);
  if ( !v9 )
  {
    if ( !*(_DWORD *)(a1 + 332) )
      goto LABEL_20;
    v10 = *(unsigned int *)(a1 + 336);
    if ( (unsigned int)v10 <= 0x40 )
      goto LABEL_17;
    j___libc_free_0(*(_QWORD *)(a1 + 320));
    *(_DWORD *)(a1 + 336) = 0;
LABEL_83:
    *(_QWORD *)(a1 + 320) = 0;
LABEL_19:
    *(_QWORD *)(a1 + 328) = 0;
    goto LABEL_20;
  }
  v20 = 4 * v9;
  v10 = *(unsigned int *)(a1 + 336);
  if ( (unsigned int)(4 * v9) < 0x40 )
    v20 = 64;
  if ( (unsigned int)v10 <= v20 )
  {
LABEL_17:
    v11 = *(_QWORD **)(a1 + 320);
    for ( k = &v11[2 * v10]; k != v11; v11 += 2 )
      *v11 = -8;
    goto LABEL_19;
  }
  v21 = v9 - 1;
  if ( v21 )
  {
    _BitScanReverse(&v21, v21);
    v22 = 1 << (33 - (v21 ^ 0x1F));
    if ( v22 < 64 )
      v22 = 64;
    if ( (_DWORD)v10 == v22 )
      goto LABEL_46;
  }
  else
  {
    v22 = 64;
  }
  j___libc_free_0(*(_QWORD *)(a1 + 320));
  v23 = sub_1CD0670(v22);
  *(_DWORD *)(a1 + 336) = v23;
  if ( !v23 )
    goto LABEL_83;
  *(_QWORD *)(a1 + 320) = sub_22077B0(16LL * v23);
LABEL_46:
  sub_1CD24C0(a1 + 312);
LABEL_20:
  v13 = *(_DWORD *)(a1 + 360);
  ++*(_QWORD *)(a1 + 344);
  if ( v13 )
  {
    v17 = 4 * v13;
    v15 = *(unsigned int *)(a1 + 368);
    if ( (unsigned int)(4 * v13) < 0x40 )
      v17 = 64;
    if ( v17 >= (unsigned int)v15 )
    {
LABEL_23:
      result = *(_QWORD *)(a1 + 352);
      for ( m = result + 16 * v15; m != result; result += 16LL )
        *(_QWORD *)result = -8;
      goto LABEL_25;
    }
    v18 = v13 - 1;
    if ( v18 )
    {
      _BitScanReverse(&v18, v18);
      v19 = 1 << (33 - (v18 ^ 0x1F));
      if ( v19 < 64 )
        v19 = 64;
      if ( v19 == (_DWORD)v15 )
        return (unsigned __int64)sub_1CD2500(a1 + 344);
    }
    else
    {
      v19 = 64;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 352));
    result = sub_1CD0670(v19);
    *(_DWORD *)(a1 + 368) = result;
    if ( !(_DWORD)result )
      goto LABEL_87;
    *(_QWORD *)(a1 + 352) = sub_22077B0(16LL * (unsigned int)result);
    return (unsigned __int64)sub_1CD2500(a1 + 344);
  }
  result = *(unsigned int *)(a1 + 364);
  if ( !(_DWORD)result )
    return result;
  v15 = *(unsigned int *)(a1 + 368);
  if ( (unsigned int)v15 <= 0x40 )
    goto LABEL_23;
  result = j___libc_free_0(*(_QWORD *)(a1 + 352));
  *(_DWORD *)(a1 + 368) = 0;
LABEL_87:
  *(_QWORD *)(a1 + 352) = 0;
LABEL_25:
  *(_QWORD *)(a1 + 360) = 0;
  return result;
}
