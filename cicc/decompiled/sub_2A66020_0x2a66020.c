// Function: sub_2A66020
// Address: 0x2a66020
//
__int64 (__fastcall *__fastcall sub_2A66020(__int64 a1))(__int64, __int64, __int64)
{
  __int64 v1; // rax
  __int64 v3; // r12
  __int64 v4; // r13
  unsigned __int64 v5; // rdi
  __int64 *v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rsi
  _QWORD *v11; // r12
  __int64 v12; // rsi
  _QWORD *v13; // r13
  unsigned __int64 v14; // r14
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 *v18; // r12
  unsigned __int64 v19; // r13
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  __int64 v22; // r14
  unsigned __int64 v23; // r12
  int v24; // eax
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  __int64 v27; // r14
  unsigned __int64 v28; // r12
  int v29; // eax
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // r12
  __int64 v34; // r13
  __int64 v35; // rax
  __int64 v36; // r12
  __int64 v37; // r13
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // r13
  __int64 (__fastcall *result)(__int64, __int64, __int64); // rax

  v1 = *(unsigned int *)(a1 + 2592);
  if ( (_DWORD)v1 )
  {
    v3 = *(_QWORD *)(a1 + 2576);
    v4 = v3 + 72 * v1;
    do
    {
      if ( *(_QWORD *)v3 != -8192 && *(_QWORD *)v3 != -4096 )
      {
        v5 = *(_QWORD *)(v3 + 40);
        if ( v5 != v3 + 56 )
          _libc_free(v5);
        sub_C7D6A0(*(_QWORD *)(v3 + 16), 8LL * *(unsigned int *)(v3 + 32), 8);
      }
      v3 += 72;
    }
    while ( v4 != v3 );
    v1 = *(unsigned int *)(a1 + 2592);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 2576), 72 * v1, 8);
  v10 = *(unsigned int *)(a1 + 2560);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD **)(a1 + 2544);
    v12 = 2 * v10;
    v13 = &v11[v12];
    do
    {
      if ( *v11 != -8192 && *v11 != -4096 )
      {
        v14 = v11[1];
        if ( v14 )
        {
          sub_2A45460(v11[1], v12 * 8, v6, v7, v8, v9);
          v12 = 75;
          j_j___libc_free_0(v14);
        }
      }
      v11 += 2;
    }
    while ( v13 != v11 );
    v10 = *(unsigned int *)(a1 + 2560);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 2544), 16 * v10, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 2512), 16LL * *(unsigned int *)(a1 + 2528), 8);
  v15 = *(_QWORD *)(a1 + 1976);
  if ( v15 != a1 + 1992 )
    _libc_free(v15);
  v16 = *(_QWORD *)(a1 + 1448);
  if ( v16 != a1 + 1464 )
    _libc_free(v16);
  v17 = *(_QWORD *)(a1 + 1368);
  if ( v17 )
  {
    v18 = *(unsigned __int64 **)(a1 + 1408);
    v19 = *(_QWORD *)(a1 + 1440) + 8LL;
    if ( v19 > (unsigned __int64)v18 )
    {
      do
      {
        v20 = *v18++;
        j_j___libc_free_0(v20);
      }
      while ( v19 > (unsigned __int64)v18 );
      v17 = *(_QWORD *)(a1 + 1368);
    }
    j_j___libc_free_0(v17);
  }
  v21 = *(_QWORD *)(a1 + 840);
  if ( v21 != a1 + 856 )
    _libc_free(v21);
  if ( !*(_BYTE *)(a1 + 708) )
    _libc_free(*(_QWORD *)(a1 + 688));
  if ( !*(_BYTE *)(a1 + 548) )
    _libc_free(*(_QWORD *)(a1 + 528));
  if ( !*(_BYTE *)(a1 + 388) )
    _libc_free(*(_QWORD *)(a1 + 368));
  sub_C7D6A0(*(_QWORD *)(a1 + 336), 8LL * *(unsigned int *)(a1 + 352), 8);
  v22 = *(_QWORD *)(a1 + 312);
  v23 = v22 + 56LL * *(unsigned int *)(a1 + 320);
  if ( v22 != v23 )
  {
    do
    {
      while ( 1 )
      {
        v24 = *(unsigned __int8 *)(v23 - 40);
        v23 -= 56LL;
        if ( (unsigned int)(v24 - 4) <= 1 )
        {
          if ( *(_DWORD *)(v23 + 48) > 0x40u )
          {
            v25 = *(_QWORD *)(v23 + 40);
            if ( v25 )
              j_j___libc_free_0_0(v25);
          }
          if ( *(_DWORD *)(v23 + 32) > 0x40u )
          {
            v26 = *(_QWORD *)(v23 + 24);
            if ( v26 )
              break;
          }
        }
        if ( v22 == v23 )
          goto LABEL_45;
      }
      j_j___libc_free_0_0(v26);
    }
    while ( v22 != v23 );
LABEL_45:
    v23 = *(_QWORD *)(a1 + 312);
  }
  if ( a1 + 328 != v23 )
    _libc_free(v23);
  sub_C7D6A0(*(_QWORD *)(a1 + 288), 24LL * *(unsigned int *)(a1 + 304), 8);
  v27 = *(_QWORD *)(a1 + 264);
  v28 = v27 + 48LL * *(unsigned int *)(a1 + 272);
  if ( v27 != v28 )
  {
    do
    {
      while ( 1 )
      {
        v29 = *(unsigned __int8 *)(v28 - 40);
        v28 -= 48LL;
        if ( (unsigned int)(v29 - 4) <= 1 )
        {
          if ( *(_DWORD *)(v28 + 40) > 0x40u )
          {
            v30 = *(_QWORD *)(v28 + 32);
            if ( v30 )
              j_j___libc_free_0_0(v30);
          }
          if ( *(_DWORD *)(v28 + 24) > 0x40u )
          {
            v31 = *(_QWORD *)(v28 + 16);
            if ( v31 )
              break;
          }
        }
        if ( v27 == v28 )
          goto LABEL_58;
      }
      j_j___libc_free_0_0(v31);
    }
    while ( v27 != v28 );
LABEL_58:
    v28 = *(_QWORD *)(a1 + 264);
  }
  if ( a1 + 280 != v28 )
    _libc_free(v28);
  sub_C7D6A0(*(_QWORD *)(a1 + 240), 16LL * *(unsigned int *)(a1 + 256), 8);
  v32 = *(unsigned int *)(a1 + 224);
  if ( (_DWORD)v32 )
  {
    v33 = *(_QWORD *)(a1 + 208);
    v34 = v33 + 48 * v32;
    do
    {
      if ( *(_QWORD *)v33 != -4096 && *(_QWORD *)v33 != -8192 )
        sub_22C0090((unsigned __int8 *)(v33 + 8));
      v33 += 48;
    }
    while ( v34 != v33 );
    v32 = *(unsigned int *)(a1 + 224);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 208), 48 * v32, 8);
  v35 = *(unsigned int *)(a1 + 192);
  if ( (_DWORD)v35 )
  {
    v36 = *(_QWORD *)(a1 + 176);
    v37 = v36 + 56 * v35;
    while ( 1 )
    {
      if ( *(_QWORD *)v36 == -4096 )
      {
        if ( *(_DWORD *)(v36 + 8) != -1 && (unsigned int)*(unsigned __int8 *)(v36 + 16) - 4 <= 1 )
          goto LABEL_76;
LABEL_72:
        v36 += 56;
        if ( v37 == v36 )
          goto LABEL_82;
      }
      else
      {
        if ( *(_QWORD *)v36 == -8192 && *(_DWORD *)(v36 + 8) == -2
          || (unsigned int)*(unsigned __int8 *)(v36 + 16) - 4 > 1 )
        {
          goto LABEL_72;
        }
LABEL_76:
        if ( *(_DWORD *)(v36 + 48) > 0x40u )
        {
          v38 = *(_QWORD *)(v36 + 40);
          if ( v38 )
            j_j___libc_free_0_0(v38);
        }
        if ( *(_DWORD *)(v36 + 32) <= 0x40u )
          goto LABEL_72;
        v39 = *(_QWORD *)(v36 + 24);
        if ( !v39 )
          goto LABEL_72;
        j_j___libc_free_0_0(v39);
        v36 += 56;
        if ( v37 == v36 )
        {
LABEL_82:
          v35 = *(unsigned int *)(a1 + 192);
          break;
        }
      }
    }
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 176), 56 * v35, 8);
  v40 = *(unsigned int *)(a1 + 160);
  if ( (_DWORD)v40 )
  {
    v41 = *(_QWORD *)(a1 + 144);
    v42 = v41 + 48 * v40;
    do
    {
      if ( *(_QWORD *)v41 != -4096 && *(_QWORD *)v41 != -8192 )
        sub_22C0090((unsigned __int8 *)(v41 + 8));
      v41 += 48;
    }
    while ( v42 != v41 );
    v40 = *(unsigned int *)(a1 + 160);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 144), 48 * v40, 8);
  if ( *(_BYTE *)(a1 + 68) )
  {
    result = *(__int64 (__fastcall **)(__int64, __int64, __int64))(a1 + 24);
    if ( !result )
      return result;
    return (__int64 (__fastcall *)(__int64, __int64, __int64))result(a1 + 8, a1 + 8, 3);
  }
  _libc_free(*(_QWORD *)(a1 + 48));
  result = *(__int64 (__fastcall **)(__int64, __int64, __int64))(a1 + 24);
  if ( result )
    return (__int64 (__fastcall *)(__int64, __int64, __int64))result(a1 + 8, a1 + 8, 3);
  return result;
}
