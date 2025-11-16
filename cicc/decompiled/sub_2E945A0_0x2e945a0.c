// Function: sub_2E945A0
// Address: 0x2e945a0
//
__int64 __fastcall sub_2E945A0(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // r8
  unsigned __int64 v4; // r15
  __int64 v5; // rbx
  unsigned __int64 v6; // r14
  _DWORD *v7; // rdx
  unsigned __int64 v8; // rcx
  _DWORD *v9; // rax
  unsigned __int64 v11; // rax
  bool v12; // cf
  unsigned __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // rdi
  _DWORD *v16; // rdx
  _DWORD *v17; // rax
  _DWORD *v18; // r12
  _DWORD *v19; // r10
  _DWORD *v20; // r11
  _DWORD *v21; // rbx
  char v22; // al
  __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rax
  const void *v26; // rsi
  size_t v27; // rdx
  unsigned __int64 v28; // rbx
  unsigned __int64 v29; // r12
  __int64 v30; // rax
  unsigned __int64 v31; // [rsp+8h] [rbp-68h]
  _DWORD *v32; // [rsp+10h] [rbp-60h]
  __int64 v33; // [rsp+18h] [rbp-58h]
  int v34; // [rsp+20h] [rbp-50h]
  _DWORD *v35; // [rsp+28h] [rbp-48h]
  unsigned __int64 v36; // [rsp+30h] [rbp-40h]
  unsigned __int64 v37; // [rsp+30h] [rbp-40h]
  unsigned __int64 v38; // [rsp+30h] [rbp-40h]
  unsigned __int64 v39; // [rsp+30h] [rbp-40h]
  unsigned __int64 v40; // [rsp+38h] [rbp-38h]

  v3 = a1[1];
  v4 = *a1;
  v40 = a2;
  v5 = v3 - *a1;
  v6 = 0xCCCCCCCCCCCCCCCDLL * (v5 >> 4);
  if ( a2 <= 0xCCCCCCCCCCCCCCCDLL * ((__int64)(a1[2] - v3) >> 4) )
  {
    v7 = (_DWORD *)(v3 + 80);
    v8 = a1[1];
    do
    {
      if ( v8 )
      {
        *(_QWORD *)v8 = 0;
        v9 = (_DWORD *)(v8 + 16);
        *(_DWORD *)(v8 + 8) = 1;
        *(_DWORD *)(v8 + 12) = 0;
        do
        {
          if ( v9 )
            *v9 = -1;
          v9 += 4;
        }
        while ( v9 != v7 );
      }
      v8 += 80LL;
      v7 += 20;
      --a2;
    }
    while ( a2 );
    a1[1] = 80 * v40 + v3;
    return 80 * v40;
  }
  if ( 0x199999999999999LL - v6 < a2 )
    sub_4262D8((__int64)"vector::_M_default_append");
  v11 = a2;
  if ( a2 < v6 )
    v11 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v3 - v4) >> 4);
  v12 = __CFADD__(v6, v11);
  v13 = v6 + v11;
  if ( v12 )
  {
    v29 = 0x7FFFFFFFFFFFFFD0LL;
  }
  else
  {
    if ( !v13 )
    {
      v31 = 0;
      v33 = 0;
      goto LABEL_16;
    }
    if ( v13 > 0x199999999999999LL )
      v13 = 0x199999999999999LL;
    v29 = 80 * v13;
  }
  v30 = sub_22077B0(v29);
  v3 = a1[1];
  v4 = *a1;
  v33 = v30;
  v31 = v30 + v29;
LABEL_16:
  v14 = (_QWORD *)(v5 + v33);
  v15 = v5 + v33 + 80;
  v16 = (_DWORD *)v15;
  while ( 1 )
  {
    if ( v14 )
    {
      *v14 = 0;
      v17 = v14 + 2;
      *(v17 - 2) = 1;
      *(v17 - 1) = 0;
      do
      {
        if ( v17 )
          *v17 = -1;
        v17 += 4;
      }
      while ( v17 != v16 );
    }
    v14 = (_QWORD *)v15;
    v16 += 20;
    if ( !--a2 )
      break;
    v15 += 80;
  }
  if ( v4 != v3 )
  {
    v18 = (_DWORD *)(v33 + 80);
    do
    {
      v19 = v18 - 20;
      if ( v18 != (_DWORD *)80 )
      {
        *((_QWORD *)v18 - 10) = 0;
        v20 = v18 - 16;
        v19[2] = 1;
        v21 = v18 - 16;
        *(v18 - 17) = 0;
        do
        {
          if ( v21 )
            *v21 = -1;
          v21 += 4;
        }
        while ( v18 != v21 );
        if ( (v19[2] & 1) == 0 )
        {
          v36 = v3;
          sub_C7D6A0(*((_QWORD *)v18 - 8), 16LL * (unsigned int)*(v18 - 14), 8);
          v20 = v18 - 16;
          v19 = v18 - 20;
          v3 = v36;
        }
        v22 = *((_BYTE *)v19 + 8) | 1;
        *((_BYTE *)v19 + 8) = v22;
        if ( (*(_BYTE *)(v4 + 8) & 1) == 0 && *(_DWORD *)(v4 + 24) > 4u )
        {
          *((_BYTE *)v19 + 8) = v22 & 0xFE;
          if ( (*(_BYTE *)(v4 + 8) & 1) != 0 )
          {
            v24 = 64;
            LODWORD(v23) = 4;
          }
          else
          {
            v23 = *(unsigned int *)(v4 + 24);
            v24 = 16 * v23;
          }
          v32 = v20;
          v34 = v23;
          v35 = v19;
          v37 = v3;
          v25 = sub_C7D670(v24, 8);
          v20 = v32;
          v19 = v35;
          v3 = v37;
          *((_QWORD *)v21 - 8) = v25;
          *(v21 - 14) = v34;
        }
        v19[2] = *(_DWORD *)(v4 + 8) & 0xFFFFFFFE | v19[2] & 1;
        *(v21 - 17) = *(_DWORD *)(v4 + 12);
        if ( (v19[2] & 1) == 0 )
          v20 = (_DWORD *)*((_QWORD *)v21 - 8);
        v26 = (const void *)(v4 + 16);
        if ( (*(_BYTE *)(v4 + 8) & 1) == 0 )
          v26 = *(const void **)(v4 + 16);
        v27 = 64;
        if ( (v19[2] & 1) == 0 )
          v27 = 16LL * (unsigned int)*(v21 - 14);
        v38 = v3;
        memcpy(v20, v26, v27);
        v3 = v38;
      }
      v4 += 80LL;
      v18 += 20;
    }
    while ( v4 != v3 );
    v28 = a1[1];
    v3 = *a1;
    if ( v28 != *a1 )
    {
      do
      {
        if ( (*(_BYTE *)(v3 + 8) & 1) == 0 )
        {
          v39 = v3;
          sub_C7D6A0(*(_QWORD *)(v3 + 16), 16LL * *(unsigned int *)(v3 + 24), 8);
          v3 = v39;
        }
        v3 += 80LL;
      }
      while ( v28 != v3 );
      v3 = *a1;
    }
  }
  if ( v3 )
    j_j___libc_free_0(v3);
  *a1 = v33;
  a1[1] = v33 + 80 * (v6 + v40);
  a1[2] = v31;
  return v31;
}
