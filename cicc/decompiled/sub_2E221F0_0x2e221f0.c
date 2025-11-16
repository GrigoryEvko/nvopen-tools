// Function: sub_2E221F0
// Address: 0x2e221f0
//
void __fastcall sub_2E221F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  _QWORD *v7; // rax
  __int64 v10; // rdx
  _QWORD *v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rdx
  _QWORD *v14; // rdx
  __int64 v15; // rax
  unsigned int v16; // ecx
  unsigned int v17; // eax
  unsigned int v18; // r15d
  __int64 v19; // rdx
  _BYTE *v20; // rdi
  int v21; // ecx
  unsigned int *v22; // r10
  unsigned int *i; // r8
  unsigned int v24; // ecx
  __int64 v25; // rsi
  unsigned int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rsi
  _QWORD *v29; // rdx
  __int64 v30; // rcx
  unsigned int *v31; // r8
  unsigned int *j; // r10
  unsigned int v33; // ecx
  __int64 v34; // rsi
  int v35; // ecx
  unsigned __int64 v36; // rcx
  unsigned __int64 v37; // r8
  int v38; // eax
  __int64 v39; // r13
  __int64 v40; // [rsp+8h] [rbp-88h]
  __int64 v41; // [rsp+10h] [rbp-80h] BYREF
  _BYTE *v42; // [rsp+18h] [rbp-78h] BYREF
  __int64 v43; // [rsp+20h] [rbp-70h]
  _BYTE s[48]; // [rsp+28h] [rbp-68h] BYREF
  unsigned int v45; // [rsp+58h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 48);
  if ( !*(_BYTE *)(v6 + 120) )
    return;
  v7 = (_QWORD *)a1[1];
  v10 = 8LL * *((unsigned int *)a1 + 4);
  v11 = &v7[(unsigned __int64)v10 / 8];
  v12 = v10 >> 3;
  v13 = v10 >> 5;
  if ( !v13 )
  {
LABEL_28:
    if ( v12 != 2 )
    {
      if ( v12 != 3 )
      {
        if ( v12 != 1 )
          goto LABEL_31;
        goto LABEL_48;
      }
      if ( *v7 )
        goto LABEL_9;
      ++v7;
    }
    if ( *v7 )
      goto LABEL_9;
    ++v7;
LABEL_48:
    if ( !*v7 )
      goto LABEL_31;
    goto LABEL_9;
  }
  v14 = &v7[4 * v13];
  while ( !*v7 )
  {
    if ( v7[1] )
    {
      if ( v11 != v7 + 1 )
        goto LABEL_10;
      goto LABEL_31;
    }
    if ( v7[2] )
    {
      if ( v11 != v7 + 2 )
        goto LABEL_10;
      goto LABEL_31;
    }
    if ( v7[3] )
    {
      if ( v11 != v7 + 3 )
        goto LABEL_10;
      goto LABEL_31;
    }
    v7 += 4;
    if ( v14 == v7 )
    {
      v12 = v11 - v7;
      goto LABEL_28;
    }
  }
LABEL_9:
  if ( v11 != v7 )
  {
LABEL_10:
    v15 = *a1;
    v45 = 0;
    v42 = s;
    v16 = *(_DWORD *)(v15 + 44);
    v41 = v15;
    v43 = 0x600000000LL;
    v45 = v16;
    v17 = (v16 + 63) >> 6;
    v18 = v17;
    if ( v17 )
    {
      v19 = v17;
      v20 = s;
      if ( v17 > 6 )
      {
        v40 = v17;
        sub_C8D5F0((__int64)&v42, s, v17, 8u, a5, a6);
        v19 = v40;
        v20 = &v42[8 * (unsigned int)v43];
      }
      memset(v20, 0, 8 * v19);
      LODWORD(v43) = v18 + v43;
      LOBYTE(v16) = v45;
    }
    v21 = v16 & 0x3F;
    if ( v21 )
      *(_QWORD *)&v42[8 * (unsigned int)v43 - 8] &= ~(-1LL << v21);
    sub_2E21C60(&v41, *(_QWORD *)(a2 + 32), *(_QWORD *)(a2 + 48));
    v22 = *(unsigned int **)(v6 + 104);
    for ( i = *(unsigned int **)(v6 + 96); v22 != i; i += 3 )
    {
      v24 = *(_DWORD *)(*(_QWORD *)(v41 + 8) + 24LL * *i + 16) & 0xFFF;
      v25 = *(_QWORD *)(v41 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v41 + 8) + 24LL * *i + 16) >> 12);
      do
      {
        if ( !v25 )
          break;
        v25 += 2;
        *(_QWORD *)&v42[8 * (v24 >> 6)] &= ~(1LL << v24);
        v24 += *(__int16 *)(v25 - 2);
      }
      while ( *(_WORD *)(v25 - 2) );
    }
    v26 = v45;
    if ( *((_DWORD *)a1 + 18) < v45 )
    {
      v35 = a1[9] & 0x3F;
      if ( v35 )
        *(_QWORD *)(a1[1] + 8LL * *((unsigned int *)a1 + 4) - 8) &= ~(-1LL << v35);
      v36 = *((unsigned int *)a1 + 4);
      *((_DWORD *)a1 + 18) = v26;
      v37 = (v26 + 63) >> 6;
      if ( v37 != v36 )
      {
        if ( v37 >= v36 )
        {
          v39 = v37 - v36;
          if ( v37 > *((unsigned int *)a1 + 5) )
          {
            sub_C8D5F0((__int64)(a1 + 1), a1 + 3, v37, 8u, v37, 1);
            v36 = *((unsigned int *)a1 + 4);
          }
          if ( 8 * v39 )
          {
            memset((void *)(a1[1] + 8 * v36), 0, 8 * v39);
            LODWORD(v36) = *((_DWORD *)a1 + 4);
          }
          v26 = *((_DWORD *)a1 + 18);
          *((_DWORD *)a1 + 4) = v39 + v36;
        }
        else
        {
          *((_DWORD *)a1 + 4) = (v26 + 63) >> 6;
        }
      }
      v38 = v26 & 0x3F;
      if ( v38 )
        *(_QWORD *)(a1[1] + 8LL * *((unsigned int *)a1 + 4) - 8) &= ~(-1LL << v38);
    }
    v27 = 0;
    v28 = 8LL * (unsigned int)v43;
    if ( (_DWORD)v43 )
    {
      do
      {
        v29 = (_QWORD *)(v27 + a1[1]);
        v30 = *(_QWORD *)&v42[v27];
        v27 += 8;
        *v29 |= v30;
      }
      while ( v28 != v27 );
    }
    if ( v42 != s )
      _libc_free((unsigned __int64)v42);
    return;
  }
LABEL_31:
  sub_2E21C60(a1, *(_QWORD *)(a2 + 32), v6);
  v31 = *(unsigned int **)(v6 + 96);
  for ( j = *(unsigned int **)(v6 + 104); j != v31; v31 += 3 )
  {
    v33 = *(_DWORD *)(*(_QWORD *)(*a1 + 8) + 24LL * *v31 + 16) & 0xFFF;
    v34 = *(_QWORD *)(*a1 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(*a1 + 8) + 24LL * *v31 + 16) >> 12);
    do
    {
      if ( !v34 )
        break;
      v34 += 2;
      *(_QWORD *)(a1[1] + 8LL * (v33 >> 6)) &= ~(1LL << v33);
      v33 += *(__int16 *)(v34 - 2);
    }
    while ( *(_WORD *)(v34 - 2) );
  }
}
